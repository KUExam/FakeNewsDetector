import pandas as pd
import re
from collections import Counter
import seaborn as sns
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import TextBlob
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
tqdm.pandas()

chunk_size = 2000
tokenizer = RegexpTokenizer(r'\w+')
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Function to replace tokens in text
def replace_tokens(text):
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '<URL>', text)  # URLs
    text = re.sub(r'\b\d+\b', '<NUM>', text)  # Numbers
    text = re.sub(r'\S*@\S*\s?', '<EMAIL>', text)  # Emails
    date_patterns = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # DD-MM-YYYY or MM-DD-YYYY
        r'\b\d{2,4}[/-]\d{1,2}[/-]\d{1,2}\b',  # YYYY-MM-DD
        r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s\d{1,2},?\s\d{4}\b',  # Month DD, YYYY
        r'\b\d{1,2}\s(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s\d{4}\b',  # DD Month YYYY
    ]
    for pattern in date_patterns:
        text = re.sub(pattern, '<DATE>', text)
    return text

# Preprocess content by tokenizing, stemming, and removing stopwords
def preprocess_text(text):
    text = replace_tokens(text)
    tokens = tokenizer.tokenize(text.lower())
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)


chunk_list1 = []  # List to hold processed chunks


# Read the new dataset
for chunk in pd.read_csv("scraped_articles.csv", usecols=[ 'content', 'datetime', 'header', 'author'], chunksize=chunk_size):
    # Fill missing content in the current chunk
    chunk['content'] = chunk['content'].fillna('')
    
    # Apply the token replacement function to the current chunk
    chunk['content'] = chunk['content'].progress_apply(replace_tokens)
    
    # Since the new data doesn't have a 'tags' column, we'll create one with empty strings
    chunk['tags'] = ''
    chunk['domain'] = 'BBC.com'
    chunk['url'] = 'bbc'
    chunk['id'] = ''
    chunk['type'] = 'reliable'
    chunk['category'] = 'reliable'

    # Create the 'article_length' column
    chunk['article_length'] = chunk['content'].progress_apply(lambda x: len(x.split()))
    
    # Filter based on 'article_length'
    chunk = chunk[chunk['article_length'] >= 0]

   
    # Remove rows with types that don't fall into our defined categories
    chunk = chunk[chunk['category'] != 'delete']

    # Tokenize the content in the current chunk
    chunk['tokenized_content'] = chunk['content'].progress_apply(lambda x: tokenizer.tokenize(x))
    
    # Filter out stopwords and stem the remaining words in the current chunk
    chunk['filtered_content'] = chunk['tokenized_content'].progress_apply(lambda x: [stemmer.stem(word) for word in x if word.lower() not in stop_words])
    
    # Process content by tokenizing, stemming, and removing stopwords in the current chunk
    chunk['processed_content'] = chunk['content'].progress_apply(preprocess_text)


    # Rename 'datetime' to 'scraped_at' and 'header' to 'title'
    chunk.rename(columns={'datetime': 'scraped_at', 'header': 'title', 'authour': 'authors', 'content':'processed_content'}, inplace=True)
    
    # Append the processed chunk to the list
    chunk_list1.append(chunk)

# Concatenate all processed chunks to form the full DataFrame
df1 = pd.concat(chunk_list1)

df1.to_csv('assignment2_data.csv' , index=False)
