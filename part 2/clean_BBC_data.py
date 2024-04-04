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

def assign_category(article_type):
    if article_type in ['reliable', 'political', 'clickbait']:
        return 'reliable'
    elif article_type in ['fake', 'satire', 'bias', 'conspiracy', 'junksci', 'hate', 'unreliable']:
        return 'fake'
    else:
        return 'delete'  # We'll use this to filter out unwanted types

chunk_list = []  # List to hold processed chunks

for chunk in pd.read_csv("Assigment2_data.csv", usecols=['id', 'domain', 'type', 'url', 'content', 'scraped_at', 'title', 'tags', 'authors'], chunksize=chunk_size):
    # Fill missing content in the current chunk
    chunk['content'] = chunk['content'].fillna('')
    
    # Apply the token replacement function to the current chunk
    chunk['content'] = chunk['content'].progress_apply(replace_tokens)
    
    # Remove all wikileaks.org articles that start with 'Tor'
    chunk = chunk.loc[~((chunk['domain'] == 'wikileaks.org') & chunk['content'].str.startswith('Tor'))]
    
    # Remove articles where 'type' is 'unknown' and drop NaNs in 'type'
    chunk = chunk[chunk['type'] != 'unknown']
    chunk.dropna(subset=['type'], inplace=True)

    # Create the 'article_length' column
    chunk['article_length'] = chunk['content'].progress_apply(lambda x: len(x.split()))
    
    # Filter based on 'article_length'
    chunk = chunk[chunk['article_length'] >= 0]

    # Assign category based on type
    chunk['category'] = chunk['type'].progress_apply(assign_category)

    # Remove rows with types that don't fall into our defined categories
    chunk = chunk[chunk['category'] != 'delete']

    # Tokenize the content in the current chunk
    chunk['tokenized_content'] = chunk['content'].progress_apply(lambda x: tokenizer.tokenize(x))
    
    # Perform sentiment analysis on the modified 'content' of the current chunk
    chunk['sentiment'] = chunk['content'].progress_apply(lambda x: TextBlob(x).sentiment.polarity)
    
    # Filter out stopwords and stem the remaining words in the current chunk
    chunk['filtered_content'] = chunk['tokenized_content'].progress_apply(lambda x: [stemmer.stem(word) for word in x if word.lower() not in stop_words])
    
    # Process content by tokenizing, stemming, and removing stopwords in the current chunk
    chunk['processed_content'] = chunk['content'].progress_apply(preprocess_text)
    
    # Append the processed chunk to the list
    chunk_list.append(chunk)

# Concatenate all processed chunks to form the full DataFrame
df = pd.concat(chunk_list)

article_type_counts = df['type'].value_counts()




if df.isnull().any().any():
    print("NaN values found. Please check and clean your DataFrame.")
else:
    print("No NaN values found. Proceeding with the train-test split.")


df.to_csv('assigment2_dataclean.csv', index = False)

