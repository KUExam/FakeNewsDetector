import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer  # Import the PorterStemmer

# Load the data
url = 'https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv'
df = pd.read_csv(url, index_col=0)

# Define the stopwords
stop_words = set(stopwords.words('english'))

# Ensure 'content' column contains only strings
df['content'] = df['content'].astype(str)

# Tokenize first, then remove stopwords
df['tokenized_content'] = df['content'].apply(word_tokenize)

# Compute the size of the vocabulary before removing stopwords
vocab_before = len(set(word.lower() for content in df['tokenized_content'] for word in content))

# Apply stopwords removal
df['filtered_content'] = df['tokenized_content'].apply(lambda x: [word for word in x if word.lower() not in stop_words])

# Initialize the PorterStemmer
stemmer = PorterStemmer()

# Apply stemming to each word in the filtered content
df['stemmed_content'] = df['filtered_content'].apply(lambda x: [stemmer.stem(word) for word in x])

# Flatten the list of lists into a single list after removing stopwords
flat_list_after_stopwords = [word for sublist in df['filtered_content'] for word in sublist]

# Correctly flatten the list of lists into a single list after stemming
flat_list_after_stemming = [word for sublist in df['stemmed_content'] for word in sublist]

# Compute the size of the vocabulary after removing stopwords
vocab_after = len(set(word.lower() for word in flat_list_after_stopwords))

# Compute the size of the vocabulary after stemming using the correctly stemmed list
vocab_after_stemming = len(set(word.lower() for word in flat_list_after_stemming))

# Compute the reduction rate of the vocabulary size after stemming
reduction_rate_after_stemming = (vocab_after - vocab_after_stemming) / vocab_after

print(f"Vocabulary size before removing stopwords: {vocab_before}")
print(f"Vocabulary size after removing stopwords: {vocab_after}")
print(f"Vocabulary size after stemming: {vocab_after_stemming}")
print(f"Reduction rate of the vocabulary size after stemming: {reduction_rate_after_stemming:.2%}")