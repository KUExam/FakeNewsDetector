import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import re

# Load the dataset
csv_file_name = "FakeNews_2000rows.csv"
columns_of_interest = ['id', 'domain', 'type', 'url', 'content', 'scraped_at', 'title', 'tags', 'authors']

df = pd.read_csv(csv_file_name, usecols=columns_of_interest)

# Replace NaN values in the 'content' column with empty strings directly
df['content'] = df['content'].fillna('')

# Function to replace tokens in text
def replace_tokens(text):
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '<URL>', text)  # URLs
    text = re.sub(r'\b\d+\b', '<NUM>', text)  # Numbers
    text = re.sub(r'\S*@\S*\s?', '<EMAIL>', text)  # Emails
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)  # Remove non-alphanumeric characters, keep special tokens
    return text

# Apply the token replacement function
df['content'] = df['content'].apply(replace_tokens)

def clean_text_with_cleantext(text):
    tokenizer = re.compile(r'\w+|<URL>|<EMAIL>|<NUM>|<PUNCT>|\.|\,|\!|\?|\;|\:|\"|\'')
    tokens = tokenizer.findall(text)
    return " ".join(tokens)

# Apply text replacement and cleaning
df['cleaned_content'] = df['content'].apply(clean_text_with_cleantext)
df['tokenized_content'] = df['cleaned_content'].apply(word_tokenize)

# Apply the token replacement function
df['content'] = df['content'].apply(replace_tokens)

# Perform sentiment analysis directly on the modified 'content' (which has had replacements)
df['sentiment'] = df['content'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Initialize NLTK's PorterStemmer and a list of stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Tokenize the content for further analysis
df['tokenized_content'] = df['content'].apply(word_tokenize)

# Filter out stopwords and stem the remaining words
df['filtered_content'] = df['tokenized_content'].apply(lambda x: [stemmer.stem(word) for word in x if word.lower() not in stop_words])

# Process text for frequency analysis before and after removing stopwords and stemming
word_counts_before = Counter([word for row in df['tokenized_content'] for word in row])
filtered_content = df['tokenized_content'].apply(lambda x: [stemmer.stem(word) for word in x if word.lower() not in stop_words])
word_counts_after = Counter([word for row in filtered_content for word in row])

# Print URL, Date, and NUM counts
url_count = df['content'].str.count('<URL>').sum()
num_count = df['content'].str.count('<NUM>').sum()
date_count = df['content'].str.count('<DATE>').sum()
print(f"URL count: {url_count}")
print(f"Date count: {date_count}")
print(f"Numeric values count: {num_count}")

# Print 100 most frequent words before and after processing
print("\n100 most frequent words BEFORE processing:")
print(word_counts_before.most_common(100))

print("\n100 most frequent words AFTER processing:")
print(word_counts_after.most_common(100))

# Count occurrences of special tokens in the 'content' AFTER replacements have been made
url_count = df['content'].str.count('<URL>').sum()
num_count = df['content'].str.count('<NUM>').sum()

print(f"URL count: {url_count}")
print(f"Numeric values count: {num_count}")

# Calculate average (mean) and median of sentiment and article length for each type of article
df['article_length'] = df['content'].apply(lambda x: len(x.split()))
aggregated_data = df.groupby('type').agg({
    'sentiment': ['mean', 'median'],
    'article_length': ['mean', 'median']
})

print("\nAverage and Median of Sentiment and Article Length by Article Type:")
print(aggregated_data)