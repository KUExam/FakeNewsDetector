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

def replace_tokens(text):
    text = re.sub(r'https?://\S+|www\.\S+', '<URL>', text)  # URLs
    text = re.sub(r'\S*@\S*\s?', '<EMAIL>', text)  # Emails
    text = re.sub(r'\d+', '<NUM>', text)  # Numbers
    text = re.sub(r'[\w\.-]+@[\w\.-]+', '<EMAIL>', text)  # More complex emails
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)  # Remove punctuations but keep special tokens
    return text

# Apply the token replacement and cleaning functions
df['content'] = df['content'].apply(replace_tokens)

def clean_text_with_cleantext(text):
    tokenizer = re.compile(r'\w+|<URL>|<EMAIL>|<NUM>|<PUNCT>|\.|\,|\!|\?|\;|\:|\"|\'')
    tokens = tokenizer.findall(text)
    return " ".join(tokens)

# Apply text replacement and cleaning
df['cleaned_content'] = df['content'].apply(clean_text_with_cleantext)
df['tokenized_content'] = df['cleaned_content'].apply(word_tokenize)

# Perform sentiment analysis on the cleaned content
df['sentiment'] = df['cleaned_content'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Initialize NLTK's PorterStemmer and stopwords list
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Process text for frequency analysis before and after removing stopwords and stemming
word_counts_before = Counter([word for row in df['tokenized_content'] for word in row])
filtered_content = df['tokenized_content'].apply(lambda x: [stemmer.stem(word) for word in x if word.lower() not in stop_words])
word_counts_after = Counter([word for row in filtered_content for word in row])

# Print URL, Date, and NUM counts
url_count = df['content'].str.count('<URL>').sum()
date_count = df['content'].str.count('<DATE>').sum()
num_count = df['content'].str.count('<NUM>').sum()
print(f"URL count: {url_count}")
print(f"Date count: {date_count}")
print(f"Numeric values count: {num_count}")

# Print 100 most frequent words before and after processing
print("\n100 most frequent words BEFORE processing:")
print(word_counts_before.most_common(100))

print("\n100 most frequent words AFTER processing:")
print(word_counts_after.most_common(100))

# Adjust the size of the plots
article_types = df['type'].dropna().unique()
num_types = len(article_types)

# Plot for 'before preprocessing'
plt.figure(figsize=(20, 10))
words_before, freqs_before = zip(*word_counts_before.most_common(10000))
plt.subplot(1, 2, 1)
sns.lineplot(x=range(len(freqs_before)), y=freqs_before)
plt.title('Word Frequencies BEFORE Preprocessing')
plt.yscale('log')
plt.xlabel('Word Rank')
plt.ylabel('Frequency')

words_after, freqs_after = zip(*word_counts_after.most_common(10000))
plt.subplot(1, 2, 2)
sns.lineplot(x=range(len(freqs_after)), y=freqs_after)
plt.title('Word Frequencies AFTER Preprocessing')
plt.yscale('log')
plt.xlabel('Word Rank')
plt.ylabel('Frequency')

# Separate graphs for sentiment by article type
for article_type in article_types:
    plt.figure(figsize=(10, 4))
    sns.histplot(df[df['type'] == article_type]['sentiment'], bins=20, kde=False, color='skyblue')
    plt.title(f'Sentiment Distribution for {article_type} Articles')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.show()

for article_type in article_types:
    plt.figure(figsize=(10, 4))
    sns.histplot(df[df['type'] == article_type]['content'].apply(lambda x: len(x.split())), bins=20, color='salmon')
    plt.title(f'Article Length Distribution: {article_type}')
    plt.xlabel('Article Length (words)')
    plt.ylabel('Frequency')
    plt.show()

for i, word in enumerate(words_before[:20]):
    plt.text(i, freqs_before[i], word, horizontalalignment='left', size='small', color='black', weight='semibold')

plt.tight_layout()
plt.show()