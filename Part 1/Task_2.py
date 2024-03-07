import pandas as pd
import numpy as np
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import re
from cleantext import clean
from textblob import TextBlob
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.decomposition import LatentDirichletAllocation
# import spacy  # Uncomment if using Spacy for NER

# Assuming necessary NLTK downloads have been executed earlier
csv_file_name = "995,000_rows.csv"

# Initialization
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Read the CSV file
df = pd.read_csv(csv_file_name)

# Clean the 'content' column
df['cleaned_content'] = df['content'].astype(str).apply(clean_text_with_cleantext)

# Tokenize
df['tokenized_content'] = df['cleaned_content'].apply(word_tokenize)

# Initialize counters for URLs, Dates, and Numbers
df['url_count'] = df['cleaned_content'].apply(lambda x: x.count("<URL>"))
df['date_count'] = df['cleaned_content'].apply(lambda x: x.count("<DATE>"))
df['num_count'] = df['cleaned_content'].apply(lambda x: x.count("<NUM>"))

# Perform sentiment analysis
df['sentiment'] = df['cleaned_content'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Calculate article length
df['article_length'] = df['tokenized_content'].apply(len)

# Preprocess text for frequency analysis before and after removing stopwords and stemming
word_counts_before = Counter([word for row in df['tokenized_content'] for word in row])
filtered_content = df['tokenized_content'].apply(lambda x: [stemmer.stem(word) for word in x if word.lower() not in stop_words])
word_counts_after = Counter([word for row in filtered_content for word in row])

# Plotting the frequency of the 10,000 most frequent words before and after preprocessing
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
words_before, frequencies_before = zip(*word_counts_before.most_common(10000))
plt.plot(frequencies_before)
plt.title('Before Preprocessing')
plt.yscale('log')
plt.xlabel('Word Rank')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
words_after, frequencies_after = zip(*word_counts_after.most_common(10000))
plt.plot(frequencies_after)
plt.title('After Preprocessing')
plt.yscale('log')
plt.xlabel('Word Rank')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Additional steps for Topic Modeling and NER could be included here,
# commented out due to their complexity and additional setup requirements.
