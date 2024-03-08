import pandas as pd
import re
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load dataset and preprocess
df = pd.read_csv("FakeNews_2000rows.csv", usecols=['id', 'domain', 'type', 'url', 'content', 'scraped_at', 'title', 'tags', 'authors'])
df = df[df['id'].apply(lambda x: str(x).isdigit())]
df['content'] = df['content'].fillna('')
# Remove all wikileaks.org articles that start with 'Tor'
df = df.loc[~((df['domain'] == 'wikileaks.org') & df['content'].str.startswith('Tor'))]
df = df[df['type'] != 'unknown']

# Improved Tokenization and Preprocessing Function
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

def preprocess_text(text):
    text = replace_tokens(text)
    tokens = tokenizer.tokenize(text.lower())
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)


# Apply preprocessing
df['processed_content'] = df['content'].apply(preprocess_text)

# Apply the token replacement function
df['content'] = df['content'].apply(replace_tokens)

# Tokenize the content
df['tokenized_content'] = df['content'].apply(lambda x: tokenizer.tokenize(x))

# Perform sentiment analysis on the modified 'content'
df['sentiment'] = df['content'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Filter out stopwords and stem the remaining words
df['filtered_content'] = df['tokenized_content'].apply(lambda x: [stemmer.stem(word) for word in x if word.lower() not in stop_words])

# Count the number of each article type
article_type_counts = df['type'].value_counts()

# Frequency analysis
word_counts_before = Counter([word for row in df['tokenized_content'] for word in row])
word_counts_after = Counter([word for row in df['filtered_content'] for word in row])

# Article length and sentiment analysis by article type
df['article_length'] = df['content'].apply(lambda x: len(x.split()))
aggregated_data = df.groupby('type').agg({
    'sentiment': ['mean', 'median'],
    'article_length': ['mean', 'median']
})

print("\nAverage and Median of Sentiment and Article Length by Article Type:")
print(aggregated_data)

most_common_words = word_counts_after.most_common(10000)  # Get the top 10,000, but we'll plot only the top 100

# Plotting the top 50 for visualization
top_words = most_common_words[:50]  # Adjust this slice for different numbers
words = [word for word, freq in top_words]
frequencies = [freq for word, freq in top_words]

plt.figure(figsize=(10, 8))
plt.barh(words[::-1], frequencies[::-1])  # Reverse to have the highest frequency on top
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.title('Top 100 Most Frequent Words')
plt.show()

# Print URL, NUM, and DATE counts
url_count = df['content'].str.count('<URL>').sum()
num_count = df['content'].str.count('<NUM>').sum()
print(f"URL count: {url_count}")
print(f"Numeric values count: {num_count}")

# 100 most frequent words before and after processing
print("\n100 most frequent words BEFORE processing:")
print(word_counts_before.most_common(100))

print("\n100 most frequent words AFTER processing:")
print(word_counts_after.most_common(100))

# Display the counts
print(article_type_counts)