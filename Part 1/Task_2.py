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

for chunk in pd.read_csv("FakeNews_2000rows.csv", usecols=['id', 'domain', 'type', 'url', 'content', 'scraped_at', 'title', 'tags', 'authors'], chunksize=chunk_size):
    # Fill missing content in the current chunk
    chunk['content'] = chunk['content'].fillna('')
    
    # Apply the token replacement function to the current chunk
    chunk['content'] = chunk['content'].apply(replace_tokens)
    
    # Remove all wikileaks.org articles that start with 'Tor'
    chunk = chunk.loc[~((chunk['domain'] == 'wikileaks.org') & chunk['content'].str.startswith('Tor'))]
    
    # Remove articles where 'type' is 'unknown' and drop NaNs in 'type'
    chunk = chunk[chunk['type'] != 'unknown']
    chunk.dropna(subset=['type'], inplace=True)

    # Create the 'article_length' column
    chunk['article_length'] = chunk['content'].apply(lambda x: len(x.split()))
    
    # Filter based on 'article_length'
    chunk = chunk[chunk['article_length'] >= 0]

    # Assign category based on type
    chunk['category'] = chunk['type'].apply(assign_category)

    # Remove rows with types that don't fall into our defined categories
    chunk = chunk[chunk['category'] != 'delete']

    # Tokenize the content in the current chunk
    chunk['tokenized_content'] = chunk['content'].apply(lambda x: tokenizer.tokenize(x))
    
    # Perform sentiment analysis on the modified 'content' of the current chunk
    chunk['sentiment'] = chunk['content'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
    # Filter out stopwords and stem the remaining words in the current chunk
    chunk['filtered_content'] = chunk['tokenized_content'].apply(lambda x: [stemmer.stem(word) for word in x if word.lower() not in stop_words])
    
    # Process content by tokenizing, stemming, and removing stopwords in the current chunk
    chunk['processed_content'] = chunk['content'].apply(preprocess_text)
    
    # Append the processed chunk to the list
    chunk_list.append(chunk)

# Concatenate all processed chunks to form the full DataFrame
df = pd.concat(chunk_list)

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

# Bar plot for article type distribution
plt.figure(figsize=(10, 6))
article_type_counts.plot(kind='bar')
plt.xlabel('Article Type')
plt.ylabel('Count')
plt.title('Distribution of Article Types')
plt.xticks(rotation=45)
plt.show()

# Box plot for sentiment distribution by article type
plt.figure(figsize=(10, 6))
df.boxplot(column='sentiment', by='type', rot=45)
plt.title('Sentiment Distribution by Article Type')
plt.xlabel('Article Type')
plt.ylabel('Sentiment Score')
plt.show()

# Violin plot for article length distribution by article type
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='type', y='article_length')
plt.ylim(0, None)
plt.title('Article Length Distribution by Article Type')
plt.xlabel('Article Type')
plt.ylabel('Article Length')
plt.xticks(rotation=45)
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

if df.isnull().any().any():
    print("NaN values found. Please check and clean your DataFrame.")
else:
    print("No NaN values found. Proceeding with the train-test split.")

# Split the data into training, validation, and test sets
train_data, test_data, train_labels, test_labels = train_test_split(
    df[['processed_content', 'id', 'domain', 'type', 'url', 'scraped_at', 'title', 'tags', 'authors', 'category']],
    df['type'],
    test_size=0.2,
    random_state=42,
    stratify=df['type']
)

# Split the test set into validation and test sets
val_data, test_data, val_labels, test_labels = train_test_split(
    test_data,
    test_labels,
    test_size=0.5,
    random_state=42,
    stratify=test_labels
)

# Create new DataFrames for each split
train_df = pd.DataFrame(train_data)
train_df['type'] = train_labels

val_df = pd.DataFrame(val_data)
val_df['type'] = val_labels

test_df = pd.DataFrame(test_data)
test_df['type'] = test_labels

# Save each split as a separate CSV file
train_df.to_csv('train_data.csv', index=False)
val_df.to_csv('val_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)

print(f"Training set: {len(train_df)} samples")
print(f"Validation set: {len(val_df)} samples")
print(f"Test set: {len(test_df)} samples")

# Reset index of df after all preprocessing to ensure alignment
df.reset_index(drop=True, inplace=True)

# Initialize TF-IDF Vectorizer without stopwords to simplify the demonstration
tfidf = TfidfVectorizer(stop_words='english', max_features=1000)

# Fit and transform the processed content
tfidf_matrix = tfidf.fit_transform(df['processed_content'])

# Feature names
feature_names = tfidf.get_feature_names_out()

# Article types
article_types = df['type'].unique()

# Dictionary to hold top words for each article type
top_words_per_type = {}

# Initialize and apply TF-IDF vectorization
for article_type in article_types:
    # Find indices of articles of the current type
    indices = df[df['type'] == article_type].index
    
    # Ensure indices are not empty
    if len(indices) > 0:
        # TF-IDF subset for current type
        tfidf_subset = tfidf_matrix[indices]
        
        # Ensure the subset is not empty
        if isinstance(tfidf_subset, csr_matrix) and tfidf_subset.shape[0] > 0:
            # Mean scores across all documents of the current type
            mean_scores = np.mean(tfidf_subset, axis=0).tolist()[0]
            # Mapping scores to feature names
            scores_features = zip(mean_scores, feature_names)
            # Sorting and selecting top 10
            sorted_scores_features = sorted(scores_features, key=lambda x: x[0], reverse=True)[:10]
            top_words = [feature for _, feature in sorted_scores_features]
            top_words_per_type[article_type] = top_words
        else:
            top_words_per_type[article_type] = ['No articles found']
    else:
        top_words_per_type[article_type] = ['No articles found']

# Print top words for each article type based on TF-IDF analysis
for article_type, words in top_words_per_type.items():
    print(f"Top 10 words for {article_type}: {words}")
