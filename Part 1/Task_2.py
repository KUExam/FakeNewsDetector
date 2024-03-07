import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import re
from cleantext import clean

# Assuming nltk.download('punkt') and nltk.download('stopwords') have been executed earlier
def clean_text_with_cleantext(text):
    date_pattern = re.compile(r'''
    \b(?:[jJ]an(?:uary)?|
    [fF]eb(?:ruary)?|
    [mM]ar(?:ch)?|
    [aA]pr(?:il)?|
    [mM]ay|
    [jJ]un(?:e)?|
    [jJ]ul(?:y)?|
    [Aa]ug(?:ust)?|
    [sS]ep(?:tember)?|
    [oO]ct(?:ober)?|
    [nN]ov(?:ember)?|
    [dD]ec(?:ember)?)\s\d{1,2},?\s\d{4}\b|
    \b\d{4}-\d{2}-\d{2}\b
    ''', re.X)
    text = re.sub(date_pattern, '<DATE>', text)
    cleaned_text = clean(
        text,
        fix_unicode=True,               # fix various unicode errors
        to_ascii=True,                  # transliterate to closest ASCII representation
        lower=True,                     # lowercase text
        normalize_whitespace=True,      # Removes multiple whitespaces
        no_line_breaks=True,            # fully strip line breaks as opposed to only normalizing them
        strip_lines=True,               # 
        keep_two_line_breaks=False,     #
        no_urls=True,                   # replace all URLs with a special token
        no_emails=True,                 # replace all email addresses with a special token
        no_numbers=True,                # replace all numbers with a special token
        no_digits=False,                # replace all digits with a special token
        no_currency_symbols=False,      # replace all currency symbols with a special token
        no_punct=False,                 # remove punctuations
        no_emoji=True,                  # Emoji remover
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_number="<NUM>",
        replace_with_punct="",
        lang="en"                       # Language, set to 'en' for English special handling
    )

    return cleaned_text

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer  # Import the PorterStemmer

# CSV file name
csv_file_name = "995,000_rows.csv"

# Initialize NLTK's PorterStemmer and define stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Initialize accumulators
word_counts_before = Counter()
word_counts_after = Counter()

# Read the CSV file
df = pd.read_csv(csv_file_name)

# Apply cleaning to the 'content' column
df['cleaned_content'] = df['content'].astype(str).apply(clean_text_with_cleantext)

# Tokenize
df['tokenized_content'] = df['cleaned_content'].apply(word_tokenize)

# Count URLs, Dates, and Numbers before removing stopwords and applying stemming
df['url_count'] = df['cleaned_content'].apply(lambda x: x.count("<URL>"))
df['date_count'] = df['cleaned_content'].apply(lambda x: x.count("<DATE>"))
df['num_count'] = df['cleaned_content'].apply(lambda x: x.count("<NUM>"))

# Process each row for word frequency before and after preprocessing
for _, row in df.iterrows():
    tokenized_content = row['tokenized_content']
    
    # Before removing stopwords and stemming
    word_counts_before.update(tokenized_content)
    
    # After removing stopwords and stemming
    filtered_content = [stemmer.stem(word) for word in tokenized_content if word.lower() not in stop_words]
    word_counts_after.update(filtered_content)

# Determine the 100 most frequent words before and after
most_common_words_before = word_counts_before.most_common(100)
most_common_words_after = word_counts_after.most_common(100)

# Plotting the frequency of the 10,000 most frequent words before and after
words_before, frequencies_before = zip(*word_counts_before.most_common(10000))
words_after, frequencies_after = zip(*word_counts_after.most_common(10000))

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(frequencies_before)
plt.title('Before Preprocessing')
plt.yscale('log')
plt.xlabel('Word Rank')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.plot(frequencies_after)
plt.title('After Preprocessing')
plt.yscale('log')
plt.xlabel('Word Rank')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
