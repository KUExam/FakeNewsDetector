import re
from cleantext import clean

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

import pandas as pd
import numpy as np
url='https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv?raw=true'
df = pd.read_csv(url, index_col=0)

df['content']=df['content'].apply(clean_text_with_cleantext)
df.head()




# from nltk.corpus import stopwords
# url1=''
# df1 = pd.read_csv(url1)
# tokens = nltk.word_tokenize(df)


# from nltk.corpus import stopwords
# url1=''
# df1 = pd.read_csv(url1)
# tokens = nltk.word_tokenize(df)


import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load the data
url1 = 'https://raw.githubusercontent.com/KUExam/FakeNewsDetector/main/cleanedsample1.csv'
df1 = pd.read_csv(url1, index_col=0)

# Define the stopwords
stop_words = set(stopwords.words('english'))

# Ensure 'content' column contains only strings
df1['content'] = df1['content'].astype(str)

# Tokenize first, then remove stopwords
df1['tokenized_content'] = df1['content'].apply(word_tokenize)

# Compute the size of the vocabulary before removing stopwords
vocab_before = len(set(word.lower() for content in df1['tokenized_content'] for word in content))

# Apply stopwords removal
df1['filtered_content'] = df1['tokenized_content'].apply(lambda x: [word for word in x if word.lower() not in stop_words])

# Flatten the list of lists into a single list after removing stopwords
flat_list_after = [word for sublist in df1['filtered_content'] for word in sublist]

# Compute the size of the vocabulary after removing stopwords
vocab_after = len(set(word.lower() for word in flat_list_after))

# Compute the reduction rate of the vocabulary size
reduction_rate = (vocab_before - vocab_after) / vocab_before

print(f"Vocabulary size before removing stopwords: {vocab_before}")
print(f"Vocabulary size after removing stopwords: {vocab_after}")
print(f"Reduction rate of the vocabulary size: {reduction_rate:.2%}")
