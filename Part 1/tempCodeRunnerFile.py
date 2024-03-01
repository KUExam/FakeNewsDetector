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



# import nltk
# from nltk.corpus import stopwords
# url1=''
# df1 = pd.read_csv(url1)
# tokens = nltk.word_tokenize(df)



from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
 
url1='https://raw.githubusercontent.com/KUExam/FakeNewsDetector/main/cleanedsample1.csv?token=GHSAT0AAAAAACO6SV2TCNQ75WTM2HO2ZAN4ZPBXEJA'
df1 = pd.read_csv(url1, index_col=0)

stop_words = set(stopwords.words('english'))
 
word_tokens = word_tokenize(df1['content'])
# converts the words in word_tokens to lower case and then checks whether 
#they are present in stop_words or not
filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
#with no lower case conversion
filtered_sentence = []
 
for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)

print(filtered_sentence[:10])
