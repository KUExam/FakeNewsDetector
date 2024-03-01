import re
from cleantext import clean

def clean_text_with_cleantext(text):
    date_pattern = r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b'
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
df.to_csv("cleanedsample_news.csv")
