import csv
from cleantext import clean

def clean_text_with_cleantext(text):
    date_pattern = r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b'
    text = re.sub(date_pattern, '<DATE>', text)
    cleaned_text = clean(text,
                         fix_unicode=True,               # fix various unicode errors
                         to_ascii=True,                  # transliterate to closest ASCII representation
                         lower=True,                     # lowercase text
                         no_line_breaks=True,            # fully strip line breaks as opposed to only normalizing them
                         normalize_whitespace=True,           
                         no_urls=True,                   # replace all URLs with a special token
                         no_emails=True,                 # replace all email addresses with a special token
                         no_numbers=True,                # replace all numbers with a special token
                         no_punct=True,                  # remove punctuations
                         replace_with_url="<URL>",       # relpacing URLS with <URL>
                         replace_with_email="<EMAIL>",   # replacing emails with <EMAIL>
                         replace_with_number="<NUM>",
                         lang="en"),
    

    return cleaned_text


import pandas as pd
import numpy as np
df = pd.read_csv('news_sample.csv')

df['content']=df['content'].apply(clean_text_with_cleantext)
df.head()
df.to_csv("cleanednews.csv")