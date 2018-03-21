# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from langdetect import detect
from googletrans import Translator

def translate_data():
    meta = pd.read_csv('../input/train.csv')
    meta['lang'] = meta['comment_text'].apply(_safe_detect)
    meta['comment_text_english'] = meta.apply(_translate_non_english, axis=1)
    train_clean_english = meta[['id','comment_text_english']]
    train_clean_english.rename(columns={'comment_text_english':'comment_text'}, inplace=True)
    train_clean_english.to_csv('../input/train_clean_english.csv', index=None)


def _safe_detect(text):
    try:
        lang = detect(text)
    except Exception:
        lang = 'en'
    return lang


def _translate_non_english(row):
    translator = Translator()
    lang = str(row['lang'])
    comment = str(row['comment_text'])
    if lang != 'en':
        try:
            english_comment = translator.translate(comment, dest='en').text
        except Exception:
            english_comment = comment
    else:
        english_comment = comment
    return english_comment

if  __name__=="__main__":
    translate_data()
