# features used in hw1
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion
import nltk
from nltk.tokenize import word_tokenize
import csv

# # Data gathering/preprocessing
# # sentiment polarity of the tweet: “negative” (0), “positive” (1), “neutral” (2).
# def label_convert(value):
#     if value == 'neutral' or value == 'objective':
#         return 2
#     elif value == 'negative':
#         return 0
#     elif value == 'positive':
#         return 1
#
# # read data, using Sentiment140
# train_data = pd.read_csv('data/train.csv', error_bad_lines=False, converters={'label': label_convert})

pos_tag = ['N','O','S','^','Z','L','M','V','A','R','!','D','P','&','T','X','Y','#','@','~','U','E','$',',','G']

emoji_helper = [':', ';', '"', '<', '^', '_', '.']

def pos_feature(all_value):

    pos_all = []
    for value in all_value:
        tokens = nltk.word_tokenize(value)
        pos = [0] * 25
        for item in tokens:
            for tag in pos_tag:
                if item == tag:
                    pos[pos_tag.index(tag)] = pos[pos_tag.index(tag)] + 1
        pos_all.append(pos)

    return pos_all

def find_emoji(word):
    for char in word:
        if char in emoji_helper and len(word) > 1:
            return True
    for char in word:
        if char.isalpha() or char.isnumeric():
            return False
    if len(word) < 2:
        return False
    return True

# features: total count of tokens with score(w, p) > 0 using unigran
def count_positive_uni(all_value, uni_dict):

    count_all = []
    for value in all_value:
        tokens = nltk.word_tokenize(value)
        count = 0
        for item in tokens:
            if item in uni_dict and uni_dict[item] > 0 :
                count = count + 1
        count_all.append([count])

    return count_all

# features: total count of tokens with score(w, p) < 0 using unigran
def count_negative_uni(all_value, uni_dict):

    count_all = []
    for value in all_value:
        tokens = nltk.word_tokenize(value)
        count = 0
        for item in tokens:
            if item in uni_dict and uni_dict[item] < 0 :
                count = count + 1
        count_all.append([count])

    return count_all

# features: total score of tokens with score(w, p) > 0 using unigran
def sum_positive_uni(all_value, uni_dict):

    sum_all = []
    for value in all_value:
        tokens = nltk.word_tokenize(value)
        sum = 0
        for item in tokens:
            if item in uni_dict and uni_dict[item] > 0 :
                sum = sum + uni_dict[item]
        sum_all.append([sum])

    return sum_all

# features: total score of tokens with score(w, p) < 0 using unigran
def sum_negative_uni(all_value, uni_dict):

    sum_all = []
    for value in all_value:
        tokens = nltk.word_tokenize(value)
        sum = 0
        for item in tokens:
            if item in uni_dict and uni_dict[item] < 0 :
                sum = sum + uni_dict[item]
        sum_all.append([sum])

    return sum_all

# features: max score of tokens with score(w, p) > 0 using unigran
def max_positive_uni(all_value, uni_dict):

    max_all = []
    for value in all_value:
        tokens = nltk.word_tokenize(value)
        max = 0
        for item in tokens:
            if item in uni_dict and uni_dict[item] > 0 :
                if uni_dict[item] > max:
                    max = uni_dict[item]
        max_all.append([max])

    return max_all

# features: min score of tokens with score(w, p) < 0 using unigran
def min_negative_uni(all_value, uni_dict):

    min_all = []
    for value in all_value:
        tokens = nltk.word_tokenize(value)
        min = 0
        for item in tokens:
            if item in uni_dict and uni_dict[item] < 0 :
                if uni_dict[item] < min:
                    min = uni_dict[item]
        min_all.append([min])

    return min_all

# features: last tokens score with score(w, p) > 0 using unigran
def last_positive_uni(all_value, uni_dict):

    last_all = []
    for value in all_value:
        tokens = nltk.word_tokenize(value)
        last = 0
        for item in tokens:
            if item in uni_dict and uni_dict[item] > 0 :
                last = uni_dict[item]
        last_all.append([last])

    return last_all

# features: last tokens score with score(w, p) < 0 using unigran
def last_negative_uni(all_value, uni_dict):

    last_all = []
    for value in all_value:
        tokens = nltk.word_tokenize(value)
        last = 0
        for item in tokens:
            if item in uni_dict and uni_dict[item] < 0 :
                last = uni_dict[item]
        last_all.append([last])

    return last_all

# features: total count of tokens with score(w, p) > 0 using bigrams
def count_positive_bi(all_value, bi_dict):

    count_all = []
    for value in all_value:
        tokens = nltk.word_tokenize(value)
        bigrm = nltk.bigrams(tokens)
        count = 0
        for item in bigrm:
            if (item[0]+' '+item[1]) in bi_dict and bi_dict[item[0]+' '+item[1]] > 0 :
                count = count + 1
        count_all.append([count])

    return count_all

# features: total count of tokens with score(w, p) < 0 using bigrams
def count_negative_bi(all_value, bi_dict):

    count_all = []
    for value in all_value:
        tokens = nltk.word_tokenize(value)
        bigrm = nltk.bigrams(tokens)
        count = 0
        for item in bigrm:
            if (item[0]+' '+item[1]) in bi_dict and bi_dict[item[0]+' '+item[1]] < 0 :
                count = count + 1
        count_all.append([count])

    return count_all

# features: total score of tokens with score(w, p) > 0 using bigrams
def sum_positive_bi(all_value, bi_dict):

    sum_all = []
    for value in all_value:
        tokens = nltk.word_tokenize(value)
        bigrm = nltk.bigrams(tokens)
        sum = 0
        for item in bigrm:
            if (item[0]+' '+item[1]) in bi_dict and bi_dict[item[0]+' '+item[1]] > 0 :
                sum = sum + bi_dict[item[0]+' '+item[1]]
        sum_all.append([sum])

    return sum_all

# features: total score of tokens with score(w, p) < 0 using bigrams
def sum_negative_bi(all_value, bi_dict):

    sum_all = []
    for value in all_value:
        tokens = nltk.word_tokenize(value)
        bigrm = nltk.bigrams(tokens)
        sum = 0
        for item in bigrm:
            if (item[0]+' '+item[1]) in bi_dict and bi_dict[item[0]+' '+item[1]] < 0 :
                sum = sum + bi_dict[item[0]+' '+item[1]]
        sum_all.append([sum])

    return sum_all

# features: max score of tokens with score(w, p) > 0 using bigrams
def max_positive_bi(all_value, bi_dict):

    max_all = []
    for value in all_value:
        tokens = nltk.word_tokenize(value)
        bigrm = nltk.bigrams(tokens)
        max = 0
        for item in bigrm:
            if (item[0]+' '+item[1]) in bi_dict and bi_dict[item[0]+' '+item[1]] > 0 :
                if bi_dict[item[0]+' '+item[1]] > max:
                    max = bi_dict[item[0]+' '+item[1]]
        max_all.append([max])

    return max_all

# features: min score of tokens with score(w, p) < 0 using bigrams
def min_negative_bi(all_value, bi_dict):

    min_all = []
    for value in all_value:
        tokens = nltk.word_tokenize(value)
        bigrm = nltk.bigrams(tokens)
        min = 0
        for item in bigrm:
            if (item[0]+' '+item[1]) in bi_dict and bi_dict[item[0]+' '+item[1]] < 0 :
                if bi_dict[item[0]+' '+item[1]] < min:
                    min = bi_dict[item[0]+' '+item[1]]
        min_all.append([min])

    return min_all

# features: last score of tokens with score(w, p) > 0 using bigrams
def last_positive_bi(all_value, bi_dict):

    last_all = []
    for value in all_value:
        tokens = nltk.word_tokenize(value)
        bigrm = nltk.bigrams(tokens)
        last = 0
        for item in bigrm:
            if (item[0]+' '+item[1]) in bi_dict and bi_dict[item[0]+' '+item[1]] > 0 :
                last = bi_dict[item[0]+' '+item[1]]
        last_all.append([last])

    return last_all

# features: last score of tokens with score(w, p) < 0 using bigrams
def last_negative_bi(all_value, bi_dict):

    last_all = []
    for value in all_value:
        tokens = nltk.word_tokenize(value)
        bigrm = nltk.bigrams(tokens)
        last = 0
        for item in bigrm:
            if (item[0]+' '+item[1]) in bi_dict and bi_dict[item[0]+' '+item[1]] < 0 :
                last = bi_dict[item[0]+' '+item[1]]
        last_all.append([last])

    return last_all

# Ecn_feature: all cap
def number_all_cap(all_value):
    count_all = []
    for value in all_value:
        tokens = nltk.word_tokenize(value)
        count = 0
        for item in tokens:
            if item.isupper() :
                count = count + 1
        count_all.append([count])

    return count_all

# Ecn_feature: hashtag
def number_hashtag(all_value):
    count_all = []
    for value in all_value:
        tokens = nltk.word_tokenize(value)
        count = 0
        for item in tokens:
            if item[0] == '#' :
                count = count + 1
        count_all.append([count])

    return count_all

# constum_feature: emoji
def number_emoji(all_value):
    count_all = []
    for value in all_value:
        tokens = nltk.word_tokenize(value)
        count = 0
        for item in tokens:
            if find_emoji(item) :
                count = count + 1
        count_all.append([count])

    return count_all
