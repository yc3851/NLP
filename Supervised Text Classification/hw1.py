import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='train', required=True,
                        help='Full path to the training file')
    parser.add_argument('--test', dest='test', required=True,
                        help='Full path to the evaluation file')
    parser.add_argument('--model', dest='model', required=True,
                        choices=["Ngram", "Ngram+Lex", "Ngram+Lex+Enc", "Custom"],
                        help='The name of the model to train and evaluate.')
    parser.add_argument('--lexicon_path', dest='lexicon_path', required=True,
                        help='The full path to the directory containing the lexica.'
                             ' The last folder of this path should be "lexica".')
    args = parser.parse_args()


    ## example access parsed args
    #print(args.model)

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import f1_score
from sklearn import svm
from scipy.sparse import coo_matrix, hstack
from nltk.tokenize import word_tokenize
from feature import *
import nltk
import csv

# Data gathering/preprocessing
# sentiment polarity of the tweet: “negative” (0), “positive” (1), “neutral” (2).
def label_convert(value):
    if value == 'neutral' or value == 'objective':
        return 2
    elif value == 'negative':
        return 0
    elif value == 'positive':
        return 1

# read data, using Sentiment140
train_data = pd.read_csv(args.train, error_bad_lines=False, converters={'label': label_convert})
test_data = pd.read_csv(args.test, error_bad_lines=False, converters={'label': label_convert})
lexica_data_uni = pd.read_csv(args.lexicon_path+'Hashtag-Sentiment-Lexicon/HS-unigrams.txt', sep="\t", header=None, quoting=csv.QUOTE_NONE, engine='python', error_bad_lines=False)
lexica_data_bi = pd.read_csv(args.lexicon_path+'Hashtag-Sentiment-Lexicon/HS-bigrams.txt', sep="\t", header=None, quoting=csv.QUOTE_NONE, engine='python', error_bad_lines=False)

# change lexica_data_uni to dictionary to save time
lexica_data_uni_dic = {}
lexica_data_bi_dic = {}

for index, row in lexica_data_uni.iterrows():
    lexica_data_uni_dic.update({row[0]:row[1]})

for index, row in lexica_data_bi.iterrows():
    lexica_data_bi_dic.update({row[0]:row[1]})

# print(lexica_data_bi_dic)

# ngram model
def ngram(train_data, test_data):
    # Vectorization with n-gram
    count_vectorizer = CountVectorizer(lowercase=False,ngram_range=(3, 5), analyzer='char')

    # Training
    X_train = count_vectorizer.fit_transform(train_data['tweet_tokens'].values)
    X_test = count_vectorizer.transform(test_data['tweet_tokens'].values)

    y_train = train_data['label'].values
    y_test = test_data['label'].values

    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, y_train)

    # Prediction and F1 score
    y_predictions = classifier.predict(X_test)
    seprate_score = f1_score(y_test, y_predictions, average=None)
    print('F1 score for negative class: ' + str(seprate_score[0]) + '\n' +
          'F1 score for positive class: ' + str(seprate_score[1]) + '\n' +
          'F1 score for neutral class: ' + str(seprate_score[2]))
    score = f1_score(y_test, y_predictions, average='macro')
    print('Macro F1 Score: ' + str(score))

    return 0

# ngram + lex model
def ngram_lex(train_data, test_data, uni_dict, bi_dict):
    # Vectorization with n-gram
    count_vectorizer = CountVectorizer(lowercase=False,ngram_range=(3, 5), analyzer='char')

    # Training , convert lex feature to spare matrix then merge
    X_train_ngram = count_vectorizer.fit_transform(train_data['tweet_tokens'].values)
    F1_train = coo_matrix(count_positive_uni(train_data['tweet_tokens'].values, uni_dict))
    F2_train = coo_matrix(count_negative_uni(train_data['tweet_tokens'].values, uni_dict))
    F3_train = coo_matrix(sum_positive_uni(train_data['tweet_tokens'].values, uni_dict))
    F4_train = coo_matrix(sum_negative_uni(train_data['tweet_tokens'].values, uni_dict))
    F5_train = coo_matrix(max_positive_uni(train_data['tweet_tokens'].values, uni_dict))
    F6_train = coo_matrix(min_negative_uni(train_data['tweet_tokens'].values, uni_dict))
    F7_train = coo_matrix(last_positive_uni(train_data['tweet_tokens'].values, uni_dict))
    F8_train = coo_matrix(last_negative_uni(train_data['tweet_tokens'].values, uni_dict))
    F9_train = coo_matrix(count_positive_bi(train_data['tweet_tokens'].values, bi_dict))
    F10_train = coo_matrix(count_negative_bi(train_data['tweet_tokens'].values, bi_dict))
    F11_train = coo_matrix(sum_positive_bi(train_data['tweet_tokens'].values, bi_dict))
    F12_train = coo_matrix(sum_negative_bi(train_data['tweet_tokens'].values, bi_dict))
    F13_train = coo_matrix(max_positive_bi(train_data['tweet_tokens'].values, bi_dict))
    F14_train = coo_matrix(min_negative_bi(train_data['tweet_tokens'].values, bi_dict))
    F15_train = coo_matrix(last_positive_bi(train_data['tweet_tokens'].values, bi_dict))
    F16_train = coo_matrix(last_negative_bi(train_data['tweet_tokens'].values, bi_dict))
    X_train = hstack([X_train_ngram,F1_train,F2_train,F3_train,F4_train,F5_train,F6_train,F7_train,F8_train,F9_train,F10_train,F11_train,F12_train,F13_train,F14_train,F15_train,F16_train])

    X_test_ngram = count_vectorizer.transform(test_data['tweet_tokens'].values)
    F1_test = coo_matrix(count_positive_uni(test_data['tweet_tokens'].values, uni_dict))
    F2_test = coo_matrix(count_negative_uni(test_data['tweet_tokens'].values, uni_dict))
    F3_test = coo_matrix(sum_positive_uni(test_data['tweet_tokens'].values, uni_dict))
    F4_test = coo_matrix(sum_negative_uni(test_data['tweet_tokens'].values, uni_dict))
    F5_test = coo_matrix(max_positive_uni(test_data['tweet_tokens'].values, uni_dict))
    F6_test = coo_matrix(min_negative_uni(test_data['tweet_tokens'].values, uni_dict))
    F7_test = coo_matrix(last_positive_uni(test_data['tweet_tokens'].values, uni_dict))
    F8_test = coo_matrix(last_negative_uni(test_data['tweet_tokens'].values, uni_dict))
    F9_test = coo_matrix(count_positive_bi(test_data['tweet_tokens'].values, bi_dict))
    F10_test = coo_matrix(count_negative_bi(test_data['tweet_tokens'].values, bi_dict))
    F11_test = coo_matrix(sum_positive_bi(test_data['tweet_tokens'].values, bi_dict))
    F12_test = coo_matrix(sum_negative_bi(test_data['tweet_tokens'].values, bi_dict))
    F13_test = coo_matrix(max_positive_bi(test_data['tweet_tokens'].values, bi_dict))
    F14_test = coo_matrix(min_negative_bi(test_data['tweet_tokens'].values, bi_dict))
    F15_test = coo_matrix(last_positive_bi(test_data['tweet_tokens'].values, bi_dict))
    F16_test = coo_matrix(last_negative_bi(test_data['tweet_tokens'].values, bi_dict))
    X_test = hstack([X_test_ngram,F1_test,F2_test,F3_test,F4_test,F5_test,F6_test,F7_test,F8_test,F9_test,F10_test,F11_test,F12_test,F13_test,F14_test,F15_test,F16_test])

    y_train = train_data['label'].values
    y_test = test_data['label'].values

    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, y_train)

    # Prediction and F1 score
    y_predictions = classifier.predict(X_test)
    seprate_score = f1_score(y_test, y_predictions, average=None)
    print('F1 score for negative class: ' + str(seprate_score[0]) + '\n' +
          'F1 score for positive class: ' + str(seprate_score[1]) + '\n' +
          'F1 score for neutral class: ' + str(seprate_score[2]))
    score = f1_score(y_test, y_predictions, average='macro')
    print('Macro F1 Score: ' + str(score))

    return 0

# ngram + lex model + Encode; intergrate all_cap; pos_tag
def ngram_lex_enc(train_data, test_data, uni_dict, bi_dict):
        # Vectorization with n-gram
        count_vectorizer = CountVectorizer(lowercase=False,ngram_range=(3, 5), analyzer='char')

        # Training , convert lex feature to spare matrix then merge
        X_train_ngram = count_vectorizer.fit_transform(train_data['tweet_tokens'].values)
        F1_train = coo_matrix(count_positive_uni(train_data['tweet_tokens'].values, uni_dict))
        F2_train = coo_matrix(count_negative_uni(train_data['tweet_tokens'].values, uni_dict))
        F3_train = coo_matrix(sum_positive_uni(train_data['tweet_tokens'].values, uni_dict))
        F4_train = coo_matrix(sum_negative_uni(train_data['tweet_tokens'].values, uni_dict))
        F5_train = coo_matrix(max_positive_uni(train_data['tweet_tokens'].values, uni_dict))
        F6_train = coo_matrix(min_negative_uni(train_data['tweet_tokens'].values, uni_dict))
        F7_train = coo_matrix(last_positive_uni(train_data['tweet_tokens'].values, uni_dict))
        F8_train = coo_matrix(last_negative_uni(train_data['tweet_tokens'].values, uni_dict))
        F9_train = coo_matrix(count_positive_bi(train_data['tweet_tokens'].values, bi_dict))
        F10_train = coo_matrix(count_negative_bi(train_data['tweet_tokens'].values, bi_dict))
        F11_train = coo_matrix(sum_positive_bi(train_data['tweet_tokens'].values, bi_dict))
        F12_train = coo_matrix(sum_negative_bi(train_data['tweet_tokens'].values, bi_dict))
        F13_train = coo_matrix(max_positive_bi(train_data['tweet_tokens'].values, bi_dict))
        F14_train = coo_matrix(min_negative_bi(train_data['tweet_tokens'].values, bi_dict))
        F15_train = coo_matrix(last_positive_bi(train_data['tweet_tokens'].values, bi_dict))
        F16_train = coo_matrix(last_negative_bi(train_data['tweet_tokens'].values, bi_dict))
        train_all_cap = coo_matrix(number_all_cap(train_data['tweet_tokens'].values))
        train_postag = coo_matrix(pos_feature(train_data['pos_tags'].values))
        X_train = hstack([X_train_ngram,F1_train,F2_train,F3_train,F4_train,F5_train,F6_train,F7_train,F8_train,F9_train,F10_train,F11_train,F12_train,F13_train,F14_train,F15_train,F16_train,train_all_cap,train_postag])

        X_test_ngram = count_vectorizer.transform(test_data['tweet_tokens'].values)
        F1_test = coo_matrix(count_positive_uni(test_data['tweet_tokens'].values, uni_dict))
        F2_test = coo_matrix(count_negative_uni(test_data['tweet_tokens'].values, uni_dict))
        F3_test = coo_matrix(sum_positive_uni(test_data['tweet_tokens'].values, uni_dict))
        F4_test = coo_matrix(sum_negative_uni(test_data['tweet_tokens'].values, uni_dict))
        F5_test = coo_matrix(max_positive_uni(test_data['tweet_tokens'].values, uni_dict))
        F6_test = coo_matrix(min_negative_uni(test_data['tweet_tokens'].values, uni_dict))
        F7_test = coo_matrix(last_positive_uni(test_data['tweet_tokens'].values, uni_dict))
        F8_test = coo_matrix(last_negative_uni(test_data['tweet_tokens'].values, uni_dict))
        F9_test = coo_matrix(count_positive_bi(test_data['tweet_tokens'].values, bi_dict))
        F10_test = coo_matrix(count_negative_bi(test_data['tweet_tokens'].values, bi_dict))
        F11_test = coo_matrix(sum_positive_bi(test_data['tweet_tokens'].values, bi_dict))
        F12_test = coo_matrix(sum_negative_bi(test_data['tweet_tokens'].values, bi_dict))
        F13_test = coo_matrix(max_positive_bi(test_data['tweet_tokens'].values, bi_dict))
        F14_test = coo_matrix(min_negative_bi(test_data['tweet_tokens'].values, bi_dict))
        F15_test = coo_matrix(last_positive_bi(test_data['tweet_tokens'].values, bi_dict))
        F16_test = coo_matrix(last_negative_bi(test_data['tweet_tokens'].values, bi_dict))
        test_all_cap = coo_matrix(number_all_cap(test_data['tweet_tokens'].values))
        test_postag = coo_matrix(pos_feature(test_data['pos_tags'].values))
        X_test = hstack([X_test_ngram,F1_test,F2_test,F3_test,F4_test,F5_test,F6_test,F7_test,F8_test,F9_test,F10_test,F11_test,F12_test,F13_test,F14_test,F15_test,F16_test,test_all_cap,test_postag])

        y_train = train_data['label'].values
        y_test = test_data['label'].values

        classifier = svm.SVC(kernel='linear')
        classifier.fit(X_train, y_train)

        # Prediction and F1 score
        y_predictions = classifier.predict(X_test)
        seprate_score = f1_score(y_test, y_predictions, average=None)
        print('F1 score for negative class: ' + str(seprate_score[0]) + '\n' +
              'F1 score for positive class: ' + str(seprate_score[1]) + '\n' +
              'F1 score for neutral class: ' + str(seprate_score[2]))
        score = f1_score(y_test, y_predictions, average='macro')
        print('Macro F1 Score: ' + str(score))

        return 0

# custom model; k for n-gram adjust: adding feature: pos_tag
def custom(train_data, test_data, uni_dict, bi_dict):
    # Vectorization with n-gram
    count_vectorizer = CountVectorizer(lowercase=False,ngram_range=(2, 6), analyzer='char')

    # Training , convert lex feature to spare matrix then merge
    X_train_ngram = count_vectorizer.fit_transform(train_data['tweet_tokens'].values)
    F1_train = coo_matrix(count_positive_uni(train_data['tweet_tokens'].values, uni_dict))
    F2_train = coo_matrix(count_negative_uni(train_data['tweet_tokens'].values, uni_dict))
    F3_train = coo_matrix(sum_positive_uni(train_data['tweet_tokens'].values, uni_dict))
    F4_train = coo_matrix(sum_negative_uni(train_data['tweet_tokens'].values, uni_dict))
    F5_train = coo_matrix(max_positive_uni(train_data['tweet_tokens'].values, uni_dict))
    F6_train = coo_matrix(min_negative_uni(train_data['tweet_tokens'].values, uni_dict))
    F7_train = coo_matrix(last_positive_uni(train_data['tweet_tokens'].values, uni_dict))
    F8_train = coo_matrix(last_negative_uni(train_data['tweet_tokens'].values, uni_dict))
    F9_train = coo_matrix(count_positive_bi(train_data['tweet_tokens'].values, bi_dict))
    F10_train = coo_matrix(count_negative_bi(train_data['tweet_tokens'].values, bi_dict))
    F11_train = coo_matrix(sum_positive_bi(train_data['tweet_tokens'].values, bi_dict))
    F12_train = coo_matrix(sum_negative_bi(train_data['tweet_tokens'].values, bi_dict))
    F13_train = coo_matrix(max_positive_bi(train_data['tweet_tokens'].values, bi_dict))
    F14_train = coo_matrix(min_negative_bi(train_data['tweet_tokens'].values, bi_dict))
    F15_train = coo_matrix(last_positive_bi(train_data['tweet_tokens'].values, bi_dict))
    F16_train = coo_matrix(last_negative_bi(train_data['tweet_tokens'].values, bi_dict))
    train_all_cap = coo_matrix(number_all_cap(train_data['tweet_tokens'].values))
    train_emoji = coo_matrix(number_emoji(train_data['tweet_tokens'].values))
    X_train = hstack([X_train_ngram,F1_train,F2_train,F3_train,F4_train,F5_train,F6_train,F7_train,F8_train,F9_train,F10_train,F11_train,F12_train,F13_train,F14_train,F15_train,F16_train,train_all_cap])

    X_test_ngram = count_vectorizer.transform(test_data['tweet_tokens'].values)
    F1_test = coo_matrix(count_positive_uni(test_data['tweet_tokens'].values, uni_dict))
    F2_test = coo_matrix(count_negative_uni(test_data['tweet_tokens'].values, uni_dict))
    F3_test = coo_matrix(sum_positive_uni(test_data['tweet_tokens'].values, uni_dict))
    F4_test = coo_matrix(sum_negative_uni(test_data['tweet_tokens'].values, uni_dict))
    F5_test = coo_matrix(max_positive_uni(test_data['tweet_tokens'].values, uni_dict))
    F6_test = coo_matrix(min_negative_uni(test_data['tweet_tokens'].values, uni_dict))
    F7_test = coo_matrix(last_positive_uni(test_data['tweet_tokens'].values, uni_dict))
    F8_test = coo_matrix(last_negative_uni(test_data['tweet_tokens'].values, uni_dict))
    F9_test = coo_matrix(count_positive_bi(test_data['tweet_tokens'].values, bi_dict))
    F10_test = coo_matrix(count_negative_bi(test_data['tweet_tokens'].values, bi_dict))
    F11_test = coo_matrix(sum_positive_bi(test_data['tweet_tokens'].values, bi_dict))
    F12_test = coo_matrix(sum_negative_bi(test_data['tweet_tokens'].values, bi_dict))
    F13_test = coo_matrix(max_positive_bi(test_data['tweet_tokens'].values, bi_dict))
    F14_test = coo_matrix(min_negative_bi(test_data['tweet_tokens'].values, bi_dict))
    F15_test = coo_matrix(last_positive_bi(test_data['tweet_tokens'].values, bi_dict))
    F16_test = coo_matrix(last_negative_bi(test_data['tweet_tokens'].values, bi_dict))
    test_all_cap = coo_matrix(number_all_cap(test_data['tweet_tokens'].values))
    test_emoji = coo_matrix(number_emoji(test_data['tweet_tokens'].values))
    X_test = hstack([X_test_ngram,F1_test,F2_test,F3_test,F4_test,F5_test,F6_test,F7_test,F8_test,F9_test,F10_test,F11_test,F12_test,F13_test,F14_test,F15_test,F16_test,test_all_cap])

    y_train = train_data['label'].values
    y_test = test_data['label'].values

    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, y_train)

    # Prediction and F1 score
    y_predictions = classifier.predict(X_test)
    seprate_score = f1_score(y_test, y_predictions, average=None)
    print('F1 score for negative class: ' + str(seprate_score[0]) + '\n' +
          'F1 score for positive class: ' + str(seprate_score[1]) + '\n' +
          'F1 score for neutral class: ' + str(seprate_score[2]))
    score = f1_score(y_test, y_predictions, average='macro')
    print('Macro F1 Score: ' + str(score))

    return 0

if args.model == 'Ngram':
    ngram(train_data, test_data)
if args.model == 'Ngram+Lex':
    ngram_lex(train_data, test_data, lexica_data_uni_dic, lexica_data_bi_dic)
if args.model == 'Ngram+Lex+Enc':
    ngram_lex_enc(train_data, test_data, lexica_data_uni_dic, lexica_data_bi_dic)
if args.model == 'Custom':
    custom(train_data, test_data, lexica_data_uni_dic, lexica_data_bi_dic)
