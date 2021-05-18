"""
May i use 3 days of late days please. Thank you.

COMS 4705 Natural Language Processing Spring 2021
Kathy McKeown
Homework 2: Emotion Classification with Neural Networks - Main File

<Yujing Chen>
<yc3851>
"""

# Imports
import nltk
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Imports - our files
import utils
import models
import argparse

# Global definitions - data
DATA_FN = 'data/crowdflower_data.csv'
LABEL_NAMES = ["happiness", "worry", "neutral", "sadness"]

# Global definitions - architecture
EMBEDDING_DIM = 100  # We will use pretrained 100-dimensional GloVe
BATCH_SIZE = 128
NUM_CLASSES = 4
USE_CUDA = torch.cuda.is_available()  # CUDA will be available if you are using the GPU image for this homework

# Global definitions - saving and loading data
FRESH_START = False  # set this to false after running once with True to just load your preprocessed data from file
#                     (good for debugging)
TEMP_FILE = "temporary_data.pkl"  # if you set FRESH_START to false, the program will look here for your data, etc.


def train_model(model, loss_fn, optimizer, train_generator, dev_generator, early_stopping_cnt=1):
    """
    Perform the actual training of the model based on the train and dev sets.
    :param model: one of your models, to be trained to perform 4-way emotion classification
    :param loss_fn: a function that can calculate loss between the predicted and gold labels
    :param optimizer: a created optimizer you will use to update your model weights
    :param train_generator: a DataLoader that provides batches of the training set
    :param dev_generator: a DataLoader that provides batches of the development set
    :return model, the trained model
    """
    ########## YOUR CODE HERE ##########
    epoch = 1
    previous_dev_loss = None
    early_stopping_num = 0
    while True:
        model.train()
        for x, y in train_generator:
            if USE_CUDA:
                x = x.cuda()
                y = y.cuda()

            optimizer.zero_grad()
            z = model(x)
            loss = loss_fn(z, y)
            loss.backward()
            optimizer.step()
        
        # Keep track of the loss
        dev_loss = 0.
        model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(dev_generator):
                if USE_CUDA:
                    x = x.cuda()
                    y = y.cuda()

                z = model(x)
                loss = loss_fn(z, y)
                dev_loss += loss.data.cpu().numpy()
            dev_loss /= i

        print('%d epoch, validation loss = %.4f' % (epoch, dev_loss))
        epoch += 1

        # --- extension-grading 1 --- #
        # a more complicated early stopping schedule: 
        # if validation loss increases for continuous #early_stopping_cnt epochs,
        # then stop the training procedure.
        # default #early_stopping_cnt = 1 is the common case.
        if previous_dev_loss is None:
            previous_dev_loss = dev_loss
        else:
            if previous_dev_loss < dev_loss:
                early_stopping_num += 1
            else:
                early_stopping_num = 0
            previous_dev_loss = dev_loss
        
        if early_stopping_num >= early_stopping_cnt:
            break


def test_model(model, loss_fn, test_generator):
    """
    Evaluate the performance of a model on the development set, providing the loss and macro F1 score.
    :param model: a model that performs 4-way emotion classification
    :param loss_fn: a function that can calculate loss between the predicted and gold labels
    :param test_generator: a DataLoader that provides batches of the testing set
    """
    gold = []
    predicted = []

    # Keep track of the loss
    loss = torch.zeros(1)  # requires_grad = False by default; float32 by default
    if USE_CUDA:
        loss = loss.cuda()

    model.eval()

    # Iterate over batches in the test dataset
    with torch.no_grad():
        for i, (X_b, y_b) in enumerate(test_generator):
            if USE_CUDA:
                X_b = X_b.cuda()
                y_b = y_b.cuda()
            # Predict
            y_pred = model(X_b)

            # Save gold and predicted labels for F1 score - take the argmax to convert to class labels
            gold.extend(y_b.cpu().detach().numpy())
            predicted.extend(y_pred.argmax(1).cpu().detach().numpy())

            loss += loss_fn(y_pred.double(), y_b.long()).data
        loss /= i

    # Print total loss and macro F1 score
    print("Test loss: ")
    print(loss)
    print("F-score: ")
    print(f1_score(gold, predicted, average='macro'))


def main(args):
    """
    Train and test neural network models for emotion classification.
    """
    # Prepare the data and the pretrained embedding matrix
    if FRESH_START:
        print("Preprocessing all data from scratch....")
        train, dev, test = utils.get_data(DATA_FN)
        # train_data includes .word2idx and .label_enc as fields if you would like to use them at any time
        train_generator, dev_generator, test_generator, embeddings, train_data = utils.vectorize_data(train, dev, test,
                                                                                                BATCH_SIZE,
                                                                                                EMBEDDING_DIM)

        print("Saving DataLoaders and embeddings so you don't need to create them again; you can set FRESH_START to "
              "False to load them from file....")
        with open(TEMP_FILE, "wb+") as f:
            pickle.dump((train_generator, dev_generator, test_generator, embeddings, train_data), f)
    else:
        try:
            with open(TEMP_FILE, "rb") as f:
                print("Loading DataLoaders and embeddings from file....")
                train_generator, dev_generator, test_generator, embeddings, train_data = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("You need to have saved your data with FRESH_START=True once in order to load it!")
            

    # Use this loss function in your train_model() and test_model()
    loss_fn = nn.CrossEntropyLoss()

    ########## YOUR CODE HERE ##########
    VOCAB_SIZE = embeddings.shape[0]

    if args.model == 'dense':
        model = models.DenseNetwork(vocab_size = VOCAB_SIZE, 
                    embedding_dim = EMBEDDING_DIM, 
                    hidden_dim = 100, 
                    ncls = len(LABEL_NAMES), 
                    pretrained_embedding = embeddings)
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
    elif args.model == 'RNN':
        model = models.RecurrentNetwork(vocab_size = VOCAB_SIZE, 
                    embedding_dim = EMBEDDING_DIM, 
                    hidden_dim = 100, 
                    ncls = len(LABEL_NAMES), 
                    pretrained_embedding = embeddings)
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
    elif args.model == 'extension1':
        model = models.RecurrentAttention(vocab_size = VOCAB_SIZE, 
                    embedding_dim = EMBEDDING_DIM, 
                    hidden_dim = 100, 
                    ncls = len(LABEL_NAMES), 
                    pretrained_embedding = embeddings)
        optimizer = optim.Adam(model.parameters(), lr=0.0005)

    
    if USE_CUDA:
        model = model.cuda()
    
    train_model(model, loss_fn, optimizer, train_generator, dev_generator)
    test_model(model, loss_fn, test_generator)

    # --- extension-grading 1 --- #
    if args.model == 'dense':
        model = models.DenseNetwork(vocab_size = VOCAB_SIZE, 
                    embedding_dim = EMBEDDING_DIM, 
                    hidden_dim = 100, 
                    ncls = len(LABEL_NAMES), 
                    pretrained_embedding = embeddings)
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
    elif args.model == 'RNN':
        model = models.RecurrentNetwork(vocab_size = VOCAB_SIZE, 
                    embedding_dim = EMBEDDING_DIM, 
                    hidden_dim = 100, 
                    ncls = len(LABEL_NAMES), 
                    pretrained_embedding = embeddings)
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
    elif args.model == 'extension1':
        model = models.RecurrentAttention(vocab_size = VOCAB_SIZE, 
                    embedding_dim = EMBEDDING_DIM, 
                    hidden_dim = 100, 
                    ncls = len(LABEL_NAMES), 
                    pretrained_embedding = embeddings)
        optimizer = optim.Adam(model.parameters(), lr=0.0005)

    if USE_CUDA:
        model = model.cuda()
    
    train_model(model, loss_fn, optimizer, train_generator, dev_generator, early_stopping_cnt=3)
    test_model(model, loss_fn, test_generator)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', required=True,
                        choices=["dense", "RNN", "extension1", "extension2"],
                        help='The name of the model to train and evaluate.')
    args = parser.parse_args()
    main(args)
