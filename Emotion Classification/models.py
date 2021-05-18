"""
May i use 3 days of late days please. Thank you.

COMS 4705 Natural Language Processing Spring 2021
Kathy McKeown
Homework 2: Emotion Classification with Neural Networks - Models File

<Yujing Chen>
<yc3851>
"""

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import numpy as np

class DenseNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, ncls, pretrained_embedding=None):
        super(DenseNetwork, self).__init__()

        ########## YOUR CODE HERE ##########
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embedding is not None:
            self.embedding_layer.weight.data.copy_(pretrained_embedding)
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, ncls),
            nn.Softmax(1)
        )

    def forward(self, x):
        ########## YOUR CODE HERE ##########
        x = self.embedding_layer(x)
        seq_len = x.shape[1]
        x = torch.sum(x, dim=1) / seq_len
        x = self.model(x)
        return x


class RecurrentNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, ncls, pretrained_embedding=None):
        super(RecurrentNetwork, self).__init__()

        ########## YOUR CODE HERE ##########
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)#.requires_grad_(False)
        if pretrained_embedding is not None:
            self.embedding_layer.weight.data.copy_(pretrained_embedding)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.model = nn.Sequential(
            nn.Linear(hidden_dim, ncls),
            nn.ReLU(inplace=True),
            nn.Softmax(1)
        )

    # x is a PaddedSequence for an RNN
    def forward(self, x):
        ########## YOUR CODE HERE ##########
        x = self.embedding_layer(x)
        x, _ = self.lstm(x)
        x = torch.mean(x, dim=1)
        x = self.model(x)
        return x


# --- extension-grading 2 --- #
class RecurrentAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, ncls, attn_dim=16, attn_heads=4, pretrained_embedding=None):
        super(RecurrentAttention, self).__init__()

        ########## YOUR CODE HERE ##########
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)#.requires_grad_(False)
        if pretrained_embedding is not None:
            self.embedding_layer.weight.data.copy_(pretrained_embedding)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.k_linears = nn.Linear(hidden_dim, attn_dim*attn_heads, bias=False) 
        self.q_linears = nn.Linear(hidden_dim, attn_dim*attn_heads, bias=False) 
        self.v_linears = nn.Linear(hidden_dim, attn_dim*attn_heads, bias=False) 
        self.model = nn.Sequential(
            nn.Linear(attn_dim*attn_heads, ncls),
            nn.ReLU(inplace=True),
            nn.Softmax(1)
        )

    # x is a PaddedSequence for an RNN
    def forward(self, x):
        ########## YOUR CODE HERE ##########
        def attention(query, key, value):
            # "Compute 'Scaled Dot Product Attention'"
            key = key.transpose(1, 2)
            scores = torch.matmul(query, key)
            p_attn = nn.Softmax(dim=-1)(scores)
            return torch.matmul(p_attn, value), p_attn

        x = self.embedding_layer(x)
        x, _ = self.lstm(x)
        keys = self.k_linears(x)
        queries = self.q_linears(x)
        values = self.v_linears(x)
        x, _ = attention(queries, keys, values)
        x = torch.mean(x, dim=1)
        x = self.model(x)
        return x
