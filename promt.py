import torch
import torch.nn as nn
import csv
import numpy as np
import torch.optim as optim
import pandas as pd
import re
import pickle
from tqdm import tqdm
from collections import Counter
import torch.nn.functional as F
from  torch.nn.utils.rnn import pad_sequence
from util import encode, decode, load_songs, get_alphabet, find_char
from Network import LyricSTM


if __name__ == "__main__":
    minibatch_size = 32
    model_file = "models/lowlr/model"

    songs, genres = load_songs()
    alphabet = get_alphabet(songs)
    unique_genres = list(set(genres))

    prompt = ["When you think "]*len(unique_genres)
    prompt_genre = unique_genres
    prompt_encoded, _ = encode(prompt, alphabet, prompt_genre, unique_genres)
    X_prompt = prompt_encoded[:-1]
    model_ = torch.load("models/fixednet/model_checkpoint_0")
    model = LyricSTM(model_.n_hidden, model_.feature_size, model_.alphabet)
    model.load_state_dict(model_.state_dict())

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    with torch.no_grad():
        predict = 2000
        pred = model(X_prompt, predict=predict, temp=0.45)
        pred = pred.permute(1,0,2)
        for m, (s, g) in enumerate(zip(decode(pred, alphabet), prompt_genre)):
            print(g)
            print(prompt[m] + s[len(prompt[m]):])
            print("--- Next ---")
