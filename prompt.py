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
from util import load_songs, get_alphabet, prompt_network, generate_genre_prompts
from Network import LyricSTM as LyricSTM


if __name__ == "__main__":
    minibatch_size = 32
    model_file = "models/lowlr/model"
    unique_genres = ["Jazz"]  # sorted(["Rock", "Metal", "Country", "Jazz", "Hip-Hop"])

    songs, genres = load_songs(unique_genres)
    alphabet = get_alphabet(songs)
    unique_genres = list(set(genres))

    prompt, prompt_genre = generate_genre_prompts(songs, genres, unique_genres, prompts_per_genre=2)
    model_ = torch.load("models/newnetonegenre2/model_checkpoint_10")
    model = LyricSTM(model_.n_hidden, len(alphabet), model_.alphabet, unique_genres)
    model.load_state_dict(model_.state_dict())

    with torch.no_grad():
        predict = 2000
        pred_songs = prompt_network(model, prompt, prompt_genre, alphabet, unique_genres, predict=predict, temperature=0.001)
        for i, s in enumerate(pred_songs):
            print(prompt_genre[i])
            print(prompt[i] + s[len(prompt[i]):])
