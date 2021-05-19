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
    #prompts the network and prints the generated results
    #prompt genre(s), number of prompts per genre, prompt length, and max (generated) lyric length can be set
    
    minibatch_size = 32
    model_file = "models/lowlr/model" # model loaction
    unique_genres = ["Jazz"] #define genre
    # sorted(["Rock", "Metal", "Country", "Jazz", "Hip-Hop"])

    songs, genres = load_songs(unique_genres)
    alphabet = get_alphabet(songs)
    unique_genres = list(set(genres))

    #modified prompt parameters are given to generate_gerrnre_prompts()
    prompt, prompt_genre = generate_genre_prompts(songs, genres, unique_genres, prompts_per_genre=2) # retrieve genre-appropriate prompts
    model = torch.load("models/newnetonegenre2/model_checkpoint_10") # load model

    with torch.no_grad():
        predict = 2000
        pred_songs = prompt_network(model, prompt, prompt_genre, alphabet, unique_genres, predict=predict, temperature=0.001) # generate lyrics
        for i, s in enumerate(pred_songs):
            print(prompt_genre[i]) #print genre and text
            print(prompt[i] + s[len(prompt[i]):])
