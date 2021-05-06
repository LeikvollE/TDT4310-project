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


class LyricSTM(nn.Module):
    def __init__(self, n_hidden: int, feature_size: int, alphabet: list):
        super().__init__()
        self.feature_size = feature_size
        self.n_hidden = n_hidden
        self.alphabet = alphabet

        self.lstm = nn.LSTMCell(self.feature_size, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.n_hidden, 128),
            nn.SELU(),
            nn.Linear(128, len(alphabet))
        )

    def forward(self, x, predict=0):
        outputs = []
        n_samples = x.size(1)

        genres = x[0,:,-len(unique_genres):]

        h_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)

        h2_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c2_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)

        for i in range(x.size()[0]):
            h_t, c_t = self.lstm(x[i], (h_t, c_t))
            h2_t, c2_t = self.lstm2(h_t, (h2_t, c2_t))
            output = self.linear(h2_t)
            outputs.append(output)

        for i in range(predict):
            max_idx = torch.argmax(output, 1, keepdim=True)
            one_hot = torch.FloatTensor(output.shape)
            one_hot.zero_()
            one_hot.scatter_(1, max_idx, 1)
            one_hot = torch.cat((one_hot, genres), dim=1)
            h_t, c_t = self.lstm(one_hot, (h_t, c_t))
            h2_t, c2_t = self.lstm2(h_t, (h2_t, c2_t))
            output = self.linear(h2_t)
            outputs.append(output)
            if all(torch.argmax(output[j]) == len(self.alphabet) - 1 for j in range(len(output))):
                break

        outputs = torch.stack(outputs, dim=0)
        return outputs


def load_songs():
    df = pd.read_csv("data/train.csv")
    df = df[df["Language"] == "en"]
    df = df[df["Genre"] != "Indie"]
    genre = df["Genre"]
    df = df.sample(frac=1).reset_index(drop=True)
    minc = 2000
    song_dfs = []
    for count, name in zip(genre.value_counts(), genre.value_counts().index):
        song = df[df["Genre"] == name]
        fraction = min(minc / len(song), 1)
        song = song.sample(frac=fraction).reset_index(drop=True)
        song_dfs.append(song)

    allsongs_df = pd.concat(song_dfs)
    allsongs_df = allsongs_df.sample(frac=1).reset_index(drop=True)
    songs = allsongs_df["Lyrics"].values.tolist()
    genres = allsongs_df["Genre"].values.tolist()

    songs = [clean_up(song) for song in songs]
    return songs, genres


def clean_up(song):
    filter = r"\[(.*?)\]"
    song = re.sub(filter, "", song)
    blacklist = "<>@[]{}\x7f\t=+|~*%/\\_^;#$`§\x19\x13"
    song = "".join(char for char in song if char not in blacklist and not char.isnumeric())
    song = song.encode("ascii", "ignore").decode()
    return song

def get_vector(key, alphabet):
    return alphabet.index(key)


def encode(data, alphabet, genres, unique_genres):
    maxlen = len(max(data, key=len))
    #maxlen = min(1000, maxlen)
    encoded = torch.zeros(maxlen+1, len(data), len(alphabet)+len(unique_genres))
    indeces = torch.zeros(maxlen+1, len(data))
    for genre, (j, song) in zip(genres, enumerate(data)):
        for i, letter in enumerate(song):
            if i > maxlen:
                continue
            encoded[i][j][alphabet.index(letter)] = 1
            encoded[i][j][len(alphabet)+unique_genres.index(genre)] = 1
            indeces[i][j] = alphabet.index(letter)
        for i in range(maxlen-min(len(song), maxlen)+1):
            encoded[i+min(len(song), maxlen)][j][-1] = 1
            indeces[i+min(len(song), maxlen)][j] = len(alphabet)-1
    return encoded, indeces


def decode(data, alphabet):
    return ["".join(alphabet[np.argmax(letter)] for letter in song) for song in data]


def get_alphabet(data):
    return sorted(list(set(char for song in data for word in song for char in word))) + ["EOS"]


if __name__ == "__main__":
    minibatch_size = 32
    model_file = "models/lowlr/model"

    songs, genres = load_songs()
    alphabet = get_alphabet(songs)
    unique_genres = list(set(genres))
    print(len(unique_genres))
    print(len(alphabet))
    """
    encoded, targets = encode(songs, alphabet)
    X = encoded[:-1]
    Y = targets[1:]
    Y = Y.long()
    Y = Y.permute(1,0)
    print(Y)"""
    count = Counter()
    for song in songs:
        count.update(song)
    print(count)

    prompt = ["When you think "]*len(unique_genres)
    prompt_genre = unique_genres
    prompt_encoded, _ = encode(prompt, alphabet, prompt_genre, unique_genres)
    X_prompt = prompt_encoded
    model = LyricSTM(n_hidden=256, feature_size=len(alphabet)+len(unique_genres), alphabet=alphabet)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    n_steps = 100
    for i in range(n_steps):
        print("Step", i)
        for j in tqdm(range(0, len(songs), minibatch_size)):
            if (j/32) % 100 == 0:
                print("-- j -- j -- j -- j -- j -- j -- j -- j -- j -- j --")
                with torch.no_grad():
                    predict = 5000
                    pred = model(X_prompt, predict=predict)
                    pred = pred.permute(1, 0, 2)
                    for s, g in zip(decode(pred, alphabet), prompt_genre):
                        print(g)
                        print(s)
                        print("--- Next ---")
            batch_songs, batch_genres = songs[j: j+minibatch_size], genres[j: j+minibatch_size]
            batch_encoded, batch_targets = encode(batch_songs, alphabet, batch_genres, unique_genres)
            X = batch_encoded[:-1]
            Y = batch_targets[1:]
            Y = Y.long()
            Y = Y.permute(1, 0)
            def closure():
                optimizer.zero_grad()
                out = model(X)
                out = out.permute(1,2,0)
                loss = criterion(out, Y)
                loss.backward()
                return loss
            optimizer.step(closure)
        torch.save(model, model_file+"_checkpoint_"+str(i))
        if i >= 0:
            with torch.no_grad():
                predict = 5000
                pred = model(X_prompt, predict=predict)
                pred = pred.permute(1,0,2)
                print(decode(pred, alphabet))

    torch.save(model, "models/final.torch")
