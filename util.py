import torch
import numpy as np
import pandas as pd
import re


def load_songs():
    df = pd.read_csv("data/train.csv")
    df = df[df["Language"] == "en"]
    df = df[df.Genre.isin(["Rock", "Metal", "Country", "Jazz", "Hip-Hop"])]
    df['Length'] = df['Lyrics'].str.len()
    df = df[(df.Length > 400) & (df.Length < 3000)]
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
    song = "".join(char.lower() for char in song if char not in blacklist and not char.isnumeric())
    song = song.encode("ascii", "ignore").decode()
    return song

def get_vector(key, alphabet):
    return alphabet.index(key)



def encode(data, alphabet, genres, unique_genres):
    maxlen = len(max(data, key=len))
    #maxlen = min(1000, maxlen)
    encoded = torch.zeros(maxlen, len(data), len(alphabet) + len(unique_genres))
    indeces = torch.zeros(maxlen, len(data))
    for genre, (j, song) in zip(genres, enumerate(data)):
        for i, letter in enumerate(song):
            if i == maxlen:
                break
            encoded[i][j][alphabet.index(letter)] = 1
            encoded[i][j][len(alphabet) + unique_genres.index(genre)] = 1
            indeces[i][j] = alphabet.index(letter)
        for i in range(maxlen - len(song)):
            encoded[i + len(song)][j][len(alphabet) - 1] = 1
            indeces[i + len(song)][j] = len(alphabet) - 1
    return encoded, indeces


def decode(data, alphabet):
    return ["".join(alphabet[np.argmax(letter)] for letter in song if letter[-1] != 1) for song in data]


def get_alphabet(data):
    return sorted(list(set(char for song in data for word in song for char in word))) + ["EOS"]


def find_char(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]


def float_range(start, stop, step):
    while start < stop:
        yield float(start)
        start += step