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
from torch.nn.utils.rnn import pad_sequence
from util import encode, decode, load_songs, get_alphabet, find_char, float_range
from collections import defaultdict
from Network import LyricSTM
import random
import nltk

import statistics
from sklearn.feature_extraction.text import TfidfVectorizer

def tf_idf(songs, preds):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(songs + preds).toarray()
    return X[:-len(preds)], X[-len(preds):]

def semantic_sim(tf_vectors, pred_vector):
    scores = []
    for vector in tf_vectors:
        scores.append(cosine_dist(vector, pred_vector))
    return statistics.mean(scores)

def cosine_dist(song1, song2):
    return np.dot(song1, song2)/(np.linalg.norm(song1) * np.linalg.norm(song2))

def lyrical_uniqueness(tf_vectors, pred_vector):
    scores = []
    duplicates = 0
    for vector in tf_vectors:
        s = cosine_dist(vector, pred_vector)
        scores.append(s if s < 1.0 else 0)
        if s == 1.0:
            duplicates += 1
    if duplicates > 1:
        print("Duplicates: ", duplicates - 1)
    return max(scores)

def vocab_quality(songs, preds):
    unique_words = set()
    for song in songs:
        unique_words.update(nltk.word_tokenize(song))

    all_words = set()
    for pred in preds:
        all_words.update(nltk.word_tokenize(pred))
    correct_words = len([1 for word in all_words if word in unique_words])

    p = str(correct_words * 100 / len(all_words))[:5] + "%"
    print(f"Model used {len(all_words)} tokens, of which {correct_words} are true words ({p})")


if __name__ == "__main__":
    minibatch_size = 32

    songs, genres = load_songs()

    alphabet = get_alphabet(songs)
    unique_genres = sorted(list(set(genres)))

    prompt = ["when you think "]*len(unique_genres)
    prompt_genre = unique_genres
    prompt_encoded, _ = encode(prompt, alphabet, prompt_genre, unique_genres)
    X_prompt = prompt_encoded[:-1]
    model = torch.load("models/alphanet/model_checkpoint_2")


    predictions = {}
    all_preds = []

    prompts_per_genre = 10
    prompt_length = 16
    predict = 1000
    with torch.no_grad():
        for genre in unique_genres:
            prompts = []
            for i, song in enumerate(songs):
                if len(prompts) == prompts_per_genre:
                    break
                if genres[i] == genre:
                    indeces = find_char(song, "\n")
                    indeces = indeces[:-1]
                    if len(indeces) == 0:
                        prompts.append(song[:prompt_length])
                        continue
                    index = indeces[random.randint(0, len(indeces)-1)]
                    prompts.append(song[index + 1:index + 1 + prompt_length])
            prompt_genre = [genre for i in range(prompts_per_genre)]
            prompt_encoded, _ = encode(prompts, alphabet, prompt_genre, unique_genres)
            X_prompt = prompt_encoded[:-1]

            pred = model(X_prompt, predict=predict, temp=0.45)
            pred = pred.permute(1,0,2)
            pred_decoded = decode(pred, alphabet)
            all_preds += pred_decoded
    all_tfs, pred_tfs = tf_idf(songs, all_preds)

    vocab_quality(songs, all_preds)

    for i, genre in zip(range(0, len(unique_genres) * prompts_per_genre, prompts_per_genre), unique_genres):
        predictions[genre] = pred_tfs[i:i + prompts_per_genre]

    songs_dict = defaultdict(lambda: [])
    for song, genre in zip(all_tfs, genres):
        songs_dict[genre].append(song)

    for genre in predictions.keys():
        avg = defaultdict(lambda: 0)
        print(f"--{genre}--")
        for s in predictions[genre]:
            for genre2 in predictions.keys():
                avg[genre2] += semantic_sim(songs_dict[genre2], s)
        for genre2 in predictions.keys():
            print(f"--{genre2} {avg[genre2]/prompts_per_genre}")
    """
    best = defaultdict(lambda: 0)
    best_i = defaultdict(lambda: 0)
    avg = defaultdict(lambda: [])
    for i, (song, genre) in enumerate(zip(all_tfs, genres)):
        sim = lyrical_uniqueness(songs_dict[genre], song)
        avg[genre].append(sim)
    for genre in avg.keys():
        print(f"--- {genre} ---")
        print(f"sim: {best[genre]}")
        print(songs[best_i[genre]])

        for genre2 in best_i.keys():
            print(f"Similarity to {genre2}: ", semantic_sim(songs_dict[genre2], all_tfs[best_i[genre]]))
        print(f"avg: {statistics.mean(avg[genre])} songs: {statistics.stdev(avg[genre])} ")
        print("\n")
    """
    print("\n")