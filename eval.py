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

def tf_idf(songs, pred):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(songs + [pred]).toarray()
    return X[:-1], X[-1]

def semantic_sim(gts, pred):
    tf_vectors, pred_vector = tf_idf(gts, pred)
    scores = []
    for vector in tf_vectors:
        scores.append(cosine_dist(vector, pred_vector))
    return statistics.mean(scores)

def cosine_dist(song1, song2):
    return np.dot(song1, song2)/(np.linalg.norm(song1) * np.linalg.norm(song2))

def lyrical_uniqueness(gts, pred):
    tf_vectors, pred_vector = tf_idf(gts, pred)
    scores = []
    for vector in tf_vectors:
        scores.append(cosine_dist(vector, pred_vector))
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
    model_file = "models/fixednet/model"

    songs, genres = load_songs()
    songs_dict = defaultdict(lambda: [])
    for song, genre in zip(songs, genres):
        songs_dict[genre].append(song)
    alphabet = get_alphabet(songs)
    unique_genres = sorted(list(set(genres)))

    prompt = ["When you think "]*len(unique_genres)
    prompt_genre = unique_genres
    prompt_encoded, _ = encode(prompt, alphabet, prompt_genre, unique_genres)
    X_prompt = prompt_encoded[:-1]
    model_ = torch.load("models/fixednet/model_checkpoint_5")
    model = LyricSTM(model_.n_hidden, model_.feature_size, model_.alphabet)
    model.load_state_dict(model_.state_dict())


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
            predictions[genre] = pred_decoded
            all_preds += pred_decoded

    vocab_quality(songs, all_preds)

    for genre in predictions.keys():
        avg = defaultdict(lambda: 0)
        print(f"--{genre}--")
        for s in predictions[genre]:
            for genre2 in predictions.keys():
                avg[genre2] += semantic_sim(songs_dict[genre2], s)
        for genre2 in predictions.keys():
            print(f"--{genre2} {avg[genre2]/prompts_per_genre}")

    print("\n")