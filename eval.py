import torch
import numpy as np
from util import load_songs, get_alphabet, float_range, generate_genre_prompts, prompt_network
from collections import defaultdict
import nltk
from Network import LyricSTM
from SingleGenreNetwork import SingleLyricSTM as LyricSTM

import statistics
from sklearn.feature_extraction.text import TfidfVectorizer

def tf_idf(songs, preds):
    '''
    Calculates TF-IDF vectors based on source songs and generated songs using sklearn
        Parameters:
            songs (list): Source songs from train
            preds (list): Generated songs
        Returns:
            (list): TF-IDF vectors of source songs
            (list): TF-IDF vectors of generated songs
    '''
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(songs + preds).toarray()
    return X[:-len(preds)], X[-len(preds):]

def semantic_sim(tf_vectors, pred_vector):
    '''
    Calculates the semantic relatedness score between a group of songs and a single song
        Parameters:
            tf_vectors (list): List of TF-IDF vectors from base songs
            pred_vector (list): TF-IDF representation of song to calculate score for
    '''
    scores = []
    for vector in tf_vectors:
        scores.append(cosine_dist(vector, pred_vector))
    return statistics.mean(scores)

def cosine_dist(song1, song2):
    '''
    Calculates the cosine distance between two vectors
        Parameters:
            song1 (list): Vector 1
            song2 (list): Vector 2
    '''
    return np.dot(song1, song2)/(np.linalg.norm(song1) * np.linalg.norm(song2))

def lyrical_uniqueness(tf_vectors, pred_vector):
    '''
        Calculates the lyrical similarity score between a group of songs and a single song
            Parameters:
                tf_vectors (list): List of TF-IDF vectors from base songs
                pred_vector (list): TF-IDF representation of song to calculate score for
        '''
    scores = []
    for vector in tf_vectors:
        scores.append(cosine_dist(vector, pred_vector))
    return max(scores)

def vocab_quality(songs, preds):
    '''
    Calculates vocaularity size of generated songs, and how many words in the vocabulary also exists in source material
        Parameters:
            songs (list): Source songs
            preds (list): Generated songs
    '''
    unique_words = set() # set of all unique tokens in the songs
    for song in songs:
        unique_words.update(nltk.word_tokenize(song))

    all_words = set() # set of all tokens in the generated lyrics
    count = 0
    bad_words = 0
    for pred in preds:
        tokens = nltk.word_tokenize(pred)
        all_words.update(tokens)
        count += len(tokens)
        bad_words += len([word for word in tokens if word not in unique_words]) 
    correct_words = len([word for word in all_words if word in unique_words])
    print([word for word in all_words if word not in unique_words]) # print all incorrect tokens
    print(len([word for word in all_words if word not in unique_words])) # print number of incorrect tokens

    p = str(correct_words * 100 / len(all_words))[:5] + "%" # fraction of incorrect tokens in vocabulary
    p2 = str(bad_words * 100 / count)[:5] + "%" # fraction of incorrect tokens in generated lyrics
    print(
        f"Model used {len(all_words)} tokens, of which {correct_words} are true words ({p}). Total incorrect fraction: ({p2})")


if __name__ == "__main__":

    unique_genres = ["Jazz"]#sorted(["Rock", "Metal", "Country", "Jazz", "Hip-Hop"])  # sorted(list(set(genres)))

    songs, genres = load_songs(unique_genres)
    songs_dict = defaultdict(lambda: [])
    for song, genre in zip(songs, genres):
        songs_dict[genre].append(song)
    alphabet = get_alphabet(songs)

    model = torch.load("models/newnetonegenre2/model_checkpoint_10")
    #model = LyricSTM(model_.n_hidden, model_.feature_size, model_.alphabet, )
    #model.load_state_dict(model_.state_dict())

    prompts_per_genre = 50
    prompt_length = 16
    predict = 2000
    temp = 0.4
    predictions = {}
    all_preds = []
    texts = defaultdict(lambda: [])
    with torch.no_grad():
        for genre in unique_genres: # generate prompts for all genres
            prompts, prompt_genre = generate_genre_prompts(songs, genres, ["Jazz"], prompts_per_genre,
                                                           prompt_length=prompt_length)
            pred_decoded = prompt_network(model, prompts, prompt_genre, model.alphabet, unique_genres,
                                          temperature=temp, predict=predict)
            all_preds.extend(pred_decoded)
            texts[genre].extend([prompts[i] + song[len(prompts[i]):] for i, song in enumerate(pred_decoded)])
        vocab_quality(songs, all_preds) # print vocab quality
    all_tfs, pred_tfs = tf_idf(songs, all_preds) # encode as TF-IDF
    songs_dict = defaultdict(lambda: [])
    for song, genre in zip(all_tfs, genres): 
        songs_dict[genre].append(song)

    for genre, i in zip(unique_genres, range(0, len(unique_genres) * prompts_per_genre, prompts_per_genre)):
        predictions[genre] = pred_tfs[i:i + prompts_per_genre]
        print(len(predictions[genre]), len(texts[genre]))

    for genre in unique_genres: # evaluate lyrics genre by genre
        genre_sems = defaultdict(lambda: [])
        best_sem = 0
        best_sem_i = 0
        w_sem = 20
        w_sem_i = 0
        genre_lyrs = []
        print("---" + genre + "---")
        for i, s in enumerate(predictions[genre]):
            genre_lyrs.append(lyrical_uniqueness(songs_dict[genre], s)) # store lyrical similarity
            for genre2 in predictions.keys():
                sem = semantic_sim(songs_dict[genre2], s)
                genre_sems[genre2].append(sem)
                if sem > best_sem: # store indeces of best and worst songs (semantic relatedness)
                    best_sem = sem
                    best_sem_i = i
                if sem < w_sem:
                    w_sem = sem
                    w_sem_i = i
        #print statistics:
        print("Lyrical similarity score (for this 1 song): ",
              lyrical_uniqueness(songs_dict[genre], predictions[genre][best_sem_i]),
              " Semantic relatedness (highest for this genre): ", best_sem)
        print("Lyrical similarity score (for this other song): ",
              lyrical_uniqueness(songs_dict[genre], predictions[genre][w_sem_i]),
              " Semantic relatedness (lowest for this genre): ", w_sem)
        print("Avg lyrical sim for this genre", statistics.mean(genre_lyrs), " std ", statistics.stdev(genre_lyrs),
              "Max: ", max(genre_lyrs), "Min: ", min(genre_lyrs))
        for genre2 in predictions.keys():
            print(f"Semantic similarity to {genre2} {sum(genre_sems[genre2]) / prompts_per_genre}")

        # print(r"\textbf{" + texts[genre][best_sem_i]) # print song

    print("\n")
