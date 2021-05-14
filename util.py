import torch
import numpy as np
import pandas as pd
import re
import random


def load_songs(unique_genres: list, n_songs: int = 10000):
    '''
    Randomly samples the training data for a number of songs from different genres
        Parameters:
            unique_genres (list): The genres to load songs from
            n_songs (int): The number of songs to load in total (each genre will get n_songs/number of genres songs)
        Returns:
            songs: A list of songs
            genres: A list of genres to the songs in the songs list
    '''
    df = pd.read_csv("data/train.csv")

    # Remove unwanted songs based on language, genre and length
    df = df[df["Language"] == "en"]
    df = df[df.Genre.isin(unique_genres)]
    df['Length'] = df['Lyrics'].str.len()
    df = df[(df.Length > 400) & (df.Length < 2500)]

    # Remove duplicates
    df.drop_duplicates(subset=["Artist", "Song"], keep="first", inplace=True)
    df.drop_duplicates(subset=["Lyrics"], keep="first", inplace=True)

    # Extract genres
    genre = df["Genre"]

    # Shuffle songs, makes sure we get different songs every time
    df = df.sample(frac=1).reset_index(drop=True)

    # The number of songs to get from each genre
    minc = int(n_songs/len(unique_genres))

    # Extracts the correct amount of songs from each genre
    song_dfs = []
    for count, name in zip(genre.value_counts(), genre.value_counts().index):
        song = df[df["Genre"] == name]
        fraction = min(minc / len(song), 1)
        song = song.sample(frac=fraction).reset_index(drop=True)
        print(len(song), name)
        song_dfs.append(song)

    # Combines all the songs and reshuffles to make sure the genres are shuffled
    allsongs_df = pd.concat(song_dfs)
    allsongs_df = allsongs_df.sample(frac=1).reset_index(drop=True)
    songs = allsongs_df["Lyrics"].values.tolist()
    genres = allsongs_df["Genre"].values.tolist()

    # Runs cleanup on the songs
    songs = [clean_up(song) for song in songs]
    return songs, genres


def clean_up(song):
    '''
    Cleans up the song by removing unwanted characters
        Parameters:
            song (str): Song to perform cleanup on
        Returns:
            song (str): Cleaned song
    '''
    #Removes information between [ and ]
    filter = r"\[(.*?)\]"
    song = re.sub(filter, "", song)

    # Removes blacklisted characters, numbers and non-ascii chars
    blacklist = "<>@[]{}\x7f\t=+|~*%/\\_^;#$`§\x19\x13"
    song = "".join(char for char in song if char not in blacklist and not char.isnumeric())
    song = song.encode("ascii", "ignore").decode()
    return song


def get_vector(key, alphabet):
    '''
    Returns the index of a char in the alphabet
        Parameters:
            key (str): Key to find index of
            alphabet (list): Alphabet to find index in
    '''
    return alphabet.index(key)



def encode(data, alphabet, genres, unique_genres):
    '''
    Encodes songs for network input and targets
        Parameters:
            data (list): list containing songs in string format
            alphabet (list): Alphabet to encode by
            genres (list): Genres of songs in data
            unique_genres (list): Unique genres to encode by
        Returns:
            encoded (list): List of binary vectors for all songs with both one-hot chars and one-hot genres
            indeces (list): List of "hot" indeces in the one-hot char vector from encoded
    '''

    maxlen = len(max(data, key=len))
    encoded = torch.zeros(maxlen, len(data), len(alphabet) + len(unique_genres))
    indeces = torch.zeros(maxlen, len(data))
    for genre, (j, song) in zip(genres, enumerate(data)):
        for i, letter in enumerate(song):
            if i == maxlen:
                break
            encoded[i][j][alphabet.index(letter)] = 1 # One hot of char
            encoded[i][j][len(alphabet) + unique_genres.index(genre)] = 1 # One hot of genre
            indeces[i][j] = alphabet.index(letter) # Setting index of the one hot char vector
        # Padding song with end-of-song tokens to match maxlen
        for i in range(maxlen - len(song)):
            encoded[i + len(song)][j][len(alphabet) - 1] = 1
            indeces[i + len(song)][j] = len(alphabet) - 1
    return encoded, indeces


def decode(data, alphabet):
    '''
    Decodes songs from one-hot vectors
        Parameters:
            data (list): One-hot encoded vector to decode
            alphabet (list): Alphabet used during encoding
        Returns:
            (list): List of decoded songs
    '''
    return ["".join(alphabet[np.argmax(letter)] for letter in song if letter[-1] != 1) for song in data]


def get_alphabet(data):
    '''
    Gets alphabet used in a collection of songs
        Parameters:
            data (list): Collection of songs to get alphabet from
        Returns:
            (list): Alphabet used in data
    '''
    return sorted(list(set(char for song in data for word in song for char in word))) + ["EOS"]


def find_char(s, ch):
    '''
    Finds all occurrences of char in a string
        Parameters:
            s (str): String to search for char
            ch (str): char to find in s
        Returns:
            (list): List of indices where ch occurs in s
    '''
    return [i for i, ltr in enumerate(s) if ltr == ch]


def float_range(start, stop, step):
    '''
    Same as pythons range, but with floating numbers
        Parameters:
            start (float): start of range
            stop (float): end of range
            step (float): interval size
        Returns:
            (generator): floating range
    '''
    while start < stop:
        yield float(start)
        start += step


def generate_genre_prompts(songs: list, genres: list, unique_genres: list, prompts_per_genre: int = 1, prompt_length: int = 16):
    '''
    Genreates genre appropriate prompts
        Parameters:
            songs (list): Songs to sample prompts from
            genres (list): Genres to retrieve prompts for
            unique_genres (list): Unique genres used in songs
            prompts_per_genre (int): Number of prompts for each genre
            prompt_length (int): Length of prompts
        Returns:
            all_prompts (list): List of prompts
            genre_prompts (list): Genres of the prompts
    '''
    all_prompts = []
    genre_prompts = []
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
                    genre_prompts.append(genre)
                    continue
                index = indeces[random.randint(0, len(indeces) - 1)]
                prompts.append(song[index + 1:index + 1 + prompt_length])
                genre_prompts.append(genre)
        all_prompts += prompts

    return all_prompts, genre_prompts


def prompt_network(model, prompt_text, prompt_genres, alphabet, unique_genres, temperature=0.45, predict=2000):
    '''
    Prompts the network and generates new songs
        Parameters:
            model (LyricSTM): model to use for inference
            prompt_text (list): Prompt text to use for generating songs
            prompt_genres (list): Genres to generate songs for
            alphabet (list): Alphabet used in model
            unique_genres (list): Genres model is trained on
            temperature (float): Temperature value used in softmax selection of chars
            predict (int): Max length of predicted songs
        Returns:
            (list): Generated songs
    '''
    prompt_encoded, _ = encode(prompt_text, alphabet, prompt_genres, unique_genres)
    X_prompt = prompt_encoded[:-1]
    pred = model(X_prompt, predict=predict, temp=temperature)
    pred = pred.permute(1, 0, 2)
    return decode(pred, alphabet)
