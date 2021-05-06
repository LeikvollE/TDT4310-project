import torch
import torch.nn as nn
import csv
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from  torch.nn.utils.rnn import pad_sequence


class LyricSTM(nn.Module):
    def __init__(self, n_hidden: int, feature_size: int):
        super().__init__()
        self.feature_size = feature_size
        self.n_hidden = n_hidden

        self.lstm = nn.LSTMCell(self.feature_size, self.n_hidden)
        self.linear = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.n_hidden, 128),
            nn.SELU(),
            nn.Linear(128, feature_size)
        )

    def forward(self, x, predict=0):
        outputs = []
        n_samples = x.size(1)

        h_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)

        for i in range(x.size()[0]):
            h_t, c_t = self.lstm(x[i], (h_t, c_t))
            output = self.linear(h_t)
            outputs.append(output)

        for i in range(predict):
            max_idx = torch.argmax(output, 1, keepdim=True)
            one_hot = torch.FloatTensor(output.shape)
            one_hot.zero_()
            one_hot.scatter_(1, max_idx, 1)
            h_t, c_t = self.lstm(one_hot, (h_t, c_t))
            output = self.linear(h_t)
            outputs.append(output)
            if torch.argmax(output[0]) == self.feature_size - 1:
                break

        outputs = torch.stack(outputs, dim=0)
        return outputs


def load_songs():
    songs = []
    with open("./data/taylor_swift_lyrics.csv", encoding='windows-1252') as file:
        csv_reader = csv.reader(file)
        last_song = ""
        for line in csv_reader:
            if not line[2] == last_song:
                songs.append("")
                last_song = line[2]
            songs[-1] += line[4]
            songs[-1] += '\n'

    return songs[1:]


def get_vector(key, alphabet):
    return alphabet.index(key)


def encode(data, alphabet, pad=True):
    maxlen = len(max(data, key=len))
    #maxlen = min(1000, maxlen)
    encoded = torch.zeros(maxlen+1, len(data), len(alphabet))
    indeces = torch.zeros(maxlen+1, len(data))
    for j, song in enumerate(data):
        for i, letter in enumerate(song):
            if i > maxlen:
                continue
            encoded[i][j][alphabet.index(letter)] = 1
            indeces[i][j] = alphabet.index(letter)
        for i in range(maxlen-min(len(song), maxlen)+1):
            encoded[i+min(len(song), maxlen)][j][-1] = 1
            indeces[i+min(len(song), maxlen)][j] = len(alphabet)-1
    return encoded, indeces


def decode(data, alphabet):
    return ["".join(alphabet[np.argmax(letter)] for letter in song) for song in data]


def get_alphabet(data):
    return list(set(char for song in data for word in song for char in word)) + ["EOS"]


if __name__ == "__main__":
    songs = load_songs()
    alphabet = get_alphabet(songs)
    encoded, targets = encode(songs, alphabet)
    X = encoded[:-1]
    Y = targets[1:]
    Y = Y.long()
    Y = Y.permute(1,0)
    print(Y)

    prompt = ["When you think "]
    prompt_encoded, _ = encode(prompt, alphabet)
    X_prompt = prompt_encoded
    model = LyricSTM(n_hidden=256, feature_size=len(alphabet))

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    n_steps = 1000
    for i in range(n_steps):
        print("Step", i)

        def closure():
            optimizer.zero_grad()
            out = model(X)
            out = out.permute(1,2,0)
            loss = criterion(out, Y)
            print(loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)

        if i > 3:
            with torch.no_grad():
                predict = 5000
                pred = model(X_prompt, predict=predict)
                pred = pred.permute(1,0,2)
                print(pred)
                print(pred.size())
                print(decode(pred, alphabet))
