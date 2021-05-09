import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import Counter
from util import encode, decode, load_songs, get_alphabet, prompt_network
import numpy as np

class LyricSTM(nn.Module):
    def __init__(self, n_hidden: int, feature_size: int, alphabet: list, unique_genres: list):
        super().__init__()
        self.feature_size = feature_size
        self.n_hidden = n_hidden
        self.alphabet = alphabet

        self.lstm = nn.LSTMCell(self.feature_size, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.n_hidden + len(unique_genres), 512),
            nn.SELU(),
            nn.Linear(512, len(alphabet))
        )
        self.soft = nn.Softmax(dim=1)

    def forward(self, x, predict=0, temp=0.45):
        outputs = []
        n_samples = x.size(1)

        genres = x[0,:,-len(unique_genres):]

        h_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)

        h2_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c2_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)

        for i in range(x.size()[0]):
            h_t, c_t = self.lstm(x[i][:,:-len(unique_genres)], (h_t, c_t))
            h2_t, c2_t = self.lstm2(h_t, (h2_t, c2_t))
            #output = self.linear(h2_t)
            output = self.linear(torch.cat((h2_t, x[i][:,-len(unique_genres):]), dim=1))
            outputs.append(output)

        for i in range(predict):
            
            output = self.soft(output / temp)
            idx = torch.zeros(n_samples, 1, dtype=torch.int64)

            for i, p in enumerate(output):
                hot = np.random.choice(np.arange(len(p)), p=p.numpy())
                idx[i][0] = hot

            one_hot = torch.FloatTensor(output.shape)
            one_hot.zero_()
            one_hot.scatter_(1, idx, 1)
            outputs.append(one_hot)

            #one_hot = torch.cat((one_hot, genres), dim=1)
            h_t, c_t = self.lstm(one_hot, (h_t, c_t))
            h2_t, c2_t = self.lstm2(h_t, (h2_t, c2_t))
            #output = self.linear(h2_t)
            output = self.linear(torch.cat((h2_t, genres), dim=1))


            if all(torch.argmax(one_hot[j]) == len(self.alphabet) - 1 for j in range(len(one_hot))):
                break

        outputs = torch.stack(outputs, dim=0)
        return outputs


if __name__ == "__main__":
    minibatch_size = 32
    model_file = "models/newnetonegenre/model"

    unique_genres = ["Jazz"]#sorted(["Rock", "Metal", "Country", "Jazz", "Hip-Hop"])

    songs, genres = load_songs(unique_genres)
    alphabet = get_alphabet(songs)

    prompt = ["When you think "]*len(unique_genres)
    prompt_genre = unique_genres

    model = LyricSTM(256, len(alphabet), alphabet, unique_genres)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    n_steps = 100
    for i in range(n_steps):
        print("Step", i)
        for j in tqdm(range(0, len(songs), minibatch_size)):
            batch_songs, batch_genres = songs[j: j+minibatch_size], genres[j: j+minibatch_size]
            batch_encoded, batch_targets = encode(batch_songs, alphabet, batch_genres, unique_genres)
            X = batch_encoded[:-1]
            Y = batch_targets[1:]
            Y = Y.long()
            Y = Y.permute(1, 0)

            def closure():
                optimizer.zero_grad()
                out = model(X, temp=0.45)
                out = out.permute(1,2,0)
                loss = criterion(out, Y)
                loss.backward()
                return loss
            optimizer.step(closure)
        torch.save(model, model_file+"_checkpoint_"+str(i))
        if i >= 0:
            with torch.no_grad():
                pred_songs = prompt_network(model, prompt, prompt_genre, alphabet, unique_genres)
                for i, s in enumerate(pred_songs):
                    print(prompt_genre[i])
                    print(prompt[i] + s[len(prompt[i]):])

    torch.save(model, model_file+"_final")
