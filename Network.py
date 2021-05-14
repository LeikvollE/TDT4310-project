import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from util import encode, decode, load_songs, get_alphabet, prompt_network
import numpy as np


class LyricSTM(nn.Module):
    def __init__(self, n_hidden: int, feature_size: int, alphabet: list, unique_genres: list):
        '''
            Parameters:
                n_hidden (int): The size of the hidden and cell state of the lstm cells
                feature_size (int): The input size to the network, should be size of alphabet + unique genres
                alphabet (list): The alphabet used by the model
                unique_genres (list): The different genres the model can use to generate lyrics
        '''
        super().__init__()
        self.feature_size = feature_size
        self.n_hidden = n_hidden
        self.alphabet = alphabet
        self.unique_genres = unique_genres

        self.lstm = nn.LSTMCell(self.feature_size, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.n_hidden, 512),
            nn.SELU(),
            nn.Linear(512, len(alphabet))
        )
        self.soft = nn.Softmax(dim=1)

    def forward(self, x, predict=0, temp=0.45):
        '''
        Performs forward pass of network
            Parameters:
                x (list): prompts for network to generate on
                predict (int): How many chars to generate after prompt. Use value > 0 during inference, and 0 during training
                temp (float): Temperature to use during softmax, only used if predict > 0
        '''
        outputs = []
        n_samples = x.size(1)

        # Store genres for use during inference
        genres = x[0,:,-len(self.unique_genres):]

        # Initialise hidden and cell state for both LSTM-cells
        h_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        h2_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c2_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)

        # Iterate over all timesteps
        for i in range(x.size()[0]):
            h_t, c_t = self.lstm(x[i], (h_t, c_t))
            h2_t, c2_t = self.lstm2(h_t, (h2_t, c2_t))
            output = self.linear(h2_t)
            outputs.append(output)

        # Inference
        for i in range(predict):
            
            output = self.soft(output / temp)
            idx = torch.zeros(n_samples, 1, dtype=torch.int64)

            # Select random char based on output probability distribution
            for i, p in enumerate(output):
                hot = np.random.choice(np.arange(len(p)), p=p.numpy())
                idx[i][0] = hot

            # One-hot encode selected character
            one_hot = torch.FloatTensor(output.shape)
            one_hot.zero_()
            one_hot.scatter_(1, idx, 1)
            outputs.append(one_hot)

            # Append genre and send through lstm and dense layer
            one_hot = torch.cat((one_hot, genres), dim=1)
            h_t, c_t = self.lstm(one_hot, (h_t, c_t))
            h2_t, c2_t = self.lstm2(h_t, (h2_t, c2_t))
            output = self.linear(h2_t)

            # If all songs in the batch have EOS-token, no need to continue generating lyrics
            if all(torch.argmax(one_hot[j]) == len(self.alphabet) - 1 for j in range(len(one_hot))):
                break

        outputs = torch.stack(outputs, dim=0)
        return outputs


if __name__ == "__main__":
    minibatch_size = 32
    model_file = "models/fixednet/model"
    unique_genres = sorted(["Rock", "Metal", "Country", "Jazz", "Hip-Hop"])

    # Load songs and generate alphabet
    songs, genres = load_songs(unique_genres)
    alphabet = get_alphabet(songs)

    prompt = ["When you think "]*len(unique_genres)
    prompt_genre = unique_genres

    model = LyricSTM(n_hidden=256, feature_size=len(alphabet) + len(unique_genres), alphabet=alphabet, unique_genres=unique_genres)

    # Model uses cross entropy loss and adam optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Main training loop
    n_steps = 10
    for i in range(n_steps):
        print("Step", i)
        for j in tqdm(range(0, len(songs), minibatch_size)):
            batch_songs, batch_genres = songs[j: j+minibatch_size], genres[j: j+minibatch_size]
            batch_encoded, batch_targets = encode(batch_songs, alphabet, batch_genres, unique_genres)
            # Input excludes last character, as that should be EOS-token and should only be predicted not predicted on
            X = batch_encoded[:-1]
            # Targets excludes first char
            Y = batch_targets[1:]
            Y = Y.long()
            # Get data dimensions on format accepted by torch lstm-cell
            Y = Y.permute(1, 0)

            def closure():
                optimizer.zero_grad()
                out = model(X, temp=0.45)
                # Reformat output dimensions to format accepted by torch cross entropy
                out = out.permute(1,2,0)
                # Calculate loss
                loss = criterion(out, Y)
                loss.backward()
                return loss
            optimizer.step(closure)

        # Save checkpoint every epoch
        torch.save(model, model_file+"_checkpoint_"+str(i))

        # Generate prompts after some epochs to see how network is progressing
        if i >= 0:
            with torch.no_grad():
                pred_songs = prompt_network(model, prompt, prompt_genre, alphabet, unique_genres)
                for i, s in enumerate(pred_songs):
                    print(prompt_genre[i])
                    print(prompt[i] + s[len(prompt[i]):])

    torch.save(model, model_file+"_final")
