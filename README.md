# TDT4310-project

Multi-genre lyric generation for project in TDT4310

## How to run
Download train.csv from https://www.kaggle.com/mateibejan/multilingual-lyrics-for-genre-classification and place it in a folder called data.
To create your own network, run Network.py. To save the trained model to a specific folder, change the model_file parameter in Network.py

To prompt the network, change the load call in prompt.py to load your model, and then run the file. Eval.py can be used to print the quantiatative measures vocabulary size and correctnes, lyrical similarity, and semantic relatedness.
