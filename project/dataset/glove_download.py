import bcolz
import pickle
import torch
from tqdm import tqdm
# init glove here
glove_path = '/data5/shashank2000'
vectors = bcolz.open(f'{glove_path}/6B.50.dat')[:]
words = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))
num_hidden = 50
glove = {w: vectors[word2idx[w]] for w in words}

float_tensor = torch.randn(len(word2idx), num_hidden)
zero_v = torch.zeros(num_hidden)
rand_v = torch.randn(num_hidden)
for word in tqdm(word2idx):
    i = word2idx[word]
    val = torch.from_numpy(glove[word])
    float_tensor[i] = val
torch.save(float_tensor, '/home/shashank2000/synced/project/emb_weights_1.data')
