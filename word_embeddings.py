import numpy as np
from utils import save_feature

embeddings_dict = {}
with open("word2vec/glove/glove.6B.300d.txt", 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector
        
save_feature(embeddings_dict, 'glove_300d')
