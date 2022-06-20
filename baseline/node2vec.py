import pickle
from node2vec import Node2Vec
from gensim.models import Word2Vec
import sys

pkl_file = open('graph.pkl', 'rb')
G = pickle.load(pkl_file)
pkl_file.close()

node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=150, workers=1)
model = node2vec.fit(window=10, min_count=1, batch_words=4)
model.save('node2vec.model')
model.wv.save_word2vec_format('node2vec.vector')

model = Word2Vec.load("node2vec.model")
# list of node IDs
node_ids = model.wv.index2word
# numpy.ndarray of size number of nodes*embeddings dimensionality
node_embeddings_orig = model.wv.vectors.tolist()

node_embeddings_dict = {}
num = 0
for i in node_ids:
    node_embeddings_dict[i] = node_embeddings_orig[num]
    num += 1

output = open('node_embeddings_dict.pkl', 'wb')
pickle.dump(node_embeddings_dict, output, 2)
output.close()
