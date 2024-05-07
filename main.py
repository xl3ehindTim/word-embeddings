from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import numpy as np
import json

def load_embeddings(filename):
	"""
   	Loads pre-generated word embeddings from a file.
  	"""
	with open(filename, 'r') as file:
		return json.load(file) 

def reduce_dimensions(vectors, n_components=2):
	"""
	Reduces the dimensionality of word embeddings using t-SNE
	"""
	tsne = TSNE(n_components=n_components)
	return tsne.fit_transform(np.array(vectors))

def figure(method, words, vectors):
	"""
	Draw 2D figure for word embeddings
	"""
	plt.figure(figsize=(8, 6))

	for word, vector in zip(words, vectors):
		plt.scatter(vector[0], vector[1], label=word)
		plt.annotate(word, xy=[vector[0], vector[1]], xytext=(0, 5), textcoords="offset points", fontsize=12)

	plt.xlabel("Dimension 1")
	plt.ylabel("Dimension 2")
	plt.title(f"{method} word embeddings")
	plt.grid(True)
	plt.show()

embeddings = {
  'CountVectorizer': load_embeddings("data/count.json"),
  'Word2Vec': load_embeddings("data/word2vec.json"),
  'TF-IDF': load_embeddings("data/tf-idf.json"),
  'spaCy': load_embeddings("data/spacy.json"),
}

for method in embeddings.keys():
	data = embeddings[method]
	
	words = list(data.keys())
	vectors = list(data.values())

	# Reduce dimensions
	embeddings_reduced = reduce_dimensions(vectors)

	# Draw figure
	figure(method=method, words=words, vectors=embeddings_reduced)

