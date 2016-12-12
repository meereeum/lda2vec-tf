import os
import random

import numpy as np
import tensorflow as tf


# def print_(var, name=None, first_n=5, summarize=5):
def print_(var, name: str, first_n=10, summarize=5):
	"""Util for debugging, by printing values of tf.Variable `var` during training"""

	# name = (next(k for k, v in globals().items() if v == var) # get name automagically
	# 		if name is None else name) # TODO make work for list ?

	# name = (next(k for k, v in globals().items() if id(v) == id(var))
	# 		if name is None else name)
	# print(name)
	# return ([k for k, v in globals().items() if id(v) == id(var)]
	# 		if name is None else name)

	try:
		return tf.Print(var, [var], '{}: '.format(name), first_n=first_n,
						summarize=summarize)

	except(TypeError): # variables are already in a list
		return tf.Print(var, var, '{}: '.format(name), first_n=first_n,
						summarize=summarize)


def chunks(n, *args):
	"""Yield successive n-sized chunks from l."""
	# From stackoverflow question 312443
	keypoints = []
	for i in range(0, len(args[0]), n):
		keypoints.append((i, i + n))
	random.shuffle(keypoints)
	for a, b in keypoints:
		yield [arg[a: b] for arg in args]


def save_embeddings(meta_graph, outdir=None):
	"""Load trained model and save embeddings"""
	outdir = ("results_{}".format(os.path.basename(meta_graph))
			  if outdir is None else outdir)

	m = LDA2Vec(47, 47, meta_graph=meta_graph)

	topics = m.sesh.run(m.topics)
	doc_embeds = m.sesh.run(m.doc_embeds)
	doc_proportions = m.sesh.run(m.doc_proportions)
	word_embeds = m.sesh.run(m.word_embeds)

	for f, arr in (("topics", topics),
				   ("doc_embeds", doc_embeds),
				   ("doc_propotions", doc_proportions),
				   ("word_embeds", word_embeds)):

		np.save(os.path.join(outdir, f), arr)


# class MovingAverage():
# 	def __init__(self, lastn=100):
# 		self.points = np.array([])
# 		self.lastn = lastn

# 	def add(self, x):
# 		self.points = np.append(self.points, x)

# 	def mean(self):
# 		return np.mean(self.points[-self.lastn:])

# 	def std(self):
# 		return np.std(self.points[-self.lastn:])

# 	def get_stats(self):
# 		return (np.mean(self.points[-self.lastn:]),
# 				np.std(self.points[-self.lastn:]))


# def most_similar(embeddings, word_index):
# 	input_vector = tf.nn.embedding_lookup(embeddings, word_index)
# 	similarities = tf.matmul(embeddings, input_vector)
# 	return similarities
