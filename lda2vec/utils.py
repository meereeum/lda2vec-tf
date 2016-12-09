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



# def most_similar(embeddings, word_index):
# 	input_vector = tf.nn.embedding_lookup(embeddings, word_index)
# 	similarities = tf.matmul(embeddings, input_vector)
# 	return similarities


def chunks(n, *args):
	"""Yield successive n-sized chunks from l."""
	# From stackoverflow question 312443
	keypoints = []
	for i in range(0, len(args[0]), n):
		keypoints.append((i, i + n))
	random.shuffle(keypoints)
	for a, b in keypoints:
		yield [arg[a: b] for arg in args]


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
