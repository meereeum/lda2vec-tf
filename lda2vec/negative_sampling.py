import tensorflow as tf


class NegativeSampling():
	"""Negative sampling loss function.
	In natural language processing, especially language modeling, the number of
	words in a vocabulary can be very large.
	Therefore, you need to spend a lot of time calculating the gradient of the
	embedding matrix.
	By using the negative sampling trick you only need to calculate the
	gradient for a few sampled negative examples.
	The objective function is below:
	.. math::
	   f(x, p) = \\log \\sigma(x^\\top w_p) + \\
	   k E_{i \\sim P(i)}[\\log \\sigma(- x^\\top w_i)],
	where :math:`\sigma(\cdot)` is a sigmoid function, :math:`w_i` is the
	weight vector for the word :math:`i`, and :math:`p` is a positive example.
	It is approximeted with :math:`k` examples :math:`N` sampled from
	probability :math:`P(i)`, like this:
	.. math::
	   f(x, p) \\approx \\log \\sigma(x^\\top w_p) + \\
	   \\sum_{n \\in N} \\log \\sigma(-x^\\top w_n).
	Each sample of :math:`N` is drawn from the word distribution :math:`P(w)`.
	This is calculated as :math:`P(w) = \\frac{1}{Z} c(w)^\\alpha`, where
	:math:`c(w)` is the unigram count of the word :math:`w`, :math:`\\alpha` is
	a hyper-parameter, and :math:`Z` is the normalization constant.
	Args:
		x (~chainer.Variable): Batch of input vectors.
		t (~chainer.Variable): Vector of groundtruth labels.
		W (~chainer.Variable): Weight matrix.
		sampler (function): Sampling function. It takes a shape and returns an
			integer array of the shape. Each element of this array is a sample
			from the word distribution. A :class:`~chainer.utils.WalkerAlias`
			object built with the power distribution of word frequency is
			recommended.
		sample_size (int): Number of samples.
	See: `Distributed Representations of Words and Phrases and their\
		 Compositionality <http://arxiv.org/abs/1310.4546>`_
	.. seealso:: :class:`~chainer.links.NegativeSampling`.
	"""
	def __init__(self, embedding_size, vocabulary_size, sample_size):
		# via https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/examples/tutorials/word2vec/word2vec_basic.py

		self.vocab_size = vocabulary_size
		self.sample_size = sample_size

		self.W = tf.Variable( # embeddings
				tf.random_uniform([vocabulary_size, embedding_size], -1., 1.),
				name="word_embeddings")

		# Construct the variables for the NCE loss
		self.nce_weights = tf.Variable(
				tf.truncated_normal([vocabulary_size, embedding_size],
									stddev=1. / tf.sqrt(embedding_size)),
				name="nce_weights")
		self.nce_biases = tf.Variable(tf.zeros([vocabulary_size]),
									  name="nce_biases")

	def __call__(train_inputs, train_labels):
		embed = tf.nn.embedding_lookup(self.W, train_inputs)

		# Compute the average NCE loss for the batch.
		# tf.nce_loss automatically draws a new sample of the negative labels each
		# time we evaluate the loss.
		loss = tf.reduce_mean(
				tf.nn.nce_loss(self.nce_weights, self.nce_biases, embed,
							   train_labels, self.sample_size, self.vocab_size),
				name="nce_batch_loss")

		return loss
