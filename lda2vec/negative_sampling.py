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

	# IGNORE_LABEL_MAX = 1 # ignore any labels <=1 (OOV or skip)

	def __init__(self, embedding_size, vocabulary_size, sample_size, power=1.,
				 freqs=None, W_in=None):
		self.vocab_size = vocabulary_size
		self.sample_size = sample_size
		self.power = power
		self.freqs = freqs

		# via https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/examples/tutorials/word2vec/word2vec_basic.py

		init_width = 0.5 / embedding_size
		self.W = (tf.Variable( # word embeddings
				# tf.random_uniform([vocabulary_size, embedding_size], -1., 1.),
				tf.random_uniform([vocabulary_size, embedding_size],
								  -init_width, init_width),
				name="word_embeddings") if W_in is None else W_in)

		# Construct the variables for the NCE loss
		self.nce_weights = tf.Variable(
				tf.truncated_normal([vocabulary_size, embedding_size],
									stddev=tf.sqrt(1 / embedding_size)),
				name="nce_weights")
		self.nce_biases = tf.Variable(tf.zeros([vocabulary_size]),
									  name="nce_biases")

	def __call__(self, embed, train_labels):

		with tf.name_scope("negative_sampling"):
			# mask out skip or OOV
			# if switched on, this yields ...
			# UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.

			# mask = tf.greater(train_labels, NegativeSampling.IGNORE_LABEL_MAX)
			# # mask = tf.not_equal(train_labels, NegativeSampling.IGNORE_LABEL)
			# embed = tf.boolean_mask(embed, mask)
			# train_labels = tf.expand_dims(tf.boolean_mask(train_labels, mask), -1)
			train_labels = tf.expand_dims(train_labels, -1)

			# Compute the average NCE loss for the batch.
			# tf.nce_loss automatically draws a new sample of the negative labels each
			# time we evaluate the loss.
			# By default this uses a log-uniform (Zipfian) distribution for sampling
			# and therefore assumes labels are sorted - which they are!
			sampler = (self.freqs if self.freqs is None else
					   tf.nn.fixed_unigram_candidate_sampler(
							   train_labels, num_true=1, num_sampled=self.sample_size,
							   unique=True, range_max=self.vocab_size,
							   #num_reserved_ids=2, # skip or OoV
							   distortion=self.power, unigrams=list(self.freqs)))

			loss = tf.reduce_mean(
					tf.nn.nce_loss(self.nce_weights, self.nce_biases,
								   embed, # summed doc and context embedding
								   train_labels, self.sample_size, self.vocab_size,
								   sampled_values=sampler), # log-unigram if not specificed
					name="nce_batch_loss")
			# TODO negative sampling versus NCE
			# TODO uniform vs. Zipf with exponent `distortion` param
			#https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#log_uniform_candidate_sampler

		return loss
