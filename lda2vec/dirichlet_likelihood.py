import tensorflow as tf


def dirichlet_likelihood(weights, alpha=None):
	""" Calculate the log likelihood of the observed topic proportions.
	A negative likelihood is more likely than a negative likelihood.
	Args:
		weights (chainer.Variable): Unnormalized weight vector. The vector
			will be passed through a softmax function that will map the input
			onto a probability simplex.
		alpha (float): The Dirichlet concentration parameter. Alpha
			greater than 1.0 results in very dense topic weights such
			that each document belongs to many topics. Alpha < 1.0 results
			in sparser topic weights. The default is to set alpha to
			1.0 / n_topics, effectively enforcing the prior belief that a
			document belong to every topics at once.
	Returns:
		~chainer.Variable: Output loss variable.
	"""
	n_topics = weights.get_shape()[1].value

	if alpha is None:
		alpha = 1 / n_topics

	# log_proportions = tf.log(tf.nn.softmax(weights))
	log_proportions = tf.nn.log_softmax(weights)

	loss = (alpha - 1) * log_proportions

	return -tf.reduce_sum(loss) # log-sum-exp
