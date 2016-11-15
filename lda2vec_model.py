from lda2vec import dirichlet_likelihood
from lda2vec import EmbedMixture
from lda2vec import NegativeSampling
# from lda2vec.utils import move

import numpy as np
import tensorflow as tf


class LDA2Vec():

	DEFAULTS = {
		"n_document_topics": 10,
		"n_units": 256, # embedding size

		"batch_size": 128,
		"learning_rate": 0.1,
		"dropout_ratio": 0.5, # keep_prob
		"word_dropout": 0.0, # keep_prob
		"power": 0.75, # negative sampling power - TODO ?
		"temperature": 1.0,
		"lmbda": 200., # strength of Dirichlet prior
		"n_samples": 15 # num negative samples
	}
	# RESTORE_KEY = "to_restore" TODO?

	def __init__(self, n_corpus, n_documents, n_vocab, #train=True,counts=None
				 d_hyperparams={}, save_graph_def=True, log_dir="./log"):

		self.__dict__.update(LDA2Vec.DEFAULTS, **d_hyperparams)

		# build graph
		self.mixture = EmbedMixture(n_documents, n_document_topics, n_units,
									keep_prob=dropout, temperature=temperature)
		self.sampler = NegativeSampling(n_units, n_vocab, n_samples)
		self._buildGraph()

		self.sesh = tf.Session()
		self.sesh.run(tf.initialize_all_variables())

		if save_graph_def: # tensorboard
			self.logger = tf.train.SummaryWriter(log_dir, self.sesh.graph)

	@property
	def step(self):
		"""Train step"""
		return self.global_step.eval(session=self.sesh)

	def _buildGraph(self):

		self.pivot_idxs = tf.placeholder([None, # enables variable batch_size
										 1])
		self.target = tf.placeholder([None,
							 ])

		pivot = tf.nn.embedding_lookup(self.sampler.W, # word embeddings
										pivot_idxs)

		self.dropout = tf.placeholder_with_default(1., shape=[], name="dropout")

		context = (tf.nn.dropout(doc, self.dropout) +
					tf.nn.dropout(pivot, self.dropout))

		# TODO penalize weights, loss
		loss_word2vec = self.sampler(context, self.target)

		# batchsize / total
		fraction = self.batch_size / n_corpus
		prior = self.prior() # dirichlet log-likelihood
		# loss is proportional to minibatch fraction
		loss_lda = lmbda * fraction * prior

		# optimization
		global_step = tf.Variable(0, trainable=False)
		# TODO combine ?
		train_op_word2vec = tf.contrib.layers.optimize_loss(
			loss_word2vec, global_step, learning_rate, "Adam", clip_gradient=5.,
			name="train_op_word2vec")
		train_op_lda = tf.contrib.layers.optimize_loss(
			loss_lda, global_step, learning_rate, "Adam", clip_gradient=5.,
			name="train_op_lda")

		return (pivot_idxs, target, dropout, global_step, loss_word2vec, loss_lda,
				train_op_word2vec, train_op_lda)


	def prior(self, alpha=None):
		# defaults to uniform pror (1/n_topics)
		return dirichlet_likelihood(self.mixture.weights, alpha=alpha)


	def fit_partial(self, rdoc_ids, rword_indices, window=5,
					update_only_docs=False):
		# TODO: placeholders, etc
		doc_ids, word_indices = move(self.xp, rdoc_ids, rword_indices)
		pivot_idx = next(move(self.xp, rword_indices[window: -window]))
		# TODO feed pivot_idx to placeholder
		# pivot = F.embed_id(pivot_idx, self.sampler.W)
		if update_only_docs:
			# TODO ?
			pivot = tf.Variable(pivot, trainable=False)
			# pivot.unchain_backward()
		doc_at_pivot = rdoc_ids[window: -window]

		# TODO placeholder or no ?
		doc = self.mixture(next(move(self.xp, doc_at_pivot)),
							update_only_docs=update_only_docs)

		loss = 0.0
		start, end = window, rword_indices.shape[0] - window
		# context = (tf.nn.dropout(doc, self.dropout) +
		# 			tf.nn.dropout(pivot, self.dropout))

		for frame in range(-window, window + 1):
			# Skip predicting the current pivot
			if frame == 0:
				continue
			# Predict word given context and pivot word
			# The target starts before the pivot
			targetidx = rword_indices[start + frame: end + frame]
			doc_at_target = rdoc_ids[start + frame: end + frame]
			doc_is_same = doc_at_target == doc_at_pivot
			rand = np.random.uniform(0, 1, doc_is_same.shape[0])
			mask = (rand > self.word_dropout).astype("bool")
			weight = np.logical_and(doc_is_same, mask).astype("int32")
			# If weight is 1.0 then targetidx
			# If weight is 0.0 then -1
			targetidx = targetidx * weight + -1 * (1 - weight)

			# TODO - feed/fetch etc
			target, = move(self.xp, targetidx)

			loss = self.sampler(context, target)
			loss.backward()
			if update_only_docs:
				# Wipe out any gradient accumulation on word vectors
				self.sampler.W.grad *= 0.0
		return loss.data
