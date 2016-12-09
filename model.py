from datetime import datetime
import os
import sys

import numpy as np
import tensorflow as tf

from lda2vec import dirichlet_likelihood
from lda2vec import EmbedMixture
from lda2vec import NegativeSampling
from lda2vec import utils


class LDA2Vec():

	DEFAULTS = {
		"n_document_topics": 15,
		"n_embedding": 100, # embedding size

		"batch_size": 4096,##264,
		"window": 5,
		"learning_rate": 1.0,#0.1,
		"dropout_ratio": 0.8, # keep_prob
		"word_dropout": 0.8, #1.

		"power": 0.75, # unigram sampler distortion TODO - not implemented
		"n_samples": 50,#15, # num negative samples

		"temperature": 1., # embed mixture temp
		"lmbda": 200., # strength of Dirichlet prior
		"alpha": None # alpha of Dirichlet process (defaults to 1/n_topics)
	}
	RESTORE_KEY = "to_restore"

	def __init__(self, n_documents, n_vocab, d_hyperparams={},#train=True,counts=None
				 meta_graph=None, save_graph_def=True, log_dir="./log"):

		self.__dict__.update(LDA2Vec.DEFAULTS, **d_hyperparams)
		tf.reset_default_graph()
		self.sesh = tf.Session()

		if not meta_graph: # new model
			self.datetime = datetime.now().strftime(r"%y%m%d_%H%M")

			# build graph
			self.mixture = EmbedMixture(
					n_documents, self.n_document_topics, self.n_embedding,
					temperature=self.temperature)
			self.sampler = NegativeSampling(
					self.n_embedding, n_vocab, self.n_samples, power=self.power)

			handles = self._buildGraph() + (
				self.mixture(), self.mixture.proportions(softmax=True),
				self.mixture.factors, self.sampler.W)

			for handle in handles:
				tf.add_to_collection(LDA2Vec.RESTORE_KEY, handle)
			self.sesh.run(tf.initialize_all_variables())

		else: # restore saved model
			datetime_prev, _ = os.path.basename(meta_graph).split("_lda2vec")
			datetime_now = datetime.now().strftime(r"%y%m%d_%H%M")
			self.datetime = "{}_{}".format(datetime_prev, datetime_now)

			# rebuild graph
			meta_graph = os.path.abspath(meta_graph)
			tf.train.import_meta_graph(meta_graph + ".meta").restore(
				self.sesh, meta_graph)
			handles = self.sesh.graph.get_collection(LDA2Vec.RESTORE_KEY)

		# unpack tensor ops to feed or fetch
		(self.pivot_idxs, self.doc_at_pivot, self.dropout, self.target_idxs,
		 self.fraction, self.loss_word2vec, self.loss_lda, self.loss,
		 self.global_step, self.train_op, self.doc_embeds, self.doc_proportions,
		 self.topics, self.word_embeds) = handles

		self.log_dir = "{}_{}".format(log_dir, self.datetime)
		if save_graph_def: # tensorboard
			self.logger = tf.train.SummaryWriter(self.log_dir, self.sesh.graph)
			# self.logger.flush()


	@property
	def step(self):
		"""Train step"""
		return self.sesh.run(self.global_step)
		# return tf.train.global_step(self.sesh, self.global_step)


	def _buildGraph(self):

		# pivot word
		pivot_idxs = tf.placeholder(tf.int32,
									shape=[None,], # None enables variable batch size
									name="pivot_idxs")
		pivot = tf.nn.embedding_lookup(self.sampler.W, pivot_idxs) # word embedding

		# doc
		doc_at_pivot = tf.placeholder(tf.int32, shape=[None,], name="doc_ids")
		doc = self.mixture(doc_at_pivot) # doc embedding

		# context is sum of doc (mixture projected onto topics) & pivot embedding
		dropout = self.mixture.dropout
		context = tf.nn.dropout(doc, dropout) + tf.nn.dropout(pivot, dropout)

		# targets
		target_idxs = tf.placeholder(tf.int64, shape=[None,], name="target_idxs")

		# NCE loss
		with tf.name_scope("nce_loss"):
		# with tf.name_scope("word2vec_loss"):
			loss_word2vec = self.sampler(context, target_idxs)
			loss_word2vec = utils.print_(loss_word2vec, "loss_word2vec")

		# dirichlet loss (proportional to minibatch fraction)
		with tf.name_scope("lda_loss"):
			fraction = tf.Variable(1, trainable=False, dtype=tf.float32)
			loss_lda = fraction * self.prior() # dirichlet log-likelihood
			loss_lda = utils.print_(loss_lda, "loss_lda")
			# TODO penalize weights ? alpha of prior ?

		# optimize
		loss = tf.identity(loss_word2vec + self.lmbda * loss_lda, "loss")

		global_step = tf.Variable(0, trainable=False)
		train_op = tf.contrib.layers.optimize_loss(
				loss, global_step, self.learning_rate, "Adam", clip_gradients=5.)

		return (pivot_idxs, doc_at_pivot, dropout, target_idxs, fraction,
				loss_word2vec, loss_lda, loss, global_step, train_op)


	def prior(self):
		# defaults to inialization with uniform prior (1/n_topics)
		return dirichlet_likelihood(self.mixture.W, alpha=self.alpha)

	def _addSummaries(self):
		# summary nodes
		tf.scalar_summary("loss_lda", self.loss_lda)
		tf.scalar_summary("loss_nce", self.loss_word2vec)
		tf.histogram_summary("word_embeddings_hist", self.word_embeds)
		tf.histogram_summary("topic_embeddings_hist", self.topics)
		tf.histogram_summary("doc_embeddings_hist", self.doc_embeds)
		tf.scalar_summary("doc_mixture_sparsity",
						  tf.nn.zero_fraction(self.doc_proportions))
		# self.check = tf.add_check_numerics_ops()

		return tf.merge_all_summaries()


	def make_feed_dict(self, doc_ids, word_indices, window=None,
					   update_only_docs=False):

		window = (self.window if window is None else window)
		pivot_idx = word_indices[window: -window]
		doc_at_pivot = doc_ids[window: -window]

		start, end = window, word_indices.shape[0] - window

		target_idxs = []

		for frame in range(-window, window + 1):

			# Skip predicting the current pivot
			if frame == 0:
				continue

			# Predict word given context and pivot word
			# The target starts before the pivot
			target_idx = word_indices[start + frame: end + frame]
			doc_at_target = doc_ids[start + frame: end + frame]
			doc_is_same = doc_at_target == doc_at_pivot

			rand = np.random.uniform(0, 1, doc_is_same.shape[0])
			mask = (rand < self.word_dropout)
			weight = np.logical_and(doc_is_same, mask).astype(np.int32)

			# If weight is 1.0 then targetidx
			# If weight is 0.0 then -1
			target_idx = target_idx * weight + -1 * (1 - weight)

			target_idxs.append(target_idx)

		pivot_idxs = np.tile(pivot_idx, window * 2)
		docs_at_pivot = np.tile(doc_at_pivot, window * 2)
		target_idxs = np.concatenate(target_idxs)

		# ignore training points due to OOV or dropout
		# TODO set OOV token once and for all
		mask = np.logical_and((target_idxs > 0), (pivot_idxs > 0))

		feed_dict = {self.pivot_idxs: pivot_idxs[mask],
					 self.doc_at_pivot: docs_at_pivot[mask],
					 self.target_idxs: target_idxs[mask],
					 self.dropout: self.dropout_ratio}

		return feed_dict


	def train(self, doc_ids, flattened, max_epochs=np.inf, verbose=False,
			  save=False, save_every=1000, outdir="./out", summarize=True,
			  summarize_every=1000):

		if save:
			saver = tf.train.Saver(tf.all_variables())

		if summarize:
			merged = self._addSummaries()
			try:
				self.logger.flush()
			except(AttributeError): # not yet logging
				self.logger = tf.train.SummaryWriter(log_dir, self.sesh.graph)

		j = 0
		epoch = 0

		fraction = self.batch_size / len(flattened) # == batch / n_corpus
		self.sesh.run(tf.assign(self.fraction, fraction))

		now = datetime.now().isoformat()[11:]
		print("------- Training begin: {} -------\n".format(now))

		while epoch < max_epochs:
			try:

				# doc_ids, word_idxs
				for d, f in utils.chunks(self.batch_size, doc_ids, flattened):
					t0 = datetime.now().timestamp()

					feed_dict = self.make_feed_dict(d, f)

					fetches = [self.loss_lda, self.loss_word2vec,
							   self.loss, merged, self.train_op]
					loss_lda, loss_word2vec, loss, summary, _ = self.sesh.run(
						fetches, feed_dict=feed_dict)

					j += 1

					if verbose and j % 1000 == 0:
						msg = ("J:{j:05d} E:{epoch:05d} L_nce:{l_word2vec:1.3e} "
							   "L_dirichlet:{l_lda:1.3e} R:{rate:1.3e}")

						t1 = datetime.now().timestamp()
						dt = t1 - t0
						rate = self.batch_size / dt
						logs = dict(l_word2vec=loss_word2vec, epoch=epoch, j=j,
									l_lda=loss_lda, rate=rate)

						print(msg.format(**logs))

					if save and j % save_every == 0:
						outfile = os.path.join(os.path.abspath(outdir),
												"{}_lda2vec".format(self.datetime))
						saver.save(self.sesh, outfile, global_step=self.step)

					if summarize and j % summarize_every == 0:
						self.logger.add_summary(summary, global_step=self.step)

				epoch += 1

			except(KeyboardInterrupt):
				break

		print("epoch", epoch)
		print("max", max_epochs)
		now = datetime.now().isoformat()[11:]
		print("------- Training end: {} -------\n".format(now))

		if save:
			outfile = os.path.join(os.path.abspath(outdir),
								   "{}_lda2vec".format(self.datetime))
			saver.save(self.sesh, outfile, global_step=self.step)

		try:
			self.logger.flush()
			self.logger.close()
		except(AttributeError): # not logging
			pass

		sys.exit(0)


	def _buildGraph_similarity(self):
		"""Build nodes to compute the cosine similarity between examples
		(doc/word/topic idxs) and corresponding embeddings
		"""
		idxs_in = tf.placeholder(tf.int32,
							  shape=[None,], # None enables variable batch size
							  name="idxs") # doc or topic or word

		n = tf.placeholder_with_default(10, shape=None, name="n")

		normalized_embedding = dict()
		for name, embedding in zip(("word", "topic", "doc"),
								   (self.word_embeds, self.topics, self.doc_embeds)):
			norm = tf.sqrt(tf.reduce_sum(embedding**2, 1, keep_dims=True))
			normalized_embedding[name] = embedding / norm

		similarities = dict()
		for in_, vs in (("word", "word"),
						("word", "topic"),
						("topic", "word"),
						("doc", "doc")):
			embeddings_in = tf.nn.embedding_lookup(normalized_embedding[in_],
												   idxs_in)
			similarity = tf.matmul(embeddings_in, normalized_embedding[vs],
								   transpose_b=True)
			values, top_idxs = tf.nn.top_k(similarity, sorted=True, k=n)
			# top_sims = tf.gather_nd(similarity, top_idxs)
			# similarities[(in_, vs)] = [top_idxs, top_sims]
			similarities[(in_, vs)] = (top_idxs, similarity)

		return (idxs_in, n, similarities)


	def compute_similarity(self, ids, in_, vs, n=10):
		"""Compute the cosine similarity between minibatch examples
		and all embeddings.

		Args: ids (1-D array of idxs)
		      in_ = "doc" or "word" or "topic" (corresponding to ids)
		      vs = "doc" or "word" or "topic" (corresponding to embedding to compare)
		"""
		# TODO pass args for embed ids ?
		while True:
			try:
				feed_dict = {self.idxs_in: ids, self.n: n}
				fetches = self.similarities[(in_, vs)]
				top_idxs, sims = self.sesh.run(fetches, feed_dict=feed_dict)
				top_sims = sims[ # select similarity to top matching idxs per id
					tuple([i]*top_idxs.shape[1] for i in range(top_idxs.shape[0])),
					 top_idxs]
				return (top_idxs, top_sims)

			except(AttributeError): # not yet initialized
				(self.idxs_in, self.n,
				 self.similarities) = self._buildGraph_similarity()


	# def validate(self, doc_ids, flattened, save=False):

	# 	loss_word2vec = self.fit_partial(doc_ids, flattened)

	# 	loss_lda = self.sesh.run(self.prior(), feed_dict=feed_dict)

	# 	loss = loss_word2vec + self.lmbda * loss_lda

	# 	if verbose and j % 1000 == 0:
	# 			# msg = ("J:{j:05d} E:{epoch:05d} L:{loss:1.3e} " # 			# 	"P:{prior:1.3e} R:{rate:1.3e}")
	# 			msg = ("J:{j:05d} E:{epoch:05d} L_nce:{l_word2vec:1.3e} "
	# 					"L_dirichlet:{l_lda:1.3e} R:{rate:1.3e}")

	# 			t1 = datetime.now().timestamp()
	# 			dt = t1 - t0
	# 			rate = self.batch_size / dt
	# 			logs = dict(l_word2vec=loss_word2vec, epoch=epoch, j=j,
	# 					l_lda=loss_lda, rate=rate)

	# 			print(msg.format(**logs))
