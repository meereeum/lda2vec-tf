from datetime import datetime
import time
import os

import numpy as np
import tensorflow as tf

from lda2vec import dirichlet_likelihood
from lda2vec import EmbedMixture
from lda2vec import NegativeSampling
# from lda2vec.utils import move



class LDA2Vec():

	DEFAULTS = {
		"n_document_topics": 10,
		"n_units": 256, # embedding size

		"batch_size": 128,
		"learning_rate": 0.1,
		"dropout_ratio": 0.5, # keep_prob
		"word_dropout": 0., # dropout (?)

		"power": 0.75, # negative sampling power - TODO ?
		"n_samples": 15, # num negative samples

		"temperature": 1.0, # embed mixture temp
		"lmbda": 200. # strength of Dirichlet prior
	}
	RESTORE_KEY = "to_restore"

	def __init__(self, n_corpus, n_documents, n_vocab, #train=True,counts=None
				 d_hyperparams={}, save_graph_def=True, log_dir="./log"):

		self.__dict__.update(LDA2Vec.DEFAULTS, **d_hyperparams)
		self.sesh = tf.Session()

		self.mixture = EmbedMixture(n_documents, n_document_topics, n_units,
									keep_prob=dropout, temperature=temperature)
		self.sampler = NegativeSampling(n_units, n_vocab, n_samples)

		if not meta_graph: # new model
			self.datetime = datetime.now().strftime(r"%y%m%d_%H%M")

			# build graph
			handles = self._buildGraph()
			for handle in handles: # TODO if use, need to account for loss tuple
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
		 self.n_corpus, losses, self.reset_accum_loss, self.global_step,
		 train_ops) = handles

		self.loss_word2vec, self.loss_lda = losses # TODO replace if decide to keep restore option
		# self.train_op_word2vec, self.train_op_lda = train_ops
		self.train_op = train_ops # TODO replace if decidee to keep combined op

		if save_graph_def: # tensorboard
			self.logger = tf.train.SummaryWriter(log_dir, self.sesh.graph)


	@property
	def step(self):
		"""Train step"""
		return self.global_step.eval(session=self.sesh)


	def _buildGraph(self):

		# pivot word
		pivot_idxs = tf.placeholder([None, 1], # None enables variable batch size
									dtype=tf.int32, name="pivot_idxs")
		pivot = tf.nn.embedding_lookup(self.sampler.W, # word embeddings
										pivot_idxs)

		# doc
		doc_at_pivot = tf.placeholder([None, 1], dtype=tf.int32, name="doc_ids")
		doc = self.mixture(doc_at_pivot)#, update_only_docs=update_only_docs)

		# context is sum of doc (mixture projected onto topics) & pivot embedding
		dropout = tf.placeholder_with_default(1., shape=[], name="dropout")
		context = tf.nn.dropout(doc, dropout) + tf.nn.dropout(pivot, dropout)

		# targets
		target_idxs = tf.placeholder([None, 1], dtype=tf.int64,
									 name="target_idxs")

		# NCE loss
		with tf.name_scope("nce_loss"):
			loss_word2vec = self.sampler(context, target_idxs)

			accum_loss_word2vec = tf.Variable(0, trainable=False)
			accum_loss_word2vec += loss_word2vec

			reset_accum_loss = tf.assign(accum_loss_word2vec,
										 tf.Variable(0, trainable=False))

		# dirichlet loss (proportional to minibatch fraction)
		prior = self.p
		with tf.name_scope("lda_loss"):
			n_corpus = tf.placeholder(tf.int32, [], name="n_corpus")
			fraction = self.batch_size / self.n_corpus
			fraction = tf.assign(fraction, self.sesh.run(fraction))
			loss_lda = lmbda * fraction * self.prior() # dirichlet log-likelihodd
			# TODO penalize weights ?

		# optimize
		global_step = tf.Variable(0, trainable=False)

		# train_ops = tuple(tf.contrib.layers.optimize_loss(
		# 	loss, global_step, self.learning_rate, "Adam", clip_gradient=5.)
		# 				  for loss in losses)

		losses = (accum_loss_word2vec, loss_lda)
		loss_total = tf.add(*losses)
		train_op = tf.contrib.layers.optimize_loss(
			loss_total,
			# accum_loss_word2vec + loss_lda,
			global_step, self.learning_rate, "Adam", clip_gradient=5.)

		# train_op_word2vec = tf.contrib.layers.optimize_loss(
		# 	loss_word2vec, global_step, learning_rate, "Adam", clip_gradient=5.,
		# 	name="train_op_word2vec")
		# train_op_lda = tf.contrib.layers.optimize_loss(
		# 	loss_lda, global_step, learning_rate, "Adam", clip_gradient=5.,
		# 	name="train_op_lda")

		return (pivot_idxs, doc_at_pivot, dropout, target_idxs, n_corpus,
				losses, reset_accum_loss, global_step, train_op)#s)


	def prior(self, alpha=None):
		# defaults to inialization with uniform prior (1/n_topics)
		return dirichlet_likelihood(self.mixture.weights, alpha=alpha)


	def fit_partial(self, doc_ids, word_indices, window=5,
					update_only_docs=False):

		pivot_idx = word_indices[window: -window]

		# if update_only_docs:
			# TODO ?
			# pivot = tf.Variable(pivot, trainable=False)
			# pivot.unchain_backward()

		doc_at_pivot = doc_ids[window: -window]

		loss = 0.
		start, end = window, word_indices.shape[0] - window

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
			mask = (rand > self.word_dropout).astype("bool")
			weight = np.logical_and(doc_is_same, mask).astype("int32")

			# If weight is 1.0 then targetidx
			# If weight is 0.0 then -1
			target_idx = target_idx * weight + -1 * (1 - weight)

			feed_dict = {self.pivot_idxs: pivot_idx,
						 self.doc_at_pivot: doc_at_pivot,
						 self.dropout: self.dropout_ratio,
						 self.target_idxs: target_idx}
			fetches = [self.loss_lda]
			accum_loss = self.sesh.run(fetches, feed_dict=feed_dict)

			# if update_only_docs:
			# 	# Wipe out any gradient accumulation on word vectors
			# 	self.sampler.W.grad *= 0.0

		return accum_loss


	def train(doc_ids, flattened, max_epochs=100, verbose=True,
			  save=False, outdir="./out"):

		if save:
			saver = tf.train.Saver(tf.all_variables())

		j = 0
		epoch = 0
		# progress = shelve.open('progress.shelve') TODO ?

		feed_dict = {self.n_corpus: len(flattened)}
		self.sesh.run(self.loss_lda, feed_dict=feed_dict) # assign n_corpus

		for epoch in range(max_epochs):
			# data = prepare_topics( # TODO
			# 	cuda.to_cpu(model.mixture.weights.W.data).copy(),
			# 	cuda.to_cpu(model.mixture.factors.W.data).copy(),
			# 	cuda.to_cpu(model.sampler.W.data).copy(),
			# 	words)
			# top_words = print_top_words_per_topic(data,do_print=False)

			# if j % 100 == 0 and j > 100:
				# coherence = topic_coherence(top_words)
				# for j in range(n_topics):
				# 	print j, coherence[(j, 'cv')]
				# kw = dict(top_words=top_words, coherence=coherence, epoch=epoch)
				# progress[str(epoch)] = pickle.dumps(kw)

			# data['doc_lengths'] = doc_lengths
			# data['term_frequency'] = term_frequency
			# np.savez('topics.pyldavis', **data)

			# doc_ids, word_idxs
			for d, f in utils.chunks(self.batch_size, doc_ids, flattened):
				t0 = time.time()

				loss_word2vec = self.fit_partial(d, f)
				loss_lda, _ = self.sesh.run([self.loss_lda, self.train_op])

				self.sesh.run(self.reset_accum_loss)

				# msg = ("J:{j:05d} E:{epoch:05d} L:{loss:1.3e} "
				# 	"P:{prior:1.3e} R:{rate:1.3e}")
				msg = ("J:{j:05d} E:{epoch:05d} L_nce:{l_word2vec:1.3e} "
					   "L_dirichlet:{l_lda:1.3e} R:{rate:1.3e}")

				t1 = time.time()
				dt = t1 - t0
				rate = self.batch_size / dt
				# logs = dict(loss=float(l), epoch=epoch, j=j,
				# 			prior=float(prior.data), rate=rate)
				logs = dict(l_word2vec=float(loss_word2vec), epoch=epoch, j=j,
							l_lda=float(loss_lda), rate=rate)
				j += 1

			if verbose:
				print msg.format(**logs)

		if save:
			outfile = os.path.join(os.path.abspath(outdir),
								   "{}_lda2vec_{}".format(self.datetime))
			saver.save(self.sesh, outfile, global_step=self.step)

		try:
			self.logger.flush()
			self.logger.close()
		except(AttributeError): # not logging
			pass


	# def train(self, X, max_iter=np.inf, max_epochs=np.inf, cross_validate=True,
	#		   verbose=True, save=False, outdir="./out", plots_outdir="./png",
	#		   plot_latent_over_time=False):
	#	 if save:
	#		 saver = tf.train.Saver(tf.all_variables())

	#	 try:
	#		 err_train = 0
	#		 now = datetime.now().isoformat()[11:]
	#		 print("------- Training begin: {} -------\n".format(now))

	#		 while True:
	#			 x, _ = X.train.next_batch(self.batch_size)
	#			 feed_dict = {self.x_in: x, self.dropout_: self.dropout}
	#			 fetches = [self.x_reconstructed, self.cost, self.global_step, self.train_op]
	#			 x_reconstructed, cost, i, _ = self.sesh.run(fetches, feed_dict)

	#			 err_train += cost


	#			 if i%1000 == 0 and verbose:
	#				 print("round {} --> avg cost: ".format(i), err_train / i)

	#			 if i%2000 == 0 and verbose:# and i >= 10000:
	#				 # visualize `n` examples of current minibatch inputs + reconstructions
	#				 plot.plotSubset(self, x, x_reconstructed, n=10, name="train",
	#								 outdir=plots_outdir)

	#				 if cross_validate:
	#					 x, _ = X.validation.next_batch(self.batch_size)
	#					 feed_dict = {self.x_in: x}
	#					 fetches = [self.x_reconstructed, self.cost]
	#					 x_reconstructed, cost = self.sesh.run(fetches, feed_dict)

	#					 print("round {} --> CV cost: ".format(i), cost)
	#					 plot.plotSubset(self, x, x_reconstructed, n=10, name="cv",
	#									 outdir=plots_outdir)

	#			 if i >= max_iter or X.train.epochs_completed >= max_epochs:
	#				 print("final avg cost (@ step {} = epoch {}): {}".format(
	#					 i, X.train.epochs_completed, err_train / i))
	#				 now = datetime.now().isoformat()[11:]
	#				 print("------- Training end: {} -------\n".format(now))

	#				 if save:
	#					 outfile = os.path.join(os.path.abspath(outdir), "{}_vae_{}".format(
	#						 self.datetime, "_".join(map(str, self.architecture))))
	#					 saver.save(self.sesh, outfile, global_step=self.step)
	#				 try:
	#					 self.logger.flush()
	#					 self.logger.close()
	#				 except(AttributeError): # not logging
	#					 continue
	#				 break

	#	 except(KeyboardInterrupt):
	#		 print("final avg cost (@ step {} = epoch {}): {}".format(
	#			 i, X.train.epochs_completed, err_train / i))
	#		 now = datetime.now().isoformat()[11:]
	#		 print("------- Training end: {} -------\n".format(now))
	#		 sys.exit(0)
