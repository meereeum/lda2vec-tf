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
		"word_dropout": 0., # dropout (?)

		"power": 0.75, # negative sampling power - TODO ?
		"n_samples": 15, # num negative samples

		"temperature": 1.0, # embed mixture temp
		"lmbda": 200. # strength of Dirichlet prior
	}
	# RESTORE_KEY = "to_restore" TODO?

	def __init__(self, n_corpus, n_documents, n_vocab, #train=True,counts=None
				 d_hyperparams={}, save_graph_def=True, log_dir="./log"):

		self.__dict__.update(LDA2Vec.DEFAULTS, **d_hyperparams)

		# build graph
		self.mixture = EmbedMixture(n_documents, n_document_topics, n_units,
									keep_prob=dropout, temperature=temperature)
		self.sampler = NegativeSampling(n_units, n_vocab, n_samples)

		(self.pivot_idxs, self.doc_at_pivot, self.dropout, self.target_idxs,
		 losses, self.global_step, train_ops) = self._buildGraph()

		self.loss_word2vec, self.loss_lda = losses
		self.train_op_word2vec, self.train_op_lda = train_ops

		# initialize
		self.sesh = tf.Session()
		self.sesh.run(tf.initialize_all_variables())

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
		with tf.name_scope("lda_loss"):
			fraction = self.batch_size / self.n_corpus
			loss_lda = lmbda * fraction * self.prior() # dirichlet log-likelihodd
			# TODO penalize weights ?

		losses = (loss_word2vec, loss_lda)

		# optimize
		global_step = tf.Variable(0, trainable=False)

		train_ops = tuple(tf.contrib.layers.optimize_loss(
			loss, global_step, self.learning_rate, "Adam", clip_gradient=5.)
						  for loss in losses)
		# train_op_word2vec = tf.contrib.layers.optimize_loss(
		# 	loss_word2vec, global_step, learning_rate, "Adam", clip_gradient=5.,
		# 	name="train_op_word2vec")
		# train_op_lda = tf.contrib.layers.optimize_loss(
		# 	loss_lda, global_step, learning_rate, "Adam", clip_gradient=5.,
		# 	name="train_op_lda")
		# TODO combine ?

		return (pivot_idxs, doc_at_pivot, dropout, target_idxs, losses,
				global_step, train_ops)


	def prior(self, alpha=None):
		# defaults to uniform pror (1/n_topics)
		return dirichlet_likelihood(self.mixture.weights, alpha=alpha)


	# def fit_partial(self, rdoc_ids, rword_indices, window=5,
	def fit_partial(self, doc_ids, word_indices, window=5,
					update_only_docs=False):
		# TODO: placeholders, etc
		# doc_ids, word_indices = move(self.xp, rdoc_ids, rword_indices)
		# pivot_idx = next(move(self.xp, rword_indices[window: -window]))
		# TODO feed pivot_idx to placeholder
		# pivot = F.embed_id(pivot_idx, self.sampler.W)

		pivot_idx = word_indices[window: -window]

		# if update_only_docs:
			# TODO ?
			# pivot = tf.Variable(pivot, trainable=False)
			# pivot.unchain_backward()

		doc_at_pivot = doc_ids[window: -window]

		# TODO placeholder or no ?
		# doc = self.mixture(next(move(self.xp, doc_at_pivot)),
		# 					update_only_docs=update_only_docs)

		loss = 0.0
		start, end = window, word_indices.shape[0] - window
		# context = (tf.nn.dropout(doc, self.dropout) +
		# 			tf.nn.dropout(pivot, self.dropout))

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

			# TODO - feed/fetch etc
			# target, = move(self.xp, targetidx)

			# loss = self.sampler(context, target)
			# loss.backward()

			# if update_only_docs:
			# 	# Wipe out any gradient accumulation on word vectors
			# 	self.sampler.W.grad *= 0.0

		return loss.data

	def train(doc_ids, flattened):
		j = 0
		epoch = 0
		# progress = shelve.open('progress.shelve') TODO ?

		for epoch in range(100):
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

			# doc_ids
			for d, f in utils.chunks(self.batch_size, doc_ids, flattened):
				t0 = time.time()
				optimizer.zero_grads()
				l = self.fit_partial(d.copy(), f.copy())
				prior = model.prior()
				loss = prior * fraction
				loss.backward()
				optimizer.update()

				msg = ("J:{j:05d} E:{epoch:05d} L:{loss:1.3e} "
					"P:{prior:1.3e} R:{rate:1.3e}")

				t1 = time.time()
				dt = t1 - t0
				rate = batchsize / dt
				logs = dict(loss=float(l), epoch=epoch, j=j,
							prior=float(prior.data), rate=rate)
				j += 1

	def preprocess(self, doc_ids, word_indices, window=5,
					update_only_docs=False):

		pivot_idx = word_indices[window: -window]
		doc_at_pivot = doc_ids[window: -window]

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
			# randomly drop out words (default is to never do this)
			rand = np.random.uniform(0, 1, doc_is_same.shape[0])
			mask = (rand > self.word_dropout).astype("bool")
			weight = np.logical_and(doc_is_same, mask).astype("int32")

			# If weight is 1.0 then target_idx
			# If weight is 0.0 then -1
			target_idx = target_idx * weight + -1 * (1 - weight)

		return loss.data


    def train(self, X, max_iter=np.inf, max_epochs=np.inf, cross_validate=True,
              verbose=True, save=False, outdir="./out", plots_outdir="./png",
              plot_latent_over_time=False):
        if save:
            saver = tf.train.Saver(tf.all_variables())

        try:
            err_train = 0
            now = datetime.now().isoformat()[11:]
            print("------- Training begin: {} -------\n".format(now))

            if plot_latent_over_time: # plot latent space over log_BASE time (i.e. training rounds)
                BASE = 2
                INCREMENT = 0.5
                pow_ = 0 # initial power/exponent

            while True:
                x, _ = X.train.next_batch(self.batch_size)
                feed_dict = {self.x_in: x, self.dropout_: self.dropout}
                fetches = [self.x_reconstructed, self.cost, self.global_step, self.train_op]
                x_reconstructed, cost, i, _ = self.sesh.run(fetches, feed_dict)

                err_train += cost

                if plot_latent_over_time:
                    while int(round(BASE**pow_)) == i:
                        plot.exploreLatent(self, nx=30, ny=30, ppf=True, outdir=
                                        plots_outdir, name="explore_ppf30_{}".format(pow_))

                        names = ("train", "validation", "test")
                        datasets = (X.train, X.validation, X.test)
                        for name, dataset in zip(names, datasets):
                            plot.plotInLatent(self, dataset.images, dataset.labels, range_=
                                              (-6, 6), title=name, outdir=plots_outdir,
                                              name="{}_{}".format(name, pow_))

                        print("{}^{} = {}".format(BASE, pow_, i))
                        pow_ += INCREMENT

                if i%1000 == 0 and verbose:
                    print("round {} --> avg cost: ".format(i), err_train / i)

                if i%2000 == 0 and verbose:# and i >= 10000:
                    # visualize `n` examples of current minibatch inputs + reconstructions
                    plot.plotSubset(self, x, x_reconstructed, n=10, name="train",
                                    outdir=plots_outdir)

                    if cross_validate:
                        x, _ = X.validation.next_batch(self.batch_size)
                        feed_dict = {self.x_in: x}
                        fetches = [self.x_reconstructed, self.cost]
                        x_reconstructed, cost = self.sesh.run(fetches, feed_dict)

                        print("round {} --> CV cost: ".format(i), cost)
                        plot.plotSubset(self, x, x_reconstructed, n=10, name="cv",
                                        outdir=plots_outdir)

                if i >= max_iter or X.train.epochs_completed >= max_epochs:
                    print("final avg cost (@ step {} = epoch {}): {}".format(
                        i, X.train.epochs_completed, err_train / i))
                    now = datetime.now().isoformat()[11:]
                    print("------- Training end: {} -------\n".format(now))

                    if save:
                        outfile = os.path.join(os.path.abspath(outdir), "{}_vae_{}".format(
                            self.datetime, "_".join(map(str, self.architecture))))
                        saver.save(self.sesh, outfile, global_step=self.step)
                    try:
                        self.logger.flush()
                        self.logger.close()
                    except(AttributeError): # not logging
                        continue
                    break

        except(KeyboardInterrupt):
            print("final avg cost (@ step {} = epoch {}): {}".format(
                i, X.train.epochs_completed, err_train / i))
            now = datetime.now().isoformat()[11:]
            print("------- Training end: {} -------\n".format(now))
            sys.exit(0)
