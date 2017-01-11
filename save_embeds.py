import os
import sys

import numpy as np


def embeddings_from_meta_graph(meta_graph, outdir=None, tf_projector=False):
	"""Load trained model and save embeddings"""
	from model import LDA2Vec

	outdir = ("results_{}".format(os.path.basename(meta_graph))
			  if outdir is None else outdir)

	m = LDA2Vec(47, 47, meta_graph=meta_graph)

	topics = m.sesh.run(m.topics)
	doc_embeds = m.sesh.run(m.doc_embeds)
	doc_proportions = m.sesh.run(m.doc_proportions)
	word_embeds = m.sesh.run(m.word_embeds)

	for f, arr in (("topics", topics),
				   ("doc_embeds", doc_embeds),
				   ("doc_proportions", doc_proportions),
				   ("word_embeds", word_embeds)):

		np.save(os.path.join(outdir, f), arr)


def tf_projector_from_embeddings(f_embed, metadata="metadata.tsv", outdir=None):
	"""Load np array of trained embeddings and save for visualization with TensorBoard Projector"""
	import tensorflow as tf
	from tensorflow.contrib.tensorboard.plugins import projector

	outdir = (os.path.join(os.path.dirname(os.path.abspath(f_embed)), "out")
			  if outdir is None else outdir)

	# tf graph for embeddings
	embeds = tf.Variable([47], dtype=tf.float32, name="embeddings")
	embed_vals = tf.placeholder(tf.float32, [None, None])
	assign_embeds = tf.assign(embeds, embed_vals, validate_shape=False)

	sesh = tf.Session()
	sesh.run(tf.global_variables_initializer())
	logger = tf.summary.FileWriter(outdir, sesh.graph)

	# tf projector
	config = projector.ProjectorConfig()
	embedding = config.embeddings.add()
	embedding.tensor_name = embeds.name
	embedding.metadata_path = os.path.join(outdir, metadata)
	projector.visualize_embeddings(logger, config)

	# assign embeddings and save
	feed_dict = {embed_vals: np.load(f_embed)}
	sesh.run(assign_embeds, feed_dict=feed_dict)

	saver = tf.train.Saver(tf.global_variables())
	outfile, _ = os.path.splitext(os.path.basename(f_embed))
	saver.save(sesh, os.path.join(outdir, outfile))

	logger.flush()
	logger.close()


if __name__ == "__main__":
	infile = sys.argv[1]

	# assert os.path.exists(infile + "meta"), "Where's your infile ?"
	# save_embeddings(meta_graph=infile)

	assert os.path.exists(infile), "Where's your infile ?"
	tf_projector_from_embeddings(f_embed=infile)
