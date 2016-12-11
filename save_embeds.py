import os
import sys

import numpy as np

from model import LDA2Vec


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
				   ("doc_proportions", doc_proportions),
				   ("word_embeds", word_embeds)):

		np.save(os.path.join(outdir, f), arr)

if __name__ == "__main__":
	meta_graph = sys.argv[1]
	assert os.path.exists(meta_graph + "meta"), "Where's your infile ?"
	save_embeddings(meta_graph)
