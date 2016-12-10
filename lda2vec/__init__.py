import sys
import os.path

assert sys.version_info.major == 3, "Must use Python 3!"

sys.path.append(os.path.dirname(__file__))

import dirichlet_likelihood
import embed_mixture
# import tracking
# import preprocess
import corpus
# import topics
import negative_sampling

dirichlet_likelihood = dirichlet_likelihood.dirichlet_likelihood
EmbedMixture = embed_mixture.EmbedMixture
# Tracking = tracking.Tracking
# tokenize = preprocess.tokenize
Corpus = corpus.Corpus
# prepare_topics = topics.prepare_topics
# print_top_words_per_topic = topics.print_top_words_per_topic
# negative_sampling = negative_sampling.negative_sampling
NegativeSampling = negative_sampling.NegativeSampling
# topic_coherence = topics.topic_coherence
