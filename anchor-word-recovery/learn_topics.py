import sys
import random_projection as rp
from numpy.random import RandomState
import numpy as np
from fastRecover import do_recovery
from anchors import findAnchors
import scipy.sparse as sparse
import time
from Q_matrix import generate_Q_matrix 
import scipy.io

class Params:

    def __init__(self, filename):
        self.log_prefix=None
        self.checkpoint_prefix=None
        self.seed = int(time.time())

        for l in file(filename):
            if l == "\n" or l[0] == "#":
                continue
            l = l.strip()
            l = l.split('=')
            if l[0] == "log_prefix":
                self.log_prefix = l[1]
            elif l[0] == "max_threads":
                self.max_threads = int(l[1])
            elif l[0] == "eps":
                self.eps = float(l[1])
            elif l[0] == "checkpoint_prefix":
                self.checkpoint_prefix = l[1]
            elif l[0] == "new_dim":
                self.new_dim = int(l[1])
            elif l[0] == "seed":
                self.seed = int(l[1])
            elif l[0] == "anchor_thresh":
                self.anchor_thresh = int(l[1])
            elif l[0] == "top_words":
                self.top_words = int(l[1])

#parse input args
if len(sys.argv) > 6:
    infile = sys.argv[1]
    settings_file = sys.argv[2]
    vocab_file = sys.argv[3]
    K = int(sys.argv[4])
    loss = sys.argv[5]
    outfile = sys.argv[6]

else:
    print "usage: ./learn_topics.py word_doc_matrix settings_file vocab_file K loss output_filename"
    print "for more info see readme.txt"
    sys.exit()

params = Params(settings_file)
params.dictionary_file = vocab_file
M = scipy.io.loadmat(infile)['M']
print "identifying candidate anchors"
candidate_anchors = []

#only accept anchors that appear in a significant number of docs
for i in xrange(M.shape[0]):
    if len(np.nonzero(M[i, :])[1]) > params.anchor_thresh:
        candidate_anchors.append(i)

print len(candidate_anchors), "candidates"

#forms Q matrix from document-word matrix
Q = generate_Q_matrix(M)

vocab = file(vocab_file).read().strip().split()

#check that Q sum is 1 or close to it
print "Q sum is", Q.sum()
V = Q.shape[0]
print "done reading documents"

#find anchors- this step uses a random projection
#into low dimensional space
anchors = findAnchors(Q, K, params, candidate_anchors)
print "anchors are:"
for i, a in enumerate(anchors):
    print i, vocab[a]

#recover topics
A, topic_likelihoods = do_recovery(Q, anchors, loss, params) 
print "done recovering"

np.savetxt(outfile+".A", A)
np.savetxt(outfile+".topic_likelihoods", topic_likelihoods)

#display
f = file(outfile+".topwords", 'w')
for k in xrange(K):
    topwords = np.argsort(A[:, k])[-params.top_words:][::-1]
    print vocab[anchors[k]], ':',
    print >>f, vocab[anchors[k]], ':',
    for w in topwords:
        print vocab[w],
        print >>f, vocab[w],
    print ""
    print >>f, ""
