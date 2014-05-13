#takes in matrix in UCI repository format and outputs a scipy sparse matrix file

import sys
import scipy.io
import scipy
import numpy as np

if len(sys.argv) < 2:
    print "usage: input_matrix output_matrix"
    sys.exit()

input_matrix = sys.argv[1]
output_matrix_name = sys.argv[2]

infile = file(input_matrix)
num_docs = int(infile.readline())
num_words = int(infile.readline())
nnz = int(infile.readline())

output_matrix = scipy.sparse.lil_matrix((num_words, num_docs))

for l in infile:
    d, w, v = [int(x) for x in l.split()]
    output_matrix[w-1, d-1] = v

scipy.io.savemat(output_matrix_name, {'M' : output_matrix}, oned_as='column')

