import numpy as np
import scipy.sparse
import scipy.io
import sys

if len(sys.argv) < 3:
    print "usage: input_matrix vocab_file cutoff"
    sys.exit()

input_matrix = sys.argv[1]
full_vocab = sys.argv[2]

output_matrix = sys.argv[1]+".trunc"
output_vocab = sys.argv[2]+".trunc"

# cutoff for the number of distinct documents that each word appears in
cutoff = int(sys.argv[3])



# Read in the vocabulary and build a symbol table mapping words to indices
table = dict()
numwords = 0
with open(full_vocab, 'r') as file:
    for line in file:
        table[line.rstrip()] = numwords
        numwords += 1


remove_word = [False]*numwords

# Read in the stopwords
with open('stopwords.txt', 'r') as file:
    for line in file:
        if line.rstrip() in table:
            remove_word[table[line.rstrip()]] = True


# Load previously generated document
S = scipy.io.loadmat(input_matrix)
M = S['M']

if M.shape[0] != numwords:
    print 'Error: vocabulary file has different number of words', M.shape, numwords
    sys.exit()
print 'Number of words is ', numwords
print 'Number of documents is ', M.shape[1]


M = M.tocsr()

new_indptr = np.zeros(M.indptr.shape[0], dtype=np.int32)
new_indices = np.zeros(M.indices.shape[0], dtype=np.int32)
new_data = np.zeros(M.data.shape[0], dtype=np.float64)

indptr_counter = 1
data_counter = 0

for i in xrange(M.indptr.size - 1):

    # if this is not a stopword
    if not remove_word[i]:

        # start and end indices for row i
        start = M.indptr[i]
        end = M.indptr[i + 1]
        
        # if number of distinct documents that this word appears in is >= cutoff
        if (end - start) >= cutoff:
            new_indptr[indptr_counter] = new_indptr[indptr_counter-1] + end - start
            new_data[new_indptr[indptr_counter-1]:new_indptr[indptr_counter]] = M.data[start:end]
            new_indices[new_indptr[indptr_counter-1]:new_indptr[indptr_counter]] = M.indices[start:end]
            indptr_counter += 1
        else:
            remove_word[i] = True

new_indptr = new_indptr[0:indptr_counter]
new_indices = new_indices[0:new_indptr[indptr_counter-1]]
new_data = new_data[0:new_indptr[indptr_counter-1]]

M = scipy.sparse.csr_matrix((new_data, new_indices, new_indptr))
M = M.tocsc()
scipy.io.savemat(output_matrix, {'M' : M}, oned_as='column')

print 'New number of words is ', M.shape[0]
print 'New number of documents is ', M.shape[1]

# Output the new vocabulary
output = open(output_vocab, 'w')
row = 0
with open(full_vocab, 'r') as file:
    for line in file:
        if not remove_word[row]:
            output.write(line)
        row += 1
output.close()
