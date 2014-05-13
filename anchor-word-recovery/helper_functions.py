from __future__ import division
import itertools
import numpy as np

def warn(condition, string):
    if condition == False:
        print "WARNING: "+string

# Normalizes the rows of a matrix M to sum up to 1
def normalize_rows(M):
    row_sums = M.sum(1)
    return M/row_sums[:, np.newaxis]
	
# Normalizes the columns of a matrix M to sum up to 1
def normalize_columns(M):
    col_sums = M.sum(0)
    return M/col_sums[np.newaxis, :]

# Calculates the maximum difference between entries of two matrices A and B
def max_diff(A, B):
    C = A-B
    return max([abs(x) for x in list(C.flatten())])

# Calculates the L1 difference between two matrices A and B
def L1_diff(A,B):
    C = A-B
    return sum([abs(x) for x in list(C.flatten())])


# Calculates lower bound on L1 error between A and B
def min_error(A, B):
    K = A[0,:].size
    if K != B[0, :].size:
        print "Matrices have different numbers of columns"
    total_err = 0
    for colA in range(K):
        min_err = float("inf")
        for colB in range(K):
            err = (abs(A[:, colA] - B[:, colB])).sum()
            if err < min_err:
                min_err = err
        total_err = total_err + min_err
    return total_err
    
# Calculates a greedy L1 error between A and B
def greedy_error(A, B):
    K = A[0,:].size
    if K != B[0, :].size:
        print "Matrices have different numbers of columns"
    total_err = 0
    columns_B = range(K)
    for colA in range(K):
        min_err = float("inf")
        col_index = -1
        for colB in columns_B:
            err = (abs(A[:, colA] - B[:, colB])).sum()
            if err < min_err:
                min_err = err
                col_index = colB
        total_err = total_err + min_err
        columns_B.remove(col_index)
    return total_err

# Saves the L1 error between all pairs of columns of A and B
def save_colerrors(A, B, filename):
    K = A[0, :].size
    if K != B[0, :].size:
        print "Matrices have different numbers of columns"
    errors = np.zeros(K*K)
    for colA in range(K):
        for colB in range(K):
            errors[colA*K + colB] = (abs(A[:, colA] - B[:, colB])).sum()
    np.savetxt(filename, errors)


# Appends greedy and min L1 errors between the true and recovered topic matrices to the given text file
def save_L1_errors1(A_proj_estimate, true_A, output_filename, numdocs, seed_W, numwords, time):
    
    file = open(output_filename, 'a')
    
    file.write(str(numdocs))
    file.write('\t')
    file.write(str(seed_W))
    file.write('\t')
    file.write(str(min(greedy_error(true_A, A_proj_estimate), greedy_error(A_proj_estimate, true_A))))
    file.write('\t')
    file.write(str(min_error(true_A, A_proj_estimate)))
    file.write('\t')
    file.write(str(np.amin(A_proj_estimate)))
    file.write('\t')
    file.write(str(np.sum(np.absolute(A_proj_estimate))))
    file.write('\t')
    file.write(str(numwords - len(true_A[:, 0])))
    file.write('\t')
    file.write(str(time))
    file.write('\n')
    
    file.close()


# Appends greedy and min L1 errors between the true and recovered topic matrices to the given text file
def save_L1_errors(A_proj_estimate, true_A, output_filename, epsilon, numdocs, seed_W, numwords, time):
    
    file = open(output_filename, 'a')
    
    file.write(str(epsilon))
    file.write('\t')
    file.write(str(numdocs))
    file.write('\t')
    file.write(str(seed_W))
    file.write('\t')
    file.write(str(min(greedy_error(true_A, A_proj_estimate), greedy_error(A_proj_estimate, true_A))))
    file.write('\t')
    file.write(str(min_error(true_A, A_proj_estimate)))
    file.write('\t')
    file.write(str(np.amin(A_proj_estimate)))
    file.write('\t')
    file.write(str(np.sum(np.absolute(A_proj_estimate))))
    file.write('\t')
    file.write(str(numwords - len(true_A[:, 0])))
    file.write('\t')
    file.write(str(time))
    file.write('\n')
    
    file.close()


# Appends greedy and min L1 errors between the true and recovered topic matrices to the given text file
def save_L1_errors2(A_proj_estimate, true_A, output_filename, epsilon, step, numdocs, seed_W, numwords, time):
    
    file = open(output_filename, 'a')
    
    file.write(str(epsilon))
    file.write('\t')
    file.write(str(step))
    file.write('\t')
    file.write(str(numdocs))
    file.write('\t')
    file.write(str(seed_W))
    file.write('\t')
    file.write(str(min(greedy_error(true_A, A_proj_estimate), greedy_error(A_proj_estimate, true_A))))
    file.write('\t')
    file.write(str(min_error(true_A, A_proj_estimate)))
    file.write('\t')
    file.write(str(np.amin(A_proj_estimate)))
    file.write('\t')
    file.write(str(np.sum(np.absolute(A_proj_estimate))))
    file.write('\t')
    file.write(str(numwords - len(true_A[:, 0])))
    file.write('\t')
    file.write(str(time))
    file.write('\n')
    
    file.close()