# -*- coding: utf-8 -*-
"""
Created on Thu May  7 15:34:37 2020

@author: gsyoume
"""

from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
#%%
df = pd.read_csv('Seattle_Hotels_Duplicates2.csv', encoding="latin-1")
#%%
df.name.value_counts()
#%%
#df.loc[df['name'] == 'Roy Street Commons']
#%%
df['name_address'] = df['name'] + ' ' + df['address']
name_address = df['name_address']
#%%
vectorizer = TfidfVectorizer("char", ngram_range=(1, 4), sublinear_tf=True)
tf_idf_matrix = vectorizer.fit_transform(name_address)
#%%
def awesome_cossim_top(A, B, ntop, lower_bound=0):
  
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape
 
    idx_dtype = np.int32
 
    nnz_max = M*ntop
 
    indptr = np.zeros(M+1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)

    ct.sparse_dot_topn(
        M, N, np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr, indices, data)

    return csr_matrix((data,indices,indptr),shape=(M,N))
#%%
matches = awesome_cossim_top(tf_idf_matrix, tf_idf_matrix.transpose(), 5)
#%%
def get_matches_df(sparse_matrix, name_vector, top=5581):
    non_zeros = sparse_matrix.nonzero()
    
    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]
    
    if top:
        nr_matches = top
    else:
        nr_matches = sparsecols.size
    
    left_side = np.empty([nr_matches], dtype=object)
    right_side = np.empty([nr_matches], dtype=object)
    similairity = np.zeros(nr_matches)
    
    for index in range(0, nr_matches):
        left_side[index] = name_vector[sparserows[index]]
        right_side[index] = name_vector[sparsecols[index]]
        similairity[index] = sparse_matrix.data[index]
    
    return pd.DataFrame({'left_side': left_side,
                          'right_side': right_side,
                           'similarity': similairity})
#%%
matches_df = get_matches_df(matches, name_address)
#%%
ff = matches_df.to_csv('xxx.csv')



















