#!/bin/bash
echo "start demo"
for corpus in nips
do
    echo "downloading UCI $corpus corpus"
    wget http://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/vocab.$corpus.txt
    wget http://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.$corpus.txt.gz
    gunzip docword.$corpus.txt.gz
    echo "preprocessing, translate from docword.txt to scipy format"
    python uci_to_scipy.py docword.$corpus.txt M_$corpus.full_docs.mat
    echo "preprocessing: removing rare words and stopwords"
    python truncate_vocabulary.py M_$corpus.full_docs.mat vocab.$corpus.txt 50
    for loss in L2
    do
        for K in 20 50 100
        do
            echo "learning with nonnegative recover method using $loss loss..."
            python learn_topics.py M_$corpus.full_docs.mat.trunc.mat settings.example vocab.$corpus.txt.trunc $K $loss demo_$loss\_out.$corpus.$K
        done
    done
done
