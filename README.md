# worvecs  

Word-context similarity vectors, alternative to [word2vec](https://code.google.com/archive/p/word2vec/), [GloVe](https://github.com/stanfordnlp/GloVe) and [hyperwords](https://bitbucket.org/omerlevy/hyperwords). This particular version is memory greedy, but results are similar to the alternatives. The goal of this project is to find a robust analytical solution instead of iterative learning.  

Word-context similarity is encoded in a sparse matrix with Jaccard index values between the word and the context, then l2 normalized and decomposed into vectors of smaller dimensions.  

Words that are close in cosine space appear to be semantically similar, same as in other algorithms mentioned above. Also word relations in the vector space are similarly preserved.  


## Usage  

Pass an iterable corpus of sentences, where each sentence is a list of tokens, to the class constructor to build a model:  

```python
from worvecs import worvecs  
model = worvecs(sentences)  
```  

to save the model:  

```python
model.save(fname)  
```  

or load an existing model:  

```python
model.load(fname)  
```  

methods to explore relationships between words:  

```python
similar_words, similarities = model.similarWords(word)  
similar_relations, similarities = model.similarRelations(word1, word2, word3)  
```  


## Requirements

 - It is recommended to have `numpy` and `scipy` installed using a package manager, like apt or pip. Otherwise `python setup.py install` will attempt to compile them from scratch, which will require atlas/blas/lapack libraries installed.


## References
 - [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
 - [GloVe model for distributed word representation](https://github.com/stanfordnlp/GloVe)
 - [Improving Distributional Similarity with Lessons Learned from Word Embeddings](http://www.aclweb.org/anthology/Q15-1016)
 - [Network Embedding as Matrix Factorization: Unifying DeepWalk, LINE, PTE, and node2vec](https://arxiv.org/abs/1710.02971)


## License
All work contained in this package is licensed under GNU GPL License. See the included LICENSE file.
