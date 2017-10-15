# -*- coding: ascii -*-

__version__ = '0.0.1'

import gzip, time
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import svds

class worvecs:
    """Word vectors modeling tool.
    Attributes:
        window (int): number of words on the either side of the word used for
            building word vectors.
        pctl (int): percentile of word counts to use for discarding less
            frequent words.
        width (int): word vectors width.
        words (np.array): words dictionary.
        vectors (np.array): word vectors.
        word_ids (dict): word to id mapping for faster lookup.
    """
    def __init__(self, sentences=None, window=10, pctl=75, width=500):
        """Class initializer.

        Args:
            sentences (iterable, optional): list of sentences spit into tokens
            window (int): number of words on the either side of the word used
                for building word vectors. Default value is 10.
            pctl (int): percentile of word counts to use for discarding rare
                words. Default value is 75.
            width (int): word vectors width. Default value 500.

        Returns:
            bool: Reurns True if sentences are provided and the model is
                succesfully built, None otherwise.
        """
        self.window = window
        self.pctl = pctl
        self.width = width
        self.words = np.array([])
        self.vectors = np.array([])
        self.word_ids = {}
        if sentences != None:
            self.buildWordVectors(sentences)
        else:
            return None

    def buildWordVectors(self, sentences):
        """Method to build word embeddings model.

        Args:
            sentences (iterable): list of sentences spit into tokens

        Returns:
            bool: The return value. True if word vectors are succesfully
                updated, False otherwise.
        """
        words = []
        word_ids = {}
        word_cnt = []
        before = []
        after = []
        for sent in sentences:
            for word in sent:
                if word not in word_ids:
                    words.append(word)
                    word_ids[word] = len(word_ids)
                    word_cnt.append(1)
                    before.append({})
                    after.append({})
                else:
                    word_cnt[word_ids[word]] += 1
            for i in range(len(sent)):
                start = 0 if i - self.window < 0 else i - self.window
                end = len(sent) if i + self.window > len(sent) else \
                    i + self.window
                w = word_ids[sent[i]]
                for j in range(start, i):
                    c = word_ids[sent[j]]
                    before[w][c] = before[w][c] + 1 if c in before[w] else 1
                for j in range(i+1,end):
                    c = word_ids[sent[j]]
                    after[w][c] = after[w][c] + 1 if c in after[w] else 1

        # build sparse matrix of the word-context jaccards
        # using only the words above the threshold frequency
        min_count = np.percentile(word_cnt, self.pctl)
        sorted_ids = np.argsort(np.array(word_cnt))[::-1]
        self.words = np.array([words[i] for i in sorted_ids if word_cnt[i] >= \
            min_count])
        if len(self.words) < 2:
            return False
        data = []
        indcs = []
        indptr = [0]
        for w in self.words:
            wid = word_ids[w]
            wf = word_cnt[wid]
            vec = np.zeros(2*len(words))
            for c in before[wid]:
                cf = word_cnt[c]
                if cf >= min_count:
                    wcf = before[wid][c]
                    vec[c] = wcf/(wf+cf-wcf)
            for c in after[wid]:
                cf = word_cnt[c]
                if cf >= min_count:
                    wcf = after[wid][c]
                    vec[c+len(words)] = wcf/(wf+cf-wcf)
            if np.linalg.norm(vec, 2) > 1e-6:
                vec /= np.linalg.norm(vec, 2)
                nonzero_indcs = np.nonzero(vec)[0]
                data += list(vec[nonzero_indcs])
                indcs += list(nonzero_indcs)
                indptr.append(indptr[-1] + len(nonzero_indcs))
            else:
                data += [0]
                indcs += [0]
                indptr.append(indptr[-1] + 1)
        m = csc_matrix(csr_matrix((data, indcs, indptr)))
        if m.shape[0] <= self.width or m.shape[1] <= self.width:
            raise Exception("The number of dimensions in the constructed " + \
                "sparse word vectors is smaller than requested vector " + \
                " length. Perhaps consider larger corpus of sentences.")
            return False

        # decompose the sparse matrix

        ut, s, vt = svds(m, self.width)
        # update dictionary and vectors
        vectors = []
        for i, vec in enumerate(ut.dot(np.diag(s))):
            if np.linalg.norm(vec, 2) > 1e-6:
                vec /= np.linalg.norm(vec, 2)
            else:
                vec *= 0
            vectors.append(vec)
        self.vectors = np.array(vectors)
        self.word_ids = {self.words[i]:i for i in range(len(self.words))}
        return True

    def save(self, fname):
        """Save word vectors to a file.

        Args:
            fname (str): file name to store the model (gzipped).

        Returns:
            bool: The return value. True if the model is saved succesfully,
                False otherwise.
        """
        if len(self.words) < 2:
            return False
        with gzip.open(fname, "wt", encoding="utf-8") as f:
            f.write("%d %d\n" % (len(self.words), self.width))
            for i, vec in enumerate(self.vectors):
                f.write("%s %s\n" % (self.words[i], " ".join("%e" % v
                    for v in vec)))
        return True

    def load(self, fname):
        """Method to load word embeddings model.

        Args:
            fname (str): file name to load the model from (gzipped).

        Returns:
            bool: The return value. True if the model is loaded succesfully,
                False otherwise.
        """
        words = []
        vectors = []
        with gzip.open(fname, "rt", encoding="utf-8") as f:
            header = f.readline()
            self.width = int(header.split(" ")[1])
            line = f.readline()
            while len(line) > 0:
                parts = line.strip().split(" ")
                if len(parts) > self.width:
                    k = " ".join(parts[:-self.width])
                    v = np.array([float(x) for x in parts[-self.width:]])
                    words.append(k)
                    vectors.append(v)
                line = f.readline()
        self.words = np.array(words)
        self.vectors = np.array(vectors)
        self.word_ids = {words[i]:i for i in range(len(words))}
        return True

    def similarWords(self, word, topN=10):
        """Method to search for most similar words in the vectors space.

        Args:
            word (str): the word.
            topN (int): the number of similar words to return.

        Returns:
            tuple of arrays of similar words and corresponding cosine
                similarities.
        """
        if word not in self.word_ids:
            return None
        dot = self.vectors.dot(self.vectors[self.word_ids[word]])
        indcs = np.argsort(dot)
        return self.words[indcs][-topN:][::-1], dot[indcs][-topN:][::-1]

    def similarRelations(self, w1, w2, w, topN=10):
        """Method to search for most similar relations in the vectors space.

        Args:
            w1, w2 (str): the words with the relationship to reproduce (w1->w2).
            w (str): the analogue of the first word to apply the relationship
                to (w->?).
            topN (int): the number of similar relations to return.

        Returns:
            tuple of arrays of siilar words and corresponding cosine
                similarities or None.
        """
        if w1 not in self.word_ids or w2 not in self.word_ids or \
            w not in self.word_ids:
            return None
        v1 = self.vectors[self.word_ids[w]] + self.vectors[self.word_ids[w2]]
        v1 = v1/np.linalg.norm(v1, 2)
        v2 = self.vectors + self.vectors[self.word_ids[w1]]
        v2 = np.array([v/(np.linalg.norm(v, 2) if np.linalg.norm(v, 2) > 0
            else 1) for v in v2])
        dot = v2.dot(v1)
        indcs = np.argsort(dot)
        return self.words[indcs][-topN:][::-1], dot[indcs][-topN:][::-1]
