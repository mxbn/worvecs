# -*- coding: ascii -*-

__version__ = '0.1.1'

import gzip, time, logging
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from multiprocessing import Pool, cpu_count

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
    def __init__(self, sentences=None, bins=2, bin_width=3, pctl=50, width=100,
        encoding=0, n_threads=0, verbose=0):
        """Class initializer.

        Args:
            sentences (iterable, optional): list of sentences spit into tokens
            bins (int): number of bins on the either side of the word.
                Default value is 3.
            bin_width (int): width of the bin. Default value is 3.
            pctl (int): percentile of word counts to use for discarding rare
                words. Default value is 75.
            width (int): word vectors width. Default value 500.
            encoding (int): word vectors encoding. Default value is 0 for
                Bayesian, 1 for Jaccard.

        Returns:
            bool: Reurns True if sentences are provided and the model is
                succesfully built, None otherwise.
        """
        self.bins = bins
        self.bin_width = bin_width
        self.pctl = pctl
        self.width = width
        self.words = np.array([])
        self.vectors = np.array([])
        self.word_ids = {}
        self.encoding = encoding
        self._encoding = self._bayesian
        if encoding == 1:
            self._encoding = self._jaccard
        if n_threads == 0:
            self.n_threads = cpu_count()
        else:
            self.n_threads = 1
        if verbose == 0:
            self.verbose = False
        else:
            self.verbose = True
        self.context_width = bin_width*bins
        self.bin_ids = {}
        for i in range(-bins, bins):
            for j in range(i* bin_width, (i+1)*bin_width):
                self.bin_ids[j + (0 if i < 0 else 1)] = i + bins
        if sentences != None:
            self.buildWordVectors(sentences)
        else:
            return None

    def _jaccard(self, wcf, wf, cf):
        return wcf/(wf+cf-wcf)

    def _bayesian(self, wcf, wf, cf):
        return wcf/cf

    def _buildDict(self, sentences):
        """Method to build the dictionary with word counts.

        Args:
            sentences (iterable): list of sentences spit into tokens
        """
        words = []
        word_ids = {}
        word_cnts = []
        for s in sentences:
            for w in s:
                if w not in word_ids:
                    words.append(w)
                    word_ids[w] = len(word_ids)
                    word_cnts.append(1)
                else:
                    word_cnts[word_ids[w]] += 1
        min_count = np.percentile(word_cnts, self.pctl)
        sorted_ids = np.argsort(np.array(word_cnts))[::-1]
        self.words = [words[i] for i in sorted_ids if word_cnts[i] >= min_count]
        self.word_ids = {w: i for i, w in enumerate(self.words)}
        self.word_cnts = [word_cnts[i] for i in sorted_ids \
            if word_cnts[i] >= min_count]

    def _getContext(words):
        rows = []
        cols = []
        vals = []
        for i in range(len(words)):
            if words[i] not in self.word_ids:
                continue
            window_start = max([i-width, 0])
            window_end = min([i+width, len(words)])
            for j in range(window_start, window_end):
                if i == j or words[j] not in word_ids:
                    continue
                context_id = bin_ids[j-i]*len(self.word_ids) + \
                    self.word_ids[words[j]]
                rows.append(self.word_ids[words[i]])
                cols.append(context_id)
                vals.append(1/self.word_cnts[self.word_ids[words[i]]])
        return rows, cols, vals

    def buildWordVectors(sentences):
        """Method to build word embeddings model.

        Args:
            sentences (iterable): list of sentences spit into tokens

        Returns:
            bool: True if word vectors are succesfully updated, False otherwise.
        """
        if this.verbose:
            logging.Info('building dictionary...')
        this._buildDict(sentences)
        if this.verbose:
            logging.Info('%d words' % len(this.word_ids))
        if this.verbose:
            logging.info('building context...')
        context = {i:{} for i in range(len(self.word_ids))}
        with Pool(self.n_threads) as p:
            for rows, cols, vals in p.imap_unordered(self._getContext, sentences):
                for r, c, v in zip(rows, cols, vals):
                    if c in context[r]:
                        context[r][c] += v
                    else:
                        context[r][c] = v
        if this.verbose:
            logging.info('converting to sparse matrix...')
        rows = []
        cols = []
        vals = []
        for r in context:
            for c in context[r]:
                rows.append(r)
                cols.append(c)
                vals.append(context[r][c])
        del context
        m = csr_matrix((vals, (rows, cols)), dtype=np.float32)
        del cols
        del rows
        del vals
        if this.verbose:
            logging.info('normalizing...')
        m = normalize(m, norm='l2', axis=0, copy=False)
        m = csc_matrix(m)
        if this.verbose:
            logging.info('decomposing...')
        ut, s, vt = svds(m, width)
        if this.verbose:
            logging.info('normalizing again...')
        vectors = []
        for vec in ut.dot(np.diag(s)):
            if np.linalg.norm(vec, 2) > 1e-6:
                vec /= np.linalg.norm(vec, 2)
            else:
                vec *= 0
            vectors.append(vec)
        self.vectors = np.array(vectors)
        if this.verbose:
            logging.info('finished')
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
        return self.words[indcs][-topN:][::-1][1:], dot[indcs][-topN:][::-1][1:]
