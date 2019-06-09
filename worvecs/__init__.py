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
    def __init__(self, sentences=None, bins=2, bin_width=3, pctl=50, width=100,
        encoding=0):
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

    # def _builContext(self, sentences):
    #     """Method to build the word context counts.
    #
    #     Args:
    #         sentences (iterable): list of sentences spit into tokens
    #     """
    #     context = [{} for i in range(len(self.words))]
    #     for s in sentences:
    #         for i in range(len(s)):
    #             if s[i] not in self.word_ids:
    #                 continue
    #             window_start = max([i-self.window, 0])
    #             window_end = min([i+self.window, len(s)])
    #             context_space = max([self.window - i, 0])
    #             for j in range(window_start, window_end):
    #                 if i == j or s[j] not in self.word_ids:
    #                     continue
    #                 context_id = context_space*len(self.words) + \
    #                     self.word_ids[s[j]]
    #                 context_space += 1
    #                 if context_id in context[self.word_ids[s[i]]]:
    #                     context[self.word_ids[s[i]]][context_id] += 1
    #                 else:
    #                     context[self.word_ids[s[i]]][context_id] = 1
    #     return context

    def _builContext(self, sentences):
        """Method to build the word context counts.

        Args:
            sentences (iterable): list of sentences spit into tokens
        """
        context = [{} for i in range(len(self.words))]
        width = self.bin_width*self.bins
        bin_ids = {}
        for i in range(-self.bins,  self.bins):
            for j in range(i* self.bin_width, (i+1)*self.bin_width):
                bin_ids[j + (0 if i < 0 else 1)] = i + self.bins
        for s in sentences:
            for i in range(len(s)):
                if s[i] not in self.word_ids:
                    continue
                window_start = max([i-width, 0])
                window_end = min([i+width, len(s)])
                for j in range(window_start, window_end):
                    if i == j or s[j] not in self.word_ids:
                        continue
                    context_id = bin_ids[j-i]*len(self.words) + \
                        self.word_ids[s[j]]
                    if context_id in context[self.word_ids[s[i]]]:
                        context[self.word_ids[s[i]]][context_id] += 1
                    else:
                        context[self.word_ids[s[i]]][context_id] = 1
        return context

    def buildWordVectors(self, sentences):
        """Method to build word embeddings model.

        Args:
            sentences (iterable): list of sentences spit into tokens

        Returns:
            bool: True if word vectors are succesfully updated, False otherwise.
        """
        self._buildDict(sentences)
        context = self._builContext(sentences)

        data = []
        indcs = []
        indptr = [0]
        for w in self.words:
            wid = self.word_ids[w]
            wf = self.word_cnts[wid]
            vec = np.zeros(len(self.words)*2*self.bins*self.bin_width)
            for c in context[wid]:
                context_id = int(abs(c/len(self.words) - \
                    c//len(self.words))*len(self.words))
                cf = self.word_cnts[context_id]
                wcf = context[wid][c]
                vec[c] = self._encoding(wcf, wf, cf)
            if np.linalg.norm(vec, 2) > 1e-6:
                vec /= np.linalg.norm(vec, 2)
            nonzero_indcs = np.nonzero(vec)[0]
            data += list(vec[nonzero_indcs])
            indcs += list(nonzero_indcs)
            indptr.append(indptr[-1] + len(nonzero_indcs))

        m = csc_matrix(csr_matrix((data, indcs, indptr)))
        if m.shape[0] <= self.width or m.shape[1] <= self.width:
            raise Exception("error")
            return False
        ut, s, vt = svds(m, self.width)

        vectors = []
        for vec in ut.dot(np.diag(s)):
            if np.linalg.norm(vec, 2) > 1e-6:
                vec /= np.linalg.norm(vec, 2)
            else:
                vec *= 0
            vectors.append(vec)
        self.vectors = np.array(vectors)
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
