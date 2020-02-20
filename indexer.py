import os
from collections import defaultdict
from collections import Counter
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, PunktSentenceTokenizer

class RawTextIndexer:
    """
    Processes raw text files from a directory and allows to query constructed
    index for a given word.

    Attributes:
        _index  Internal data structure which holds word offsets [see _make_index()]
        _freq   Holds frequency of the words encountered
        _reader Streaming text reader from NLTK library
    """
    def __init__(self, txtDir):
        # keep an index of words
        self._index = defaultdict(dict)
        # and their frequency
        self._freq = Counter()
        
        assert os.path.isdir(txtDir)

        # takes care of punctuation better than WordPunctTokenizer
        word_token = RegexpTokenizer(r'\w+')
        sent_token = PunktSentenceTokenizer()
        self._reader = PlaintextCorpusReader(txtDir, r'.*\.txt', word_tokenizer=word_token, sent_tokenizer=sent_token)


    def _make_index(self, fname, sents, words):
        """
        Constructs a custom index of words from sentences.
        The finished index looks like this:
        'word1' -> {
            'filename1' -> [(1, 0), (2, 0), (2, 5), ...]
            'filename2' -> [(5, 9), (6, 0), (6, 16), ...]
            ...
        }
        'word2' -> {
            'filename1' -> [...]
            'filename2' -> [...]
            ...
        }

        This function builds a list of tuples for a given filename. The tuples
        represent an index of a sentence and an index of a word within the sentence.

        Parameters:
            fname   Filename containing sentences and words
            sents   List of sentences in the order they appear in the file.
            words   List of words in the order they appear in the sentences.
        """
        for w in words:
            # word index for this file only
            findex = []

            for ixS, s in enumerate(sents):
                # iterate over each word in the sentencep
                for ixT, token in enumerate(s):
                    # could use regex for substring matching instead
                    if w == token.lower():
                        findex.append((ixS, ixT))
                        # keep track of word use frequency
                        self._freq[w] += 1

            # grow the main index 
            self._index[w][fname]= findex


    def index_files(self):
        """
        Processes files from a directory.
        """
        stop_words = set(stopwords.words('english'))

        for fname in self._reader.fileids():
            # lowercase all words upfront - may be an issue for "us" vs "US"
            all_words = set([w.lower() for w in self._reader.words(fname)])

            # clean up common words
            words = [w for w in all_words if w not in stop_words]
            sents = self._reader.sents(fname)
            self._make_index(fname, sents, words)


    def fetch_index(self, word):
        """
        Returns lists of files and sentences from the index for a given word.
        """
        files_ = []
        sents_ = []
        # pull dictionaries specific to the token
        for fname in self._index[word]:
            # preserve filename
            files_.append(fname)

            # format tokens for output
            for i, j in self._index[word][fname]:
                s = self._reader.sents(fname)[i]  # list
                s[j] = '*' + s[j] + '*'
                sents_.append(' '.join(s))

        return (files_, sents_)


    def format_top_n(self, n=10):
        """
        Picks top N most frequent words and formats the names of files and sentences
        into a list suitable for further output.
        """
        output = []
        for t, c in self._freq.most_common(n):
            files_, sents_ = self.fetch_index(t)
            word = t + ' (' + str(c) + ')'
            output.append([word, ','.join(files_), "\n".join(sents_)])

        return output
