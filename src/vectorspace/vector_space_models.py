"""Abstract data type definitions for vector space model that supports
   cosine similarity queries using TF-IDF matrix built from the corpus.
"""

import sys
import concurrent.futures
from math import sqrt, log10
from typing import Callable, Iterable
from nltk.stem import StemmerI
from nltk.stem.snowball import SnowballStemmer

__author__ = "Carson Brase"
__copyright__ = "Copyright 2023, Westmont College, Mike Ryu"
__credits__ = ["Carson Brase", "Mike Ryu"]
__license__ = "MIT"
__email__ = "mryu@westmont.edu"


class Vector:
    """
    Class used for keeping track of TF-IDF scores of terms within documents in the form of a list.

    Includes methods for calculating the Euclidean norm, dot product, and cosine similarity scores.

    Attributes:
        _vec (list[float]): returns elements stored within the Vector
    """
    def __init__(self, elements: list[float] | None = None):
        self._vec = elements if elements else []

    def __getitem__(self, index: int) -> float:
        if index < 0 or index >= len(self._vec):
            raise IndexError(f"Index out of range: {index}")
        else:
            return self._vec[index]

    def __setitem__(self, index: int, element: float) -> None:
        if 0 <= index < len(self._vec):
            self._vec[index] = element
        else:
            raise IndexError(f"Index out of range: {index}")

    def __eq__(self, other) -> bool:
        if other is self:
            return True
        elif other is None or not isinstance(other, Vector):
            return False
        else:
            return self._vec == other.vec

    def __str__(self) -> str:
        return str(self._vec)

    @property
    def vec(self):
        return self._vec

    @staticmethod
    def _get_cannot_compute_msg(computation: str, instance: object):
        return f"Cannot compute {computation} with an instance that is not a DocumentVector: {instance}"

    def norm(self) -> float:
        """Euclidean norm of the vector."""
        norm_sq = 0.0
        # If vector isn't None, compute norm
        if self._vec is not None:
            for x in self._vec:
                norm_sq += (x ** 2)
            return sqrt(norm_sq)
        # If None, return 0
        else:
            return 0.0

    def dot(self, other: object) -> float:
        """Dot product of `self` and `other` vectors."""
        if not isinstance(other, Vector):
            raise ValueError(self._get_cannot_compute_msg("dot product", other))
        else:
            # If other is a vector, compute dot product
            dot_product = sum(x * y for x, y in zip(self._vec, other._vec))
            return dot_product

    def cossim(self, other: object) -> float:
        """Cosine similarity of `self` and `other` vectors."""
        if not isinstance(other, Vector):
            raise ValueError(self._get_cannot_compute_msg("cosine similarity", other))
        else:
            # Save numerator and denominator
            numerator = self.dot(other)
            denominator = (self.norm() * other.norm())
            # Ensure not dividing by 0
            if denominator:
                return numerator / denominator
            else:
                return 0.0

    def boolean_intersect(self, other: object) -> list[tuple[float, float]]:
        """Returns a list of tuples of elements where both `self` and `other` had nonzero values."""
        if not isinstance(other, Vector):
            raise ValueError(self._get_cannot_compute_msg("boolean intersection", other))
        else:
            return [(e1, e2) for e1, e2 in zip(self._vec, other._vec) if e1 and e2]


class LegoSet:
    """
    Class responsible for the tracking of a given LegoSet within a corpus.

    Utilizes NLTK's Snowball Stemmer for the processing of product description.

    Attributes:
        _title (str): Name of the LegoSet
        _words (list[str]): List of all the words (unfiltered and not stemmed) in a LegoSet product description
        processors=: input chosen stemmer and words to exclude for filter_words
        _price (float): list_price of a LegoSet
        _difficulty (str): build difficulty of a LegoSet
    """
    _iid = 0

    def __init__(self, title: str = None, words: list[str] = None, processors: tuple[set[str], SnowballStemmer] = None,
                 list_price: float = 0.0, review_difficulty: str = ""):
        LegoSet._iid += 1
        self._iid = LegoSet._iid
        self._title = title if title else f"(Untitled {self._iid})"
        self._words = list(words) if words else []
        self._price = list_price
        self._difficulty = review_difficulty

        if processors:
            exclude_words = processors[0]
            stemmer = processors[1]
            if not isinstance(exclude_words, set) or not isinstance(stemmer, StemmerI):
                raise ValueError(f"Invalid processor type(s): ({type(exclude_words)}, {type(stemmer)})")
            else:
                self.stem_words(stemmer)
                self.filter_words(exclude_words)

    def __iter__(self):
        return iter(self._words)

    def __eq__(self, other) -> bool:
        if other is self:
            return True
        elif other is None or not isinstance(other, LegoSet):
            return False
        else:
            return self._title == other.title and self._words == other.words

    def __hash__(self) -> int:
        return hash((self._title, tuple(self._words)))

    def __str__(self) -> str:
        words_preview = ["["]
        preview_size = 5
        index = 0

        while index < len(self._words) and index < preview_size:
            words_preview.append(f"{self._words[index]}, ")
            index += 1
        words_preview.append("... ]")

        return "[{i:04d}]: {title} {words}".format(
            i=self._iid,
            title=self._title,
            words="".join(words_preview)
        )

    @property
    def iid(self):
        return self._iid

    @property
    def title(self):
        return self._title

    @property
    def words(self):
        return self._words

    @property
    def price(self):
        return self._price

    @property
    def difficulty(self):
        return self._difficulty


    def filter_words(self, exclude_words: set[str]) -> None:
        """Removes any words from `_words` that appear in `exclude_words` passed in."""
        filtered_words = []

        # Loop through all words in document
        for word in self._words:
            # Add word to new list if it is in alphanumeric characters and not an excluded word
            if word.isalpha() and word not in exclude_words:
                filtered_words.append(word)

        self._words = filtered_words

    def stem_words(self, stemmer: SnowballStemmer) -> list:
        """Stems each word in `_words` using the `stemmer` passed in."""
        words = []
        self._words = self._words

        # Loop through each word in the document and stem it
        for word in self._words:
            word = stemmer.stem(word)
            words.append(word)
        # Assign the new stemmed words list to the document _words attribute
        self._words = words

        return words

    def tf(self, term: str) -> int:
        """Computes and returns the term frequency of the `term` passed in among `_words`."""
        freq = 0
        # Simple counter of word frequencies
        for word in self._words:
            if word == term:
                freq += 1
        return freq


class Corpus:
    """
    Class for tracking a list of documents and the creation of TF-IDF vectors and matrices.

    Attributes:
        _docs (list[Document]): List of all descriptions of Lego Sets
        _terms (dict[str, int]): Dictionary of the filtered, unique, and stemmed words
        _dfs (dict[str, int]): Dictionary of the terms and their associated document frequencies
        _tf_idf (dict[str, Vector]): Dictionary of the LegoSet title and associated document vector
    """
    def __init__(self, lego_sets: list[LegoSet], threads=1, debug=False):
        self._docs: list[LegoSet] = lego_sets

        # Setting flags.
        self._threads: int = threads
        self._debug: bool = debug

        # Bulk of the processing (and runtime) occurs here.
        self._terms = self._compute_terms()
        self._dfs = self._compute_dfs()
        self._tf_idf = self._compute_tf_idf_matrix()

    def __getitem__(self, index) -> LegoSet:
        if 0 <= index < len(self._docs):
            return self._docs[index]
        else:
            raise IndexError(f"Index out of range: {index}")

    def __iter__(self):
        return iter(self._docs)

    def __len__(self):
        return len(self._docs)

    @property
    def docs(self):
        return self._docs

    @property
    def terms(self):
        return self._terms

    @property
    def dfs(self):
        return self._dfs

    @property
    def tf_idf(self):
        return self._tf_idf

    def _compute_terms(self) -> dict[str, int]:
        """Computes and returns the terms (unique, stemmed, and filtered words) of the corpus."""
        words = [word for doc in self.docs for word in doc.words]
        index_dict = self._build_index_dict(words)
        return index_dict

    def _compute_df(self, term) -> int:
        """Computes and returns the document frequency of the `term` in the context of this corpus (`self`)."""
        if self._debug:
            print(f"Started working on DF for '{term}'")
            sys.stdout.flush()

        def check_membership(t: str, doc: LegoSet) -> bool:
            """An efficient method to check if the term `t` occurs in a list of words `doc`."""
            for word in doc:
                if t == word:
                    return True
            return False

        return sum([1 if check_membership(term, doc) else 0 for doc in self._docs])

    def _compute_dfs(self) -> dict[str, int]:
        """Computes document frequencies for each term in this corpus and returns a dictionary of {term: df}s."""
        if self._threads > 1:
            return Corpus._compute_dict_multithread(self._threads, self._compute_df, self._terms.keys())
        else:
            return {term: self._compute_df(term) for term in self._terms.keys()}

    def _compute_tf_idf(self, term, doc=None, index=None):
        """Computes and returns the TF-IDF score for the term and a given document.

        An arbitrary document may be passed in directly (`doc`) or be passed as an `index` within the corpus.
        """
        dfs = self._dfs
        doc = self._get_doc(doc, index)

        df = dfs.get(term)
        # Check term membership in doc frequency dictionary
        if term in dfs.keys():
            # Calculations for TF-IDF
            tf = log10(1 + doc.tf(term))
            idf = log10(len(self.docs)/(1 + df))
            tf_idf = tf * idf
            # Checking for None types and term membership in _terms
            if term in self._terms and len(self.docs) > 1:
                return tf_idf
        else:
            return 0.0

    def compute_tf_idf_vector(self, doc=None, index=None) -> Vector:
        """Computes and returns the TF-IDF vector for the given document.

        An arbitrary document may be passed in directly (`doc`) or be passed as an `index` within the corpus.
        """
        doc = self._get_doc(doc, index)
        all_tf_idfs = []

        # Iterate through all words in the unique document terms
        for word in self._terms:
            # Calculate tf_idf for word in doc and add to new list
            tf_idf = self._compute_tf_idf(word, doc=doc)
            all_tf_idfs.append(tf_idf)

        # Create vector of all tf-idf scores and return it
        vec = Vector(all_tf_idfs)
        return vec

    def _compute_tf_idf_matrix(self) -> dict[str, Vector]:
        """Computes and returns the TF-IDF matrix for the whole corpus.

        The TF-IDF matrix is a dictionary of {document title: TF-IDF vector for the document}.

        """
        def tf_idf(document):
            if self._debug:
                print(f"Processing '{document.title}'")
                sys.stdout.flush()
            vector = self.compute_tf_idf_vector(doc=document)
            return vector

        matrix = {}
        if self._threads > 1:
            matrix = Corpus._compute_dict_multithread(self._threads, tf_idf, self._docs,
                                                      lambda d: d, lambda d: d.title)
        else:
            for doc in self._docs:
                # For all the documents, calculate document TF-IDF vector
                doc_vector = tf_idf(doc)
                matrix[doc.title] = doc_vector

                if self._debug:
                    print(f"Done with doc {doc.title}")
        return matrix

    def _get_doc(self, document, index):
        """A helper function to None-guard the `document` argument and fetch documents per `index` argument."""
        if document is not None and index is None:
            return document
        elif index is not None and document is None:
            if 0 <= index < len(self):
                return self._docs[index]
            else:
                raise IndexError(f"Index out of range: {index}")

        elif document is None and index is None:
            raise ValueError("Either document or index is required")
        else:
            raise ValueError("Either document or index must be passed in, not both")

    @staticmethod
    def _compute_dict_multithread(num_threads: int, op: Callable, iterable: Iterable,
                                  op_arg_func= lambda x: x, key_arg_func=lambda x: x) -> dict:
        """Experimental generic multithreading dispatcher and collector to parallelize dictionary construction.

        Args:
            num_threads (int): maximum number of threads (workers) to utilize.
            op: (Callable): operation (function or method) to execute.
            iterable: (Iterable): iterable to call the `op` on each item.
            op_arg_func: a function that maps an item of the `iterable` to an argument for the `op`.
            key_arg_func: a function that maps an item of the `iterable` to the key to use in the resulting dict.

        Returns:
            A dictionary of {key_arg_func(an item of `iterable`): op(p_arg_func(an item of `iterable`))}.

        """
        result = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_keys = {executor.submit(op, op_arg_func(item)): key_arg_func(item) for item in iterable}
            for future in concurrent.futures.as_completed(future_to_keys):
                key = future_to_keys[future]
                try:
                    result[key] = future.result()
                except Exception as e:
                    print(f"Key '{key}' generated exception:", e, file=sys.stderr)
        return result

    @staticmethod
    def _build_index_dict(lst: list) -> dict:
        """Given a list, returns a dictionary of {item from list: index of item}."""
        return {item: index for (index, item) in enumerate(lst)}

