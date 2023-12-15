import unittest
from nltk.stem import SnowballStemmer
from src.vectorspace.vector_space_models import Vector, LegoSet, Corpus

__author__ = "Carson Brase"
__copyright__ = "Copyright 2023, Westmont College, Carson Brase"
__credits__ = ["Carson Brase"]
__license__ = "MIT"
__email__ = "cbrase@westmont.edu"


class VectorTest(unittest.TestCase):
    def setUp(self):
        self.v1 = Vector([3.0, 4.0])
        self.v2 = Vector([5.0, 12.0])

        self.v3 = Vector([3.0, 4.0, 5.0])
        self.v4 = Vector([6.0, 7.0, 8.0])

        self.v5 = Vector([])
        self.v6 = Vector(None)

    def test_norm(self):
        self.assertEqual(self.v1.norm(), 5.0)
        self.assertEqual(self.v2.norm(), 13.0)

        self.assertAlmostEqual(self.v3.norm(), 7.071, places=3)
        self.assertAlmostEqual(self.v4.norm(), 12.207, places=3)

        self.assertEqual(self.v5.norm(), 0.0)
        self.v6._vec = None
        self.assertEqual(self.v6.norm(), 0.0)

    def test_dot(self):
        self.assertEqual(self.v1.dot(self.v2), 63.0)
        self.assertEqual(self.v3.dot(self.v4), 86.0)

    def test_cossim(self):
        self.assertAlmostEqual(self.v1.cossim(self.v2), 0.96923, places=5)
        self.assertAlmostEqual(self.v3.cossim(self.v4), 0.99637, places=5)
        self.assertEqual(self.v6.cossim(self.v6), 0.0)


class DocumentTest(unittest.TestCase):
    def setUp(self):
        words1 = ["dog", "cat", "fish", "bat", "pig", "horse", "cow"]
        words2 = ["eating", "runs", "programming", "jumping", "coding"]
        words3 = ["strawberry", "strawberry", "blueberry", "apple", "apple", "apple", "kiwi"]

        self.exclude_words = {"dog", "fish", "pig", "cow"}
        stemmer = SnowballStemmer('english')

        self.doc1 = LegoSet(title="Set 1", words=words1, processors=(self.exclude_words, stemmer),
                            list_price=40.0, review_difficulty="Easy")
        self.doc2 = LegoSet(title="Set 2", words=words2, processors=(self.exclude_words, stemmer),
                            list_price=50.0, review_difficulty="Average")
        self.doc3 = LegoSet(title="Set 3", words=words3, processors=(self.exclude_words, stemmer),
                            list_price=60.0, review_difficulty="Challenging")

    def test_lego_set(self):
        a = self.doc1.title
        b = self.doc1.words
        c = self.doc1.price
        d = self.doc1.difficulty
        self.assertEqual(a, "Set 1")
        self.assertEqual(b, ["cat", "bat", "hors"])
        self.assertEqual(c, 40.0)
        self.assertEqual(d, "Easy")


    def test_filter_words(self):
        self.doc1.filter_words(self.exclude_words)

        self.assertEqual(self.doc1.words, ["cat", "bat", "hors"])

    def test_stem_words(self):
        stemmer = SnowballStemmer('english')
        self.doc2.stem_words(stemmer=stemmer)
        self.assertEqual(self.doc2.words, ["eat", "run", "program", "jump", "code"])

    def test_tf(self):
        self.assertEqual(self.doc3.tf("appl"), 3)
        self.assertEqual(self.doc3.tf("strawberri"), 2)
        self.assertEqual(self.doc3.tf("kiwi"), 1)
        self.assertEqual(self.doc3.tf("blueberri"), 1)


class CorpusTest(unittest.TestCase):
    def setUp(self):
        words1 = ["dog", "dog", "cat", "bat", "apple", "fish"]
        words2 = ["eating", "runs", "programming", "jumping", "running", "runs", "eats", "cow"]
        words3 = ["strawberry", "strawberry", "apple", "apple", "apple", "pig"]

        words4 = ["running", "jumping", "swimming"]
        words5 = ["eating", "playing", "walked"]

        exclude_words = {"fish", "pig", "cow"}
        stemmer = SnowballStemmer('english')

        self.doc1 = LegoSet(title="Doc 1", words=words1, processors=(exclude_words, stemmer),
                            list_price=40.0, review_difficulty="Easy")
        self.doc2 = LegoSet(title="Doc 2", words=words2, processors=(exclude_words, stemmer),
                            list_price=50.0, review_difficulty="Average")
        self.doc3 = LegoSet(title="Doc 3", words=words3, processors=(exclude_words, stemmer),
                            list_price=60.0, review_difficulty="Challenging")

        self.doc4 = LegoSet(title="doc 4", words=words4, processors=(exclude_words, stemmer),
                            list_price=70.0, review_difficulty="Easy")
        self.doc5 = LegoSet(title="doc 5", words=words5, processors=(exclude_words, stemmer),
                            list_price=45.0, review_difficulty="Very Easy")

        self.corp1 = Corpus([self.doc1, self.doc2, self.doc3])
        self.corp2 = Corpus([])
        self.corp3 = Corpus([self.doc4, self.doc5])

    def test_compute_terms(self):
        terms_dict = self.corp3._compute_terms()
        terms = terms_dict.keys()
        test_dict = {"run": 0, "jump": 1, "swim": 2, "eat": 0, "play": 1, "walk": 2}
        test_terms = test_dict.keys()
        self.assertEqual(terms, test_terms)

        terms_dict1 = self.corp2._compute_terms()
        self.assertEqual(terms_dict1, {})

    def test_compute_df(self):
        df = self.corp1._compute_df("appl")
        self.assertEqual(df, 2)

        df1 = self.corp2._compute_df("appl")
        self.assertEqual(df1, 0)

    def test_compute_tf_idf(self):
        tf_idf1 = self.corp1._compute_tf_idf("cat", doc=self.doc1)
        self.assertAlmostEqual(tf_idf1, 0.053009, places=6)

        tf_idf2 = self.corp1._compute_tf_idf("appl", doc=self.doc3)
        self.assertEqual(tf_idf2, 0)

        tf_idf3 = self.corp1._compute_tf_idf("run", doc=self.doc2)
        self.assertAlmostEqual(tf_idf3, 0.1060175, places=7)

        tf_idf4 = self.corp2._compute_tf_idf("run", doc=self.doc1)
        self.assertEqual(tf_idf4, 0.0)

    def test_compute_tf_idf_vector(self):

        i = 0
        vec_list = []
        print(self.corp1.terms)
        while i < len(self.corp1.terms):
            for word in self.corp1.terms:
                score = self.corp1._compute_tf_idf(word, doc=self.doc1)
                vec_list.append(score)
                i += 1
        test_vector = Vector(vec_list)

        tf_idf_vector = self.corp1.compute_tf_idf_vector(doc=self.doc1)

        print("actual: ", tf_idf_vector)
        print("expected: ", test_vector)

        self.assertEqual(tf_idf_vector, test_vector)

    def test_compute_tf_idf_matrix(self):
        tf_idf_vec1 = self.corp1.compute_tf_idf_vector(doc=self.doc1)
        tf_idf_vec2 = self.corp1.compute_tf_idf_vector(doc=self.doc2)
        tf_idf_vec3 = self.corp1.compute_tf_idf_vector(doc=self.doc3)

        test_matrix = {self.doc1.title: tf_idf_vec1, self.doc2.title: tf_idf_vec2, self.doc3.title: tf_idf_vec3}

        tf_idf_matrix = self.corp1._compute_tf_idf_matrix()

        self.assertEqual(tf_idf_matrix, test_matrix)


if __name__ == '__main__':
    unittest.main()
