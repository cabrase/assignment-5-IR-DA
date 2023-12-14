"""Query driver for the vector space model using NLTK's Inaugural corpus.
"""

import sys
import time
import pickle
import argparse
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from src.vectorspace.vector_space_models import LegoSet, Corpus, Vector

__author__ = ["Mike Ryu", "Carson Brase"]
__copyright__ = "Copyright 2023, Westmont College, Mike Ryu"
__credits__ = ["Mike Ryu", "Carson Brase"]
__license__ = "MIT"
__email__ = ["mryu@westmont.edu", "cbrase@westmont.edu"]


def main() -> None:
    """
    Credit to ChatGPT for edits, changing from the NLTK inaugural corpus to the lego_sets dataset.
    Using the following prompt and providing reference code:
    “For this main method, instead of taking in all the words from the NLTK inaugural corpus, I need it
    to read in a file called 'lego_sets.csv' and read in all words in the column 'prod_long_desc'”
    """
    pars = setup_argument_parser()
    args = pars.parse_args()
    timer = Timer()

    document_processors = (set(stopwords.words('english')), SnowballStemmer('english'))

    try:
        with open(args.pickle_file_path, "rb") as pickle_file:
            corpus = timer.run_with_timer(pickle.load, [pickle_file],
                                          label="corpus load from pickle")
    except FileNotFoundError:
        # Read words from 'lego_sets.csv' in the 'prod_long_desc' column
        lego_data = pd.read_csv('/Users/CarsonBrase/Desktop/CS128/assignment-5-IR/DA/data/lego_sets.csv')
        lego_documents = [
            LegoSet(file_name, word_tokenize(description), document_processors, list_price, review_difficulty)
            for file_name, description, list_price, review_difficulty
            in zip(lego_data['set_name'], lego_data['prod_long_desc'],
                   lego_data['list_price'], lego_data['review_difficulty'])]

        corpus = timer.run_with_timer(Corpus, [lego_documents, args.num_threads, args.debug],
                                      label="corpus instantiation (includes TF-IDF matrix)")
        with open(args.pickle_file_path, "wb") as pickle_file:
            pickle.dump(corpus, pickle_file)

    keep_querying(corpus, document_processors, 10)


def setup_argument_parser() -> argparse.ArgumentParser:
    pars = argparse.ArgumentParser(prog="python3 -m vectorspace.vector_space_runner")
    pars.add_argument("num_threads", type=int,
                      help="required integer indicating how many threads to utilize")
    pars.add_argument("pickle_file_path", type=str,
                      help="required string containing the path to a pickle (data) file")
    pars.add_argument("-d", "--debug", action="store_true",
                      help="flag to enable printing debug statements to console output")
    return pars


def keep_querying(corpus: Corpus, processors: tuple[set[str], SnowballStemmer], num_results: int) -> None:
    again_response = 'y'

    while again_response == 'y':
        # Added in two more queries to account for user budget and desired build difficulty
        raw_query = input("What kind of Lego Set are you looking for? ")
        price_budget = float(input("What is your price budget? "))
        difficulty = input("What is your desired difficulty? (Very Easy, Easy, Average, Challenging) ")

        query_document = LegoSet("query", raw_query.split(), processors=processors)
        query_vector = corpus.compute_tf_idf_vector(query_document)

        filtered_results = filter_results(query_vector, corpus, price_budget, difficulty, num_results)
        display_query_result(raw_query, filtered_results, corpus, num_results)

        again_response = input("Again (y/N)? ").lower()


def filter_results(query_vector: Vector, corpus: Corpus, price_budget: float,
                   difficulty: str, num_results: int) -> list[tuple[str, float]]:
    """
    Method for filtering the resulting Lego Sets based on the three user inputs
    Credit to ChatGPT using the following prompt and providing reference code:
    “I would like the program to ask the user for two additional inputs and then filter
    out lego sets that are above the price budget or above the desired difficulty:
        1. What is your price budget? (float. Corresponds to 'list_price' attribute in the lego set dataset)
        2. What is your desired difficulty? (str. Corresponds to 'review_difficulty' attribute in the lego set dataset)”
    """
    filtered_results = []

    for title, doc_vector in corpus.tf_idf.items():
        cosine_similarity = query_vector.cossim(doc_vector)
        set_instance = next((doc for doc in corpus.docs if doc.title == title), None)

        # Logic for filtering results based on price and difficulty
        if set_instance and set_instance.price <= price_budget and set_instance.difficulty == difficulty:
            filtered_results.append((title, cosine_similarity))

    # Return Lego sets sorted by their similarity scores
    return sorted(filtered_results, key=lambda item: item[1], reverse=True)[:num_results]


def display_query_result(query: str, filtered_results: list[tuple[str, float]], corpus: Corpus, num_results: int) -> None:
    if num_results > len(filtered_results):
        num_results = len(filtered_results)
    # Variables for tracking average price of Lego sets
    total_price = 0.0
    count = 0
    print(f"\nFor query: {query}")
    for i in range(num_results):
        title, score = filtered_results[i]
        set_instance = next((doc for doc in corpus.docs if doc.title == title), None)
        if set_instance:
            total_price += set_instance.price
            count += 1
            print(f"Result {i + 1:02d} : [{score:0.6f}] {title} (Price: ${set_instance.price}, Difficulty: {set_instance.difficulty})")
    # Calculate average Lego set price for the sets returned by the query
    print(f"\nAverage Set Price: ${round((total_price / count if count != 0 else 0), 2)}")
    print()


class Timer:
    def __init__(self):
        self._start = 0.0
        self._stop = 0.0

    def run_with_timer(self, op, op_args=None, label="operation"):
        if not op_args:
            op_args = []

        self.start()
        result = op(*op_args)
        self.stop()

        self.print_elapsed(label=label)
        return result

    def print_elapsed(self, label: str = "operation", file=sys.stdout):
        print(f"Elapsed time for {label}: {self.get_elapsed():0.4f} seconds", file=file)

    def get_elapsed(self) -> float:
        return self._stop - self._start

    def start(self) -> None:
        self._start = time.time()

    def stop(self) -> None:
        self._stop = time.time()


if __name__ == '__main__':
    main()