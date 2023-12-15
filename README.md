# Assignment 5: Information Retrieval and Data Analysis
**Westmont College Fall 2023**

**CS 128 Information Retrieval and Big Data**

*Assistant Professor* Mike Ryu (mryu@westmont.edu) 

## Presentation Link
Click [here](https://docs.google.com/presentation/d/1ehfVuy2LTHe2juXYsHSdMzEnnCG2U4A1tMujq4q9qvE/edit?usp=sharing
) for the Google Slides presentation.

## Autor Information
* **Name(s)**: Carson Brase
* **Email(s)**: cbrase@westmont.edu
* Demo Code: `round`

## License
License information: [MIT](https://github.com/git/git-scm.com/blob/main/MIT-LICENSE.txt)

## Problem Description

Have you ever wished you could find the perfect Lego set for you based on your budget and difficulty preference but 
got tired of the endless scrolling through the Lego website? This is the exact problem I sought to solve with this 
program. I also take a look into how to find the best value Lego set, looking into piece count to dollar ratios.

## Solution Description

To create this program, I drew on the information retrieval techniques utilized in Assignment 3 and expanded on 
them to account for additional user inputs.

### LegoSet class

I modified the `Document` class to include the `price` and `difficulty` attributes.
  - Added new methods `get_price` and `get_difficulty` to retrieve the price and difficulty values.

### main method

I modified the `main` method so that instead of taking in all the words from the NLTK inaugural corpus, 
it reads in a file called 'lego_sets.csv' and creates LegoSet instances using all words in the dataset 
column `prod_long_desc`, the prices in `list_price`, and the difficulties in `review_difficulty`.

**Credit:** ChatGPT for writing this method using the following prompt:

    For this main method, instead of taking in all the words from the NLTK inaugural corpus, I need it
    to read in a file called 'lego_sets.csv' and read in all words in the column 'prod_long_desc

### keep_querying method

I added in two more user input queries to this method to account for user budget and desired build difficulty.

### filter_results

This method filters the resulting Lego Sets based on the three user inputs from `keep_querying`.

**Credit:** ChatGPT for writing this method using the following prompt:

    I would like the program to ask the user for two additional inputs and then filter
    out lego sets that are above the price budget or above the desired difficulty:
        1. What is your price budget? (float. Corresponds to 'list_price' attribute in the lego set dataset)
        2. What is your desired difficulty? (str. Corresponds to 'review_difficulty' attribute in the lego set dataset)

### display_query_result

The filtering from this method was moved to the `filter_results` method. An additional calculation was added to find 
the average price for the Lego sets returned by the query.

### analyze_lego_sets.py

The purpose of this file is to analyze the piece count to dollar ratios of each Lego set and visualize it on a graph. 
It also output the Lego sets with the 10 highest piece court to dollar ratios as the "best value" Lego sets.

**Credit:** ChatGPT for writing this method using the following prompt:

      I want to make a new python file that opens the lego_sets.csv file and
      analyzes how price changes as piece count increases and graphs it.

# LEGO Set Search and Analysis Tool

Welcome to the Lego Set Search and Analysis Tool! This tool allows you to search for Lego sets based on your budget
and build difficulty preferences and analyze the dataset to gain insights into piece count to dollar ratios.

## Getting Started

Follow these instructions to set up and run the Lego Set Search and Analysis Tool on your local machine.

### Prerequisites

Install the required Python packages.

Run the following code in your terminal:

    pip install -r requirements.txt

### Usage

#### Lego Set Search

1. Run the `vector_space_runner.py` file and answer the following prompts to search for Lego sets 
based on your preferences:
   1. What kind of Lego Set are you looking for?
   2. What is your price budget?
   3. What is your desired difficulty? (Very Easy, Easy, Average, Challenging)
      1. Be sure to only enter one of these 4 options exactly (case-sensitive) to ensure correct results.
2. View the resulting Lego sets and choose one to your liking!

#### Piece Count to Dollar Ratio Analysis

1. Run the `analyze_lego_sets.py` file

2. View the console output to see the top Lego sets with the highest piece count to dollar ratio.

3. Check the "out" folder for the scatter plot visualizing the price vs. piece count for Lego sets.

## Credits

### Mike Ryu

Code from Mike Ryu's [Assignment 3](https://github.com/cs-with-mike/assignment-3-vectorspace-cabrase/tree/main) was 
utilized and altered in the creation of this program.

### ChatGPT

ChatGPT was utilized throughout this project to assist with the creation of certain methods including:
`main`, `filter_results`, and `analyze_lego_sets.py`. See their specific subheadings to view the prompts that generated 
this code.

### Lego Dataset

From user 'MattieTerzolo' on [kaggle.com](https://www.kaggle.com/datasets/mterzolo/lego-sets)

Permission: [CC0 Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)
