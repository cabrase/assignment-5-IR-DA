"""
This script analyzes a dataset of Lego sets obtained from lego.com. It performs the following tasks:
1. Creates a scatter plot depicting the relationship between the piece count and the list price for Lego sets.
2. Saves the scatter plot to the "out" folder.
3. Lists the top Lego sets with the highest piece count to dollar ratio.

Credit to ChatGPT using the following prompt:
“I want to make a new python file that opens the lego_sets.csv file and
analyzes how price changes as piece count increases and graphs it.”
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

__author__ = "Carson Brase"
__copyright__ = "Copyright 2023, Westmont College, Carson Brase"
__credits__ = ["Carson Brase"]
__license__ = "MIT"
__email__ = "cbrase@westmont.edu"

# Load the Lego Sets data from the CSV file
lego_data = pd.read_csv('/Users/CarsonBrase/Desktop/CS128/assignment-5-IR:DA/data/lego_sets.csv')

# Extract relevant columns
piece_count = lego_data['piece_count']
list_price = lego_data['list_price']

# Convert list_price to numeric (removing dollar signs, if any)
list_price = pd.to_numeric(list_price.replace('[\\$,]', '', regex=True))

# Calculate the piece count to dollar ratio
piece_count_to_dollar_ratio = piece_count / list_price

# Add a new column to the DataFrame for the ratio
lego_data['piece_count_to_dollar_ratio'] = piece_count_to_dollar_ratio

# Sort the DataFrame by the ratio in descending order to get the highest ratios first
sorted_lego_data = lego_data.sort_values(by='piece_count_to_dollar_ratio', ascending=False)

exclude_names = ["Creative Box", "Go Brick Me"]
filtered_lego_data = sorted_lego_data[~sorted_lego_data['set_name'].isin(exclude_names)]
# Display the top n sets with the highest piece count to dollar ratio
top_sets = sorted_lego_data.head(10)

# Print the top n sets
print("Sets with Highest Piece Count to Dollar Ratio:")
print(top_sets[['set_name', 'piece_count_to_dollar_ratio']])

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(piece_count, list_price, alpha=0.5)
plt.title('Price vs Piece Count for Lego Sets')
plt.xlabel('Piece Count')
plt.ylabel('List Price')
plt.grid(True)

# Create the "out" folder if it doesn't exist
output_folder = '/Users/CarsonBrase/Desktop/CS128/assignment-5-IR:DA/out'
os.makedirs(output_folder, exist_ok=True)

# Save the plot to a file in the "out" folder
output_file_path = os.path.join(output_folder, 'price_vs_piece_count.png')
plt.savefig(output_file_path)

# Show the plot
plt.show()
