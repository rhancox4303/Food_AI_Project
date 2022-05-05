import csv

# Define the filename.
filename = "data.csv"

# Assign food item to it's row in the CSV file.
nutrition_index = {
    "dumplings": 0,
    "french fries": 1,
    "chicken wings": 2,
    "fried rice": 3,
    "steak": 4,
    "pancakes": 5,
    "pizza": 6,
    "pho": 7,
    "hot dog": 8,
    "cheesecake": 9,
    "beans": 10,
    "cabbage": 11,
    "radish": 12,
    "potato": 13,
    "carrot": 14,
}

# Read the file into nutrition_data.
with open(filename, 'r') as data:
    nutrition_data = [{heading: x for heading, x in row.items()}
                      for row in csv.DictReader(data, skipinitialspace=True)]


# Return the nutrition index.
def get_index():
    return nutrition_index;


# Return the nutrition data.
def get_data():
    return nutrition_data;