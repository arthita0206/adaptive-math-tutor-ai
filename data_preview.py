import pandas as pd

# Load the training data
df = pd.read_csv('train.csv')

# Preview the first 5 rows
print(df.head())

# View all the column names
print("\nColumns in your dataset:")
print(df.columns)
