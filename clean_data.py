import pandas as pd

# Load the training data
df = pd.read_csv('train.csv')

# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())

# Remove rows with missing answers and 'Level ?'
df = df[df['answer'].notnull()]
df = df[df['level'] != 'Level ?']

# Optionally strip whitespace from topic/difficulty labels
df['level'] = df['level'].str.strip()
df['type'] = df['type'].str.strip()

# Print unique difficulty levels and math topics for inspection
print("Difficulty Levels after cleaning:", df['level'].unique())
print("Math Topics after cleaning:", df['type'].unique())

# Save cleaned data
df.to_csv('clean_train.csv', index=False)

print("Cleaned data saved as clean_train.csv")
