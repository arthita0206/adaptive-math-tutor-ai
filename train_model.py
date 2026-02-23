import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load cleaned data
df = pd.read_csv('clean_train.csv')

# Feature engineering: convert categorical 'level' and 'type' to numbers
df['level_code'] = df['level'].str.extract('(\d+)').astype(int)
df['type_code'] = df['type'].astype('category').cat.codes

# Optional feature: problem length (indicator of complexity)
df['problem_length'] = df['problem'].str.len()

# Prepare X and y — simulate as a binary classification (right/wrong)
# Here, let's say all correct answers are '1' (simulate for now)
X = df[['level_code', 'type_code', 'problem_length']]
y = [1] * len(df)  # All correct by default, to demonstrate process

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)

# Save the trained model
with open('quiz_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("Model trained and saved as quiz_model.pkl")
