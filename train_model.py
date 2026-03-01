import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# EXAMPLE DATA — Replace with your dataset
data = {
    'followers': [10, 300, 5, 1000, 20, 2, 5000, 15],
    'following': [500, 100, 1000, 150, 200, 300, 50, 800],
    'has_profile_pic': [0, 1, 0, 1, 0, 0, 1, 1],
    'has_bio': [0, 1, 0, 1, 0, 0, 1, 1],
    'fake': [1, 0, 1, 0, 1, 1, 0, 0]
}

df = pd.DataFrame(data)

X = df[['followers', 'following', 'has_profile_pic', 'has_bio']]
y = df['fake']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

pickle.dump(model, open("model.pkl", "wb"))

print("MODEL TRAINED AND SAVED AS model.pkl")
