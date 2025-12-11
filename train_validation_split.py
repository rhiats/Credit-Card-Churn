from sklearn.model_selection import train_test_split
import pandas as pd


X = pd.read_pickle("X_train.pkl")
y = pd.read_pickle("y_train.pkl")

X_train, X_validation, y_train, y_validation = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)

X_train.to_pickle("X_train.pkl")
X_validation.to_pickle("X_validation.pkl")
y_train.to_pickle("y_train.pkl")
y_validation.to_pickle("y_validation.pkl")




