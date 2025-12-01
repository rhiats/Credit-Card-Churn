from sklearn.model_selection import train_test_split
import pandas as pd


df = pd.read_csv("credit_card_churn.csv")

X = df.iloc[:, :20]
y = df["Attrition_Flag"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)

X_train.to_pickle("X_train.pkl")
X_test.to_pickle("X_test.pkl")
y_train.to_pickle("y_train.pkl")
y_test.to_pickle("y_test.pkl")




