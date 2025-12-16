from sklearn.model_selection import train_test_split
import pandas as pd


X = pd.read_pickle("X_train.pkl")
y = pd.read_pickle("y_train.pkl")

X["y"] = y

existing_customer_num = X["y"].value_counts()["Existing Customer"]
attrited_customer_num = X["y"].value_counts()["Attrited Customer"]

print("Number of Existing Customers are: {}".format(existing_customer_num))
print("Number of Attrited Customers are: {}".format(attrited_customer_num))

#Select random sample of majority class(868)
df_sample_maj_class = X[X['y']=="Existing Customer"].sample(n=attrited_customer_num, random_state=1)

df = pd.concat([df_sample_maj_class,X[X['y']=="Attrited Customer"]],ignore_index=True)

df.to_pickle("training_random_sample.pkl")


