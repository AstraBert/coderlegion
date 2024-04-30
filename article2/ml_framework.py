import pandas as pd

df = pd.read_csv("italian_cities.csv")

X = df.drop(columns=["Destination"])
y = df["Destination"]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model = model.fit(X_train, y_train)


from sklearn.metrics import accuracy_score, classification_report
y_pred = model.predict(X_test)
print(f"Accuracy is {accuracy_score(y_test, y_pred)}")
print(f"Classification report:\n{classification_report(y_test,y_pred)}")

import joblib

joblib.dump(model, "model.joblib")