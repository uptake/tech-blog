# ./python_code/make_model.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

## GETTING AND PARSING THE DATA

training_data = pd.read_csv("iris.data")

# Appropriately name the columns; see details on the data website.
training_data.columns = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "class",
]

# Split up the columns of the dataframe above into features and labels.
feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
label_cols = ["class"]

# Sklearn can be picky with how the data is formatted...
X = training_data.loc[:, feature_cols]
y = training_data.loc[:, label_cols].values.ravel()

# Make our training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

## TRAINING A MODEL
# We don't use anything fancy, just plain old Logistic Regression!
# We use cross validation here, but you could also use a training-test set.

# Instantiate and fit the model:
logreg = LogisticRegression(solver="liblinear", multi_class="ovr")
clf = logreg.fit(X_train, y_train)

# See if the model is reasonable.
print("Score: ", clf.score(X_test, y_test))

# Pickle to save the model for use in our API.
pickle.dump(clf, open("./our_model.pkl", "wb"))
