import pandas as pd
import numpy as np
import sklearn
from keras.src.activations import linear
from sklearn import linear_model
from sklearn.utils import shuffle
import tensorflow
import keras
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# predicting the final grade. This is also known as a label.
predict = "G3"


# defining the attributes.
x = np.array(data.drop([predict], axis=1))
# defining the labels.
y = np.array(data[predict])
# splitting the labels and attributes into 4 variables: "x_train, y_train, x_test, y_test".
# x_train is a section of x array.
# y_train is a section of y array.
# x_test, y_test: these are testing the accuracy of the model we are going to create.
# (Splitting the data into training and testing sets) The code below is splitting %10 of our data into test samples so when we test we can test off of that and the model has never seen the actual data. (if the model sees the actual data it will memorize it, and we don't want that).
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

# creating a training model:
"""linear = linear_model.LinearRegression()
# fitting the data to find the best fit line and store it in "linear".
linear.fit(x_train, y_train)
# testing the accuracy of our model.
acc = linear.score(x_test, y_test)
print(acc)

with open("studentmodel.pickle", "wb") as f:
    pickle.dump(linear, f)"""

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)) :
    #printing the prediction with the actual attributes and the actual final grade
    print(predictions[x], x_test[x], y_test[x])