import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

#Using panda to read in the file
data = pd.read_csv("KNN/car.data")
# print(data.head())

#Taking the labels and encoding them to integer values
le = preprocessing.LabelEncoder()
#Taking the columns and turning them into a list and then transforming those into integer values
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

#Converting everything into one big list
x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

#Assigning K
model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

#Uses the trained K-Nearest Neighbors (KNN) model to predict the class labels for the test data x_test
predicted = model.predict(x_test)
#Defines a list of class names corresponding to the encoded class labels.
names = ["unacc", "acc", "good", "vgood"]

#For each test instance, it prints the predicted class name, the test data instance, and the actual class name.
for x in range(len(x_test)):
    print("Predicted ", names[predicted[x]], "Data ", x_test[x], "Actual: ", names[y_test[x]])
