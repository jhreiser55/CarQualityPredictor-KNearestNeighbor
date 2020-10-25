import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
import pandas as pd
import numpy as np

#loading in the data
data = pd.read_csv("car.data")

#Takes the labels and initializes them in to relative computable integers
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
doors = le.fit_transform(list(data["doors"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
clss = le.fit_transform(list(data["class"]))

predict = "class"

#Converts all above data in to a single list for the Label then class is the Feature
x = list(zip(buying, maint, doors, persons, lug_boot, safety))
y = list(clss)

#training and testing the model, same as linear regression
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

#Creating the model based off KNN
model = KNeighborsClassifier(n_neighbors=9)

#Training the model
model.fit(x_train, y_train)

#Finding out how accurate our model is
accuracy = model.score(x_test, y_test)

#Getting the actual predictions with the rest of the input
predicted_data = model.predict(x_test)

#printing out the data to determine the quality of car and showing the magnitude to closest neighbors
names = ["unacc", "acc", "good", "vgood"]
for x in range(len(predicted_data)):
    print("Predicted: ", names[predicted_data[x]], ", Data: ", x_test[x], ", Actual: ", names[y_test[x]])
    magnitude = model.kneighbors([x_test[x]], 5, True)
    print("Magnitude to Neighbor: ", magnitude)

