import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#hyper parameters
learning_rate = 0.01
epochs = 5000
def train(X, Y):
    weight = np.random.rand(len(X[0]))# Support Vector
    bias = 0 #to improve the classifier
    for epoch in range(1, epochs):
        #itterate on every element in train dataset
        #check if it missclassify or not
        #if it missclassify update weights as its
        for i, raw_x in enumerate(X):
            if (Y[i] * (np.dot(raw_x, weight)+bias)) > 1:
                weight += learning_rate * (-2 * (1 / epoch) * weight)
                bias += learning_rate * (-2 * (1 / epoch) * bias)
            else:
                weight += learning_rate * ((raw_x * Y[i]) + (-2 * (1 / epoch) * weight))
                bias += learning_rate * ((1 * Y[i]) + (-2 * (1 / epoch) * bias))
    return weight , bias

def test(X,weight , bias):
    return np.sign(X.dot(weight)+bias)

def read_dataset(Normalize = 1):
    train = pd.read_csv('Dataset/Titanic/train.csv')
    train_y = train['Survived']
    train = train.drop(['Survived','Name','Ticket','Cabin'] , axis=1)
    train['Sex'] = train['Sex'].replace(['male','female'] , (1,0))
    train['Embarked'] = train['Embarked'].replace(['C','S','Q'] , (0,1,2))
    train_x = np.asarray(train)
    train_y = np.asarray(train_y)
    train_x = np.nan_to_num(train_x)
    train_x, test_x , train_y,test_y = train_test_split(train_x, train_y,test_size=0.2, random_state=0)
    if Normalize == 1:
        scaler = MinMaxScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)
    return train_x ,train_y ,test_x ,test_y
train_x ,train_y ,test_x ,test_y = read_dataset()

train_y[train_y == 0] = -1
test_y[test_y == 0] = -1

weight,bias = train(train_x ,train_y)

predict_y = test(test_x,weight,bias)

correct = np.sum(predict_y == test_y)

print("%d out of %d predictions correct" % (correct, len(predict_y)))

print("accuracy = ", correct / len(predict_y) * 100)
