import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA as sklearnPCA

class Logistic_Regression(object):
    def __init__(self, learning_rate=0.01, epochs=50):
        self.__epochs= epochs
        self.__learning_rate = learning_rate

    def fit(self, X, Y):
        self.weight = np.zeros(X.shape[1])
        self.bias = 0
        self.cost_ = []
        self.gradient_devent(X,Y)

    def gradient_devent(self, X, Y):
        for i in range(self.__epochs):
            # 1- Calculate the net input X * W^T
            # 2- Get the activation using Sigmoid function
            # 3- Calculate the gradient
            # 4- Update the weights and bias using the gradient and learning rate
            # 5- Uncomment the cost collecting line


            #self.cost_.append(self.__logit_cost(y, self.__activation(y_)))
          pass

    def __logit_cost(self, y, y_val):
        logit = -y.dot(np.log(y_val)) - ((1 - y).dot(np.log(1 - y_val)))

        return logit

    def __sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def __net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def __activation(self, X):
        return self.__sigmoid(X)

    def predict(self, X):

        pass
    def plot(self):
        plt.plot(range(1, len(self.cost_) + 1), np.log10(self.cost_))
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.title('Logistic Regression - Learning rate 0.1')
        plt.tight_layout()
        plt.show()


def read_dataset(Normalize=1):
    data = pd.read_csv('../../Dataset/Iris/Iris_2_classes.csv')#read dataset
    data['Species'] = data['Species'].replace(["Iris-setosa", "Iris-versicolor"], (0, 1))#encode label
    y = data['Species']#lable
    x = data.drop(['Species', 'Id'], axis=1)#drop lable and id
    x = np.asarray(x)#put it in array
    y = np.asarray(y)#put it in array
    x = np.nan_to_num(x)#convert any nan to 0
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=50)#split dataset to 80 train and 20 test
    if Normalize == 1:#normalize dataset
        scaler = MinMaxScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)
    # draw data
    pca = sklearnPCA(n_components=2)  # 2-dimensional PCA
    transX = pd.DataFrame(pca.fit_transform(x))
    plt.scatter(transX[y == 0][0], transX[y == 0][1], label='Class 1', c='red')
    plt.scatter(transX[y == 1][0], transX[y == 1][1], label='Class 2', c='blue')
    plt.legend()
    plt.show()
    ##################
    return x , y , train_x, train_y, test_x, test_y

x , y , train_x, train_y, test_x, test_y = read_dataset(Normalize = 1)

lr = Logistic_Regression(learning_rate=0.01, epochs=100)

lr.fit(train_x ,train_y)

predict_y = lr.predict(test_x)

correct = np.sum(predict_y == test_y)

print("%d out of %d predictions correct" % (correct, len(predict_y)))

print("accuracy = ", correct / len(predict_y) * 100)

lr.plot()