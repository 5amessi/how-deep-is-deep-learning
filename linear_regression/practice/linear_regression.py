import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#gradient decent you should implement it
def step_gradient(b_current, m_current, points, learningRate):
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # h(x) = predicted_y = mx + b
        # m is slope, b is y-intercept or m is theta 1, b is theta 0
        # Squared error function
        # theta0 = theta0 + eta * (1/n)*sum(y(i) - h(xi))
        # theta1 = theta1 + eta * (1/n)*sum(y(i) - h(xi))*xi

    return [b_current, m_current] #return theta0 , theta1

#fun to coputer error
def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(2*len(points))

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    counter = 0 # counter used for the drawing
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        #   The drawing staff, we will update it once after each 10 iterations
        print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(counter, b, m,compute_error_for_line_given_points(b, m, points)))
        if counter%100 is 0:
            plt.plot(points[:, 0], points[:, 1], 'bo') # Draw the dataset
            plt.plot([0, 80], [b, 80*m+b], 'b')
            plt.show()
        b, m = step_gradient(b, m, np.array(points), learning_rate)
        counter+=1
    return [b, m]

def Train():
    print ("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    print ("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))


#========================================================================================================================================================
#The Main:
#read_dataset
points = pd.read_csv("../Dataset/Regression_dataset/data.csv", delimiter=",") # Function in pandas that reads the data from a file and organize it.
points = np.asarray(points) #make it as
#hyberprameters
learning_rate = 0.000001 # Eta
num_iterations = 2000
initial_b = 0 # initial y-intercept guess
initial_m = 0 # initial slope guess
# m is slope, b is y-intercept or b is theta 0 , m is theta 1

Train()
#code cycle
#1- read dataset , hyberprameters
#2- Train
#3- gradient_descent_runner
#4- step_gradient where u will write your code #####
