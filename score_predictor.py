import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


def regression(intercept, slope, features):
    """
    Performs a simple linear regression algorithm
    :param intercept: x-intercept of the function
    :param slope: weights of the features
    :param features: function features
    :return: Predicted answer
    """
    prediction = intercept + slope*features
    return prediction


def cost_function(intercept, slope, features, target):
    """
    Calculates the cost function of the algorithm
    :return:
    """
    answer = regression(intercept, slope, features)
    loss = tf.keras.losses.mse(target, answer)
    return loss


student_data = pd.read_csv(r"D:\Usman\Data Science\The Sparks Foundation\Task 1\student_scores.csv")
# converting data into a numpy array
hours = np.asarray(student_data['Hours'])
scores = np.asarray(student_data['Scores'])
# displaying the data to see the dataframe structure
print(student_data.head())
print(student_data.tail())
student_data.describe()

# setting up hyper parameters
learning_rate = 0.05
training_epochs = 1000

# initializing the function weights and bias as a tensorflow random variable
rng = np.random
theta0 = tf.Variable(rng.randn(), name="Bias")
theta = tf.Variable(rng.randn(), name="Weights")

# setting up the optimizer and training regression model
opt = tf.keras.optimizers.Adam(learning_rate)
for j in range(training_epochs):
    opt.minimize(lambda: cost_function(theta0, theta, hours, scores),
                 var_list=[theta0, theta])

# taking input from the user for predictive analysis
enter_hours = float(input("Enter number of hours studied "
                          "to predict the student score: "))
score = theta0 + (theta * enter_hours)
print("The student is predicted to score {0} by studying {1} "
      "hours".format(score, enter_hours))

# plotting data and the learned model function
student_data.plot(kind='scatter', x='Hours', y='Scores',
                  figsize=(14, 7), marker='x', color='r',
                  title='Student score variation by hours studied',
                  )
x = np.linspace(0, 10, num=20)
y = theta0 + (theta * x)
plt.grid()
plt.plot(x, y)
plt.show()
