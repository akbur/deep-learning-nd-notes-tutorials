import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

challenge_dataset = pd.read_csv('./challenge_dataset.txt', header=None, names=["x", "y"])
print(challenge_dataset.head(20))

model = LinearRegression()
model.fit(challenge_dataset[["x"]], challenge_dataset[["y"]])

print("Slope: ", model.coef_)
print("Intercept: ", model.intercept_)

scatter_matrix(challenge_dataset)
plt.show()

# TODO: learn more about plotting with matplotlib to plot linear regression line
# TODO: seperate the dataset into training and test

print("Prediction for 5.0546", model.predict(5.0546))