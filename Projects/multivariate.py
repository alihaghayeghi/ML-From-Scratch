import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


dataset = pd.read_csv("FuelConsumption.csv")


x1 = dataset.iloc[:, 4:5].values
x2 = dataset.iloc[:, 5:6].values
x3 = dataset.iloc[:, 8:9].values
x4 = dataset.iloc[:, 9:10].values
x5 = dataset.iloc[:, 10:11].values
x6 = dataset.iloc[:, 11:12].values

X = np.concatenate((x1, x2, x3, x4, x5, x6), axis=1)

y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = np.zeros(X_test.shape[0])

for i in range(X_test.shape[0]):
    y_pred[i] = model.intercept_
    for j in range(X_test.shape[1]):
        y_pred[i] += model.coef_[j] * X_test[i, j]
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R² Score: {r2}")
