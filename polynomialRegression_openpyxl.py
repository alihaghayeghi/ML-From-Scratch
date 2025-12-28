import openpyxl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

wb = openpyxl.load_workbook("data.xlsx")
ws = wb["Sheet1"]

X = []
Y = []
for row in ws.iter_rows(min_row=2, values_only=True):
    X.append(row[0])
    Y.append(row[1])

X = np.array(X).reshape(-1,1)
Y = np.array(Y)

degrees = [1, 2, 5, 10]
x_plot = np.linspace(X.min(), X.max(), 200).reshape(-1,1)

plt.scatter(X, Y)

for d in degrees:
    poly = PolynomialFeatures(degree=d, include_bias=False)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, Y)
    x_poly_plot = poly.transform(x_plot)
    y_plot = model.predict(x_poly_plot)
    plt.plot(x_plot, y_plot, label=f"Degree {d}")

plt.legend()
plt.title("Polynomial Regression")
plt.show()
