import openpyxl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

wb = openpyxl.load_workbook("data.xlsx")
ws = wb["Sheet1"]

X = []
Y = []
for row in ws.iter_rows(min_row=2, values_only=True):
    X.append(row[0])
    Y.append(row[1])

X = np.array(X).reshape(-1,1)
Y = np.array(Y)

model = LinearRegression()
model.fit(X, Y)

x_plot = np.linspace(X.min(), X.max(), 200).reshape(-1,1)
y_plot = model.predict(x_plot)

plt.scatter(X, Y)
plt.plot(x_plot, y_plot)
plt.title("Linear Regression")
plt.show()
