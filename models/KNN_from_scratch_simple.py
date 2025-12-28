import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_indices]
        return Counter(k_labels).most_common(1)[0][0]


def main():
    columns = ['BI-RADS', 'Age', 'Shape', 'Margin', 'Density', 'Severity']

    data = pd.read_csv(
        'mammographic_masses.data',
        names=columns,
        na_values='?'
    )

    data.dropna(inplace=True)

    X = data.drop('Severity', axis=1).values
    y = data['Severity'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    knn = KNN(k=7)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    plt.figure(figsize=(8, 6))
    plt.scatter(
        X_train[y_train == 0, 0],
        X_train[y_train == 0, 1],
        color='green',
        label='Benign',
        alpha=0.6
    )
    plt.scatter(
        X_train[y_train == 1, 0],
        X_train[y_train == 1, 1],
        color='red',
        label='Malignant',
        alpha=0.6
    )
    plt.xlabel('Feature 1 (Scaled)')
    plt.ylabel('Feature 2 (Scaled)')
    plt.title('Mammographic Mass Dataset - KNN')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
