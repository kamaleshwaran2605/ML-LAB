import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = np.array([
    [25, 50000],
    [30, 60000],
    [45, 80000],
    [35, 65000],
    [22, 48000],
    [40, 78000],
    [28, 52000],
    [50, 90000]
])

y = np.array([0, 0, 1, 1, 0, 1, 0, 1])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = DecisionTreeClassifier(
    criterion="gini",   
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Predictions:", y_pred)
print("Accuracy:", accuracy)


plt.figure(figsize=(12, 6))
plot_tree(
    model,
    feature_names=["Age", "Income"],
    class_names=["No", "Yes"],
    filled=True
)
plt.show()