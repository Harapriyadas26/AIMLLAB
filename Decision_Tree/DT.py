import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load dataset
iris = load_iris()
X = iris.data
y = iris.target


# Split dataset (80% training, 20% testing) with equal class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)


# Create Decision Tree model
model = DecisionTreeClassifier(max_depth=3, random_state=1)


# Train the model
model.fit(X_train, y_train)


# Predict test data
y_pred = model.predict(X_test)


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Evaluation")
print("Accuracy:", round(accuracy,4))


# Classification report formatting
report = classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names,
    output_dict=True
)

print("\nClassification Report")
print(f"{'Class':<12}{'precision':<10}{'recall':<10}{'f1-score':<10}{'support'}")

print(f"{'Setosa':<12}{report['setosa']['precision']:<10.2f}{report['setosa']['recall']:<10.2f}{report['setosa']['f1-score']:<10.2f}{int(report['setosa']['support'])}")
print(f"{'Versicolor':<12}{report['versicolor']['precision']:<10.2f}{report['versicolor']['recall']:<10.2f}{report['versicolor']['f1-score']:<10.2f}{int(report['versicolor']['support'])}")
print(f"{'Virginica':<12}{report['virginica']['precision']:<10.2f}{report['virginica']['recall']:<10.2f}{report['virginica']['f1-score']:<10.2f}{int(report['virginica']['support'])}")

print("\nOverall Accuracy =", round(accuracy*100,2), "%")


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)


# Visualize confusion matrix
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
print("\nProgram Finished")