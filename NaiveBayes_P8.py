import numpy as np
import matplotlib.pyplot as plt
import sklearn

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split  into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Naive Bayes classifier
nb_classifier = GaussianNB()

# Train 
nb_classifier.fit(X_train, y_train)

# Manually predict for new data
new_data_point = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example data point
predicted_label = nb_classifier.predict(new_data_point)
predicted_class = iris.target_names[predicted_label]
print("Predicted class:", predicted_class[0])

# Plot the data points colored by class
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=20, label='Data')
plt.scatter(new_data_point[:, 0], new_data_point[:, 1], c='blue', marker='X', s=150, label='New Data Point')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Data Points with New Data Point')
plt.show()