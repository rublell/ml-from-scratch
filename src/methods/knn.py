import numpy as np


class KNN:
    """
    k-nearest neighbors for classification (majority vote) or regression (mean).
    """

    def __init__(self, k=1, task_kind="classification"):
        """
        Initialize the model.
        """
        self.k = k
        self.task_kind = task_kind

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): labels of shape (N,)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        # KNN has no real training step, so we keep the data for prediction.
        self.training_data = training_data
        self.training_labels = training_labels

        pred_labels = self.predict(training_data)
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            test_labels (np.array): labels of shape (N,)
        """
        pred_labels = []

        for x in test_data:
            # Compare the test point with every stored training point.
            distances = np.linalg.norm(self.training_data - x, axis=1)

            # Keep only the labels of the k closest neighbors.
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.training_labels[nearest_indices]

            if self.task_kind == "classification":
                # Classification: use majority vote among the neighbors.
                values, counts = np.unique(nearest_labels, return_counts=True)
                pred = values[np.argmax(counts)]
            else:
                # Regression: use the average value of the neighbors.
                pred = np.mean(nearest_labels)

            pred_labels.append(pred)

        test_labels = np.array(pred_labels)
        return test_labels
