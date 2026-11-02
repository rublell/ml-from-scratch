import numpy as np


class LinearRegression:
    """
    Ordinary least-squares linear regression (closed-form solution).
    """

    def __init__(self):
        """
        Initialize the model.
        """
        # Learned weights, including the bias coefficient.
        self.weights = None

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): regression target of shape (N,)
        Returns:
            pred_labels (np.array): target of shape (N,)
        """
        # Add a column of ones so the model can learn the bias term.
        x = np.concatenate([np.ones((training_data.shape[0], 1)), training_data], axis=1)

        # Closed-form least-squares solution using the pseudo-inverse.
        self.weights = np.linalg.pinv(x) @ training_labels

        pred_labels = x @ self.weights
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        # Use the same bias convention as during training.
        x = np.concatenate([np.ones((test_data.shape[0], 1)), test_data], axis=1)
        pred_labels = x @ self.weights
        return pred_labels
