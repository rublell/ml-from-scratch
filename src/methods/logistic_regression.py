import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label


class LogisticRegression:
    """
    Multi-class logistic regression (softmax) trained with batch gradient descent.
    """

    def __init__(self, lr, max_iters=500):
        """
        Initialize the model.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters
        self.W = None

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): class labels of shape (N,)
        Returns:
            pred_labels (np.array): predicted labels of shape (N,)
        """
        N = training_data.shape[0]

        # Add the bias term as an extra input feature.
        x = np.concatenate([np.ones((N, 1)), training_data], axis=1)

        # Multi-class logistic regression uses one-hot labels.
        C = get_n_classes(training_labels)
        y_onehot = label_to_onehot(training_labels, C)

        # One column of W is learned for each class.
        self.W = np.zeros((x.shape[1], C))

        for _ in range(self.max_iters):
            scores = x @ self.W

            # Stabilize the softmax to avoid very large exponentials.
            scores = scores - np.max(scores, axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # Gradient of the multi-class cross-entropy loss.
            grad = (x.T @ (probs - y_onehot)) / N
            self.W = self.W - self.lr * grad

        pred_onehot = probs
        pred_labels = onehot_to_label(pred_onehot)
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        N = test_data.shape[0]

        # Use the same bias convention as during training.
        x = np.concatenate([np.ones((N, 1)), test_data], axis=1)

        scores = x @ self.W

        # Same numerical stabilization as in fit().
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # The predicted class is the one with the highest softmax score.
        pred_labels = onehot_to_label(probs)
        return pred_labels
