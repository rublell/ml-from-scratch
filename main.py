import argparse
import numpy as np
import os
import time

from src.methods.logistic_regression import LogisticRegression
from src.methods.linear_regression import LinearRegression
from src.methods.knn import KNN
from src.utils import normalize_fn, accuracy_fn, macrof1_fn, mse_fn

np.random.seed(100)


def main(args):
    """Load the dataset, train the selected model and report its metrics."""

    dataset_path = args.data_path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # 1. Load the data

    feature_data = np.load(dataset_path, allow_pickle=True)
    train_features, test_features, train_labels_reg, test_labels_reg, train_labels_classif, test_labels_classif = (
        feature_data["xtrain"],
        feature_data["xtest"],
        feature_data["ytrainreg"],
        feature_data["ytestreg"],
        feature_data["ytrainclassif"],
        feature_data["ytestclassif"],
    )

    # 2. Validation split and standardization

    eval_name = "Test set"

    if not args.test:
        n = train_features.shape[0]
        indices = np.random.permutation(n)
        split = int(0.8 * n)

        train_idx = indices[:split]
        val_idx = indices[split:]

        test_features = train_features[val_idx]
        train_features = train_features[train_idx]

        test_labels_reg = train_labels_reg[val_idx]
        train_labels_reg = train_labels_reg[train_idx]

        test_labels_classif = train_labels_classif[val_idx]
        train_labels_classif = train_labels_classif[train_idx]

        eval_name = "Validation set"

    # Normalization based only on the training set
    means = np.mean(train_features, axis=0, keepdims=True)
    stds = np.std(train_features, axis=0, keepdims=True)
    stds[stds == 0] = 1

    train_features = normalize_fn(train_features, means, stds)
    test_features = normalize_fn(test_features, means, stds)

    # 3. Model

    if args.method == "knn":
        method_obj = KNN(k=args.K, task_kind=args.task)

    elif args.method == "logistic_regression":
        method_obj = LogisticRegression(lr=args.lr, max_iters=args.max_iters)

    elif args.method == "linear_regression":
        method_obj = LinearRegression()

    else:
        raise ValueError(f"Unknown method: {args.method}")

    print(f"\nMethod: {args.method}")
    print(f"Task: {args.task}")
    print(f"Train size: {train_features.shape[0]}")
    print(f"{eval_name} size: {test_features.shape[0]}")

    if args.method == "knn":
        print(f"K = {args.K}")
    elif args.method == "logistic_regression":
        print(f"Learning rate = {args.lr}")
        print(f"Max iterations = {args.max_iters}")

    # 4. Training and evaluation

    if args.task == "classification":
        assert args.method != "linear_regression", "linear regression is a regression method"

        s1 = time.time()
        preds_train = method_obj.fit(train_features, train_labels_classif)
        s2 = time.time()

        preds = method_obj.predict(test_features)
        s3 = time.time()

        acc = accuracy_fn(preds_train, train_labels_classif)
        macrof1 = macrof1_fn(preds_train, train_labels_classif)
        print(f"\nTrain set: {acc:.3f}% accuracy - {macrof1:.6f} macro-F1")

        acc = accuracy_fn(preds, test_labels_classif)
        macrof1 = macrof1_fn(preds, test_labels_classif)
        print(f"{eval_name}: {acc:.3f}% accuracy - {macrof1:.6f} macro-F1")

        print(f"Training time: {s2 - s1:.6f} s")
        print(f"Prediction time: {s3 - s2:.6f} s")

    elif args.task == "regression":
        assert args.method != "logistic_regression", "logistic regression is a classification method"

        s1 = time.time()
        preds_train = method_obj.fit(train_features, train_labels_reg)
        s2 = time.time()

        preds = method_obj.predict(test_features)
        s3 = time.time()

        train_mse = mse_fn(preds_train, train_labels_reg)
        print(f"\nTrain set: MSE = {train_mse:.6f}")

        test_mse = mse_fn(preds, test_labels_reg)
        print(f"{eval_name}: MSE = {test_mse:.6f}")

        print(f"Training time: {s2 - s1:.6f} s")
        print(f"Prediction time: {s3 - s2:.6f} s")

    else:
        raise ValueError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="classification",
        type=str,
        help="classification / regression",
    )
    parser.add_argument(
        "--method",
        default="logistic_regression",
        type=str,
        help="knn / logistic_regression / linear_regression",
    )
    parser.add_argument(
        "--data_path",
        default="data/features.npz",
        type=str,
        help="path to the .npz dataset",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=1,
        help="number of neighboring datapoints used for knn",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="learning rate for methods with learning rate",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=100,
        help="max iters for methods which are iterative",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="train on whole training data and evaluate on the test data, otherwise use a validation set",
    )

    args = parser.parse_args()
    main(args)