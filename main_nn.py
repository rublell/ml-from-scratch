import argparse
import numpy as np
import os
import time

from src.methods.mlp import MLP
from src.losses import MSE, WeightedCrossEntropy
from src.activations import Sigmoid, ReLU, Identity, Softmax
from src.methods.kmeans import KMeans
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn, \
    label_to_onehot, onehot_to_label, get_n_classes

np.random.seed(100)


def main(args):
    """Train and evaluate the MLP or the K-Means classifier."""

    dataset_path = args.data_path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # 1. Load the data

    feature_data = np.load(dataset_path, allow_pickle=True)
    train_features, test_features, train_labels_reg, test_labels_reg, train_labels_classif, test_labels_classif = (
        feature_data['xtrain'],feature_data['xtest'],feature_data['ytrainreg'],
        feature_data['ytestreg'],feature_data['ytrainclassif'],feature_data['ytestclassif']
    )

    # 2. Validation split and standardization

    eval_name = "Test set"
    if not args.test:
        n = train_features.shape[0]
        idx = np.random.permutation(n)
        n_val = int(args.val_ratio * n)
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        test_features = train_features[val_idx]
        test_labels_reg = train_labels_reg[val_idx]
        test_labels_classif = train_labels_classif[val_idx]
        train_features = train_features[train_idx]
        train_labels_reg = train_labels_reg[train_idx]
        train_labels_classif = train_labels_classif[train_idx]
        eval_name = "Validation set"

    # normalize with the training statistics only (no leakage)
    means = train_features.mean(axis=0, keepdims=True)
    stds = train_features.std(axis=0, keepdims=True)
    stds[stds == 0] = 1.0
    train_features = normalize_fn(train_features, means, stds)
    test_features = normalize_fn(test_features, means, stds)

    # 3. Model
    if args.method == "kmeans":
        method_obj = KMeans(K=args.K, max_iters=args.max_iters)

    elif args.method == "mlp":
        input_dim = train_features.shape[1]
        if args.task == "classification":
            n_classes = get_n_classes(train_labels_classif)
            # --weighted: softmax output + weighted cross-entropy (handles class imbalance)
            out_act = Softmax if args.weighted else Sigmoid
            method_obj = MLP(dimensions=(input_dim, args.hidden_nodes, n_classes),
                             activations=(ReLU, out_act))
        else:  # regression -> linear output
            method_obj = MLP(dimensions=(input_dim, args.hidden_nodes, 1),
                             activations=(ReLU, Identity))
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # 4. Training and evaluation

    if args.task == "classification":

        if args.method == "mlp":
            n_classes = get_n_classes(train_labels_classif)
            y_train = label_to_onehot(train_labels_classif, n_classes)
            if args.weighted:
                # inverse-frequency class weights for the weighted cross-entropy
                counts = np.bincount(train_labels_classif.astype(int), minlength=n_classes)
                weights = len(train_labels_classif) / (n_classes * counts)
                loss_fn = WeightedCrossEntropy(weights)
            else:
                loss_fn = MSE
            s1 = time.time()
            method_obj.fit(train_features, y_train, loss=loss_fn,
                           epochs=args.max_iters, batch_size=args.batch_size, learning_rate=args.lr)
            s2 = time.time()
            preds_train = onehot_to_label(method_obj.predict(train_features))
            preds = onehot_to_label(method_obj.predict(test_features))
            s3 = time.time()
        else:
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
        assert args.method != "kmeans", "K-Means is a classification method"

        if args.method == "mlp":
            y_train = train_labels_reg.reshape(-1, 1).astype(float)
            s1 = time.time()
            method_obj.fit(train_features, y_train, loss=MSE,
                           epochs=args.max_iters, batch_size=args.batch_size, learning_rate=args.lr)
            s2 = time.time()
            preds_train = method_obj.predict(train_features).reshape(-1)
            preds = method_obj.predict(test_features).reshape(-1)
            s3 = time.time()
        else:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="classification",
        type=str,
        help="classification / regression / clustering",
    )
    parser.add_argument(
        "--method",
        default="mlp",
        type=str,
        help="kmeans / mlp",
    )
    parser.add_argument(
        "--data_path",
        default="data/features.npz",
        type=str,
        help="path to your dataset CSV file",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=10,
        help="number of clusters datapoints used for kmeans",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
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
        help="train on whole training data and evaluate on the test data, "
             "otherwise use a validation set",
    )
    # Feel free to add more arguments here if you need!
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="fraction of the training data used for validation")
    parser.add_argument("--batch_size", type=int, default=32, help="mini-batch size for the MLP")
    parser.add_argument("--hidden_nodes", type=int, default=64, help="hidden layer size for the MLP")
    parser.add_argument("--weighted", action="store_true",
                        help="MLP classification with softmax + weighted cross-entropy (for class imbalance)")

    args = parser.parse_args()
    main(args)
