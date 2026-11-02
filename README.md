# ML from Scratch — Predicting Gaming Addiction

Classic machine-learning models implemented **from scratch in NumPy** (no scikit-learn) and benchmarked on a gaming & mental-health dataset, on two tasks:

- **Classification** — predict a player's addiction level
- **Regression** — predict a player's addiction score

## Models

| Model | Task | Implementation |
|---|---|---|
| k-Nearest Neighbors | classification & regression | vectorized Euclidean distances, majority vote / mean of the k neighbors |
| Logistic Regression | classification | multi-class softmax, cross-entropy loss, batch gradient descent, numerically stable softmax |
| Linear Regression | regression | closed-form least squares (pseudo-inverse) with bias term |

## Pipeline

1. **Standardization** of every feature, with mean and standard deviation fitted on the training split only (no data leakage)
2. **80/20 train/validation split** for model selection; the test set is used once, for the final comparison
3. **Hyperparameter search** — K ∈ {1, 3, 5, 7, 11, 15} for KNN; learning rate ∈ {0.001, 0.01, 0.05} and iterations ∈ {500, 1000} for logistic regression
4. **Evaluation** — accuracy and macro-F1 (classification), MSE (regression), plus training and inference time

## Results (test set)

| Task | Model | Score |
|---|---|---|
| Classification | KNN (K = 3) | 76.5 % accuracy · 0.552 macro-F1 |
| Classification | **Logistic Regression** (lr = 0.05, 1000 iters) | **85.8 % accuracy · 0.597 macro-F1** |
| Regression | KNN (K = 11) | MSE 1.880 |
| Regression | **Linear Regression** | **MSE 0.994** |

**Takeaways**

- Small K overfits: near-perfect training accuracy but weaker validation scores; validation MSE stabilizes around K = 11.
- Accuracy alone is misleading on imbalanced classes — macro-F1 shows the per-class picture.
- KNN is the most expensive model at inference (each prediction scans the whole training set); linear models predict in microseconds.

## Usage

```bash
pip install -r requirements.txt

# Validation run (80/20 split of the training data)
python main.py --task classification --method knn --K 3
python main.py --task classification --method logistic_regression --lr 0.05 --max_iters 1000

# Final evaluation on the test set
python main.py --task regression --method linear_regression --test
```

The dataset is expected at `data/features.npz` with keys `xtrain`, `xtest`, `ytrainreg`, `ytestreg`, `ytrainclassif`, `ytestclassif`.

## Project structure

```text
main.py                     # CLI: data loading, split, standardization, training, evaluation
src/utils.py                # One-hot encoding, normalization, accuracy, macro-F1, MSE
src/methods/knn.py
src/methods/logistic_regression.py
src/methods/linear_regression.py
```

## Credits

Built with two teammates.
