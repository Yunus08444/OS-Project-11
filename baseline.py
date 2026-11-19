import time

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def train_baseline(X_train, X_test, y_train, y_test):

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000))
    ])

    start_time = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start_time

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n=== BASELINE MODEL ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Training time: {training_time:.6f} seconds")
    print(f"Number of features: {X_train.shape[1]}")

    return {
        "name": "Baseline",
        "accuracy": accuracy,
        "f1": f1,
        "time": training_time,
        "features": X_train.shape[1]
    }


