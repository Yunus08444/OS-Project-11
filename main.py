from src.data_loader import load_data
from src.baseline import train_baseline


def main():
    X_train, X_test, y_train, y_test = load_data()

    baseline_results = train_baseline(
        X_train,
        X_test,
        y_train,
        y_test
    )


if __name__ == "__main__":
    main()

