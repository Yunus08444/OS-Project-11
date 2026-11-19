from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_data():
    # Загружаем датасет
    dataset = load_breast_cancer()

    X = dataset.data
    y = dataset.target

    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Данные успешно загружены!")
    print(f"Всего объектов: {X.shape[0]}")
    print(f"Количество признаков: {X.shape[1]}")
    print(f"Обучающая выборка: {X_train.shape}")
    print(f"Тестовая выборка: {X_test.shape}")

    return X_train, X_test, y_train, y_test

