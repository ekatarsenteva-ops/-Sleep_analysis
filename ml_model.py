"""
ml_model.py — модуль машинного обучения.

Функции:
    - train_model()       : обучение классификатора RandomForest
    - evaluate_model()    : метрики качества (accuracy, confusion matrix,
                            classification report)
    - save_model()        : сохранение модели и энкодеров в файл .joblib
    - load_model()        : загрузка сохранённой модели
    - predict_single()    : прогноз для одной записи (для интерфейса)
    - predict_batch()     : прогноз для всего датасета
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
import joblib

from data_processing import prepare_features, get_feature_names, CLASS_LABELS

# ── Константы ──────────────────────────────────────────────────────────────────
MODEL_PATH = Path("sleep_model.joblib")

# Параметры RandomForestClassifier
RF_PARAMS = {
    "n_estimators": 100,      # количество деревьев
    "max_depth":    10,       # максимальная глубина дерева
    "min_samples_split": 4,   # минимум образцов для разбиения узла
    "random_state": 42,       # воспроизводимость результатов
    "class_weight": "balanced",  # компенсация лёгкого дисбаланса классов
    "n_jobs":       -1,       # использовать все доступные ядра CPU
}

TEST_SIZE   = 0.2   # доля тестовой выборки
RANDOM_SEED = 42


# ── Обучение ───────────────────────────────────────────────────────────────────

def train_model(
    csv_path: Path | None = None,
) -> tuple[RandomForestClassifier, dict, dict]:
    """
    Загружает данные, обучает классификатор RandomForestClassifier.

    Returns
    -------
    model    : обученная модель
    encoders : словарь OrdinalEncoder'ов (нужен для predict_single)
    metrics  : словарь с метриками качества на тестовой выборке
    """
    kwargs = {"csv_path": csv_path} if csv_path else {}
    X, y, encoders = prepare_features(**kwargs)

    # Разбиение на обучающую и тестовую выборки (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    # Обучение
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)

    # Метрики на тестовой выборке
    y_pred = model.predict(X_test)
    metrics = evaluate_model(model, X_test, y_test, y_pred)

    # Кросс-валидация (5 фолдов) на полной выборке — дополнительная проверка
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    metrics["cv_mean"]  = round(float(cv_scores.mean()), 4)
    metrics["cv_std"]   = round(float(cv_scores.std()), 4)

    print(f"[ml_model] Обучение завершено.")
    print(f"  Accuracy (test):     {metrics['accuracy']:.4f}")
    print(f"  CV accuracy (5-fold): {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")

    return model, encoders, metrics


# ── Оценка качества ────────────────────────────────────────────────────────────

def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray | None = None,
) -> dict:
    """
    Рассчитывает метрики качества классификации.

    Returns
    -------
    dict с ключами:
        accuracy        : доля правильных ответов (0–1)
        confusion_matrix: матрица ошибок (список списков)
        report          : текстовый отчёт (precision/recall/f1 по классам)
        report_dict     : то же, но в виде словаря (для Streamlit-таблицы)
        feature_importance: список (признак, важность) по убыванию
    """
    if y_pred is None:
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)
    report_text = classification_report(
        y_test, y_pred,
        target_names=[CLASS_LABELS[i] for i in range(3)],
    )
    report_dict = classification_report(
        y_test, y_pred,
        target_names=[CLASS_LABELS[i] for i in range(3)],
        output_dict=True,
    )

    # Важность признаков
    feature_names = get_feature_names()
    importances = model.feature_importances_
    fi = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True,
    )

    return {
        "accuracy":           round(float(acc), 4),
        "confusion_matrix":   cm.tolist(),
        "report":             report_text,
        "report_dict":        report_dict,
        "feature_importance": fi,
    }


# ── Сохранение / загрузка ──────────────────────────────────────────────────────

def save_model(
    model: RandomForestClassifier,
    encoders: dict,
    path: Path = MODEL_PATH,
) -> None:
    """Сохраняет модель и энкодеры в один .joblib-файл."""
    payload = {
        "model":         model,
        "encoders":      encoders,
        "feature_names": get_feature_names(),
        "class_labels":  CLASS_LABELS,
    }
    joblib.dump(payload, path)
    print(f"[ml_model] Модель сохранена: {path.resolve()}")


def load_model(path: Path = MODEL_PATH) -> dict:
    """
    Загружает модель из файла.

    Returns
    -------
    dict с ключами: model, encoders, feature_names, class_labels
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Файл модели не найден: {path.resolve()}\n"
            "Запустите train_model() и save_model() для создания модели."
        )
    payload = joblib.load(path)
    print(f"[ml_model] Модель загружена: {path.name}")
    return payload


# ── Прогноз ────────────────────────────────────────────────────────────────────

def predict_single(record: dict, payload: dict) -> dict:
    """
    Прогноз продуктивности для одной записи (ввод через интерфейс).

    Parameters
    ----------
    record  : dict с ключами, соответствующими FEATURE_COLS
              (числовые значения уже закодированы через encode_single)
    payload : dict, возвращённый load_model()

    Returns
    -------
    dict с ключами:
        productivity_class : int (0/1/2)
        productivity_label : str («Низкая»/«Средняя»/«Высокая»)
        confidence         : float (0.0–1.0) — вероятность предсказанного класса
        probabilities      : dict {label: prob} по всем трём классам
    """
    model         = payload["model"]
    feature_names = payload["feature_names"]
    class_labels  = payload["class_labels"]

    X = pd.DataFrame([record])[feature_names]
    pred_class  = int(model.predict(X)[0])
    proba       = model.predict_proba(X)[0]

    return {
        "productivity_class": pred_class,
        "productivity_label": class_labels[pred_class],
        "confidence":         round(float(proba[pred_class]), 4),
        "probabilities": {
            class_labels[i]: round(float(p), 4)
            for i, p in enumerate(proba)
        },
    }


def predict_batch(df: pd.DataFrame, payload: dict) -> pd.DataFrame:
    """
    Прогноз для всего датафрейма (используется для массовой аналитики).

    Parameters
    ----------
    df      : DataFrame с колонками из FEATURE_COLS
    payload : dict, возвращённый load_model()

    Returns
    -------
    df с добавленными колонками:
        productivity_class, productivity_label, confidence
    """
    model         = payload["model"]
    feature_names = payload["feature_names"]
    class_labels  = payload["class_labels"]

    X = df[feature_names]
    df = df.copy()
    df["productivity_class"] = model.predict(X)
    proba = model.predict_proba(X)
    df["confidence"] = [
        round(float(proba[i][cls]), 4)
        for i, cls in enumerate(df["productivity_class"])
    ]
    df["productivity_label"] = df["productivity_class"].map(class_labels)
    return df


# ── Быстрая проверка ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Обучаем
    model, encoders, metrics = train_model()

    # 2. Выводим метрики
    print("\n── Матрица ошибок ──")
    cm = np.array(metrics["confusion_matrix"])
    print(cm)
    print("\n── Classification Report ──")
    print(metrics["report"])
    print("── Важность признаков (топ-5) ──")
    for feat, imp in metrics["feature_importance"][:5]:
        print(f"  {feat:<22} {imp:.4f}")

    # 3. Сохраняем
    save_model(model, encoders)

    # 4. Загружаем и делаем тестовый прогноз
    payload = load_model()
    test_record = {
        "age":               30,
        "sleep_duration":    6.0,
        "sleep_quality":     5,
        "physical_activity": 30,
        "stress_level":      7,
        "heart_rate":        80,
        "daily_steps":       5000,
        "sleep_index":       0.625,  # 0.6*(5/10) + 0.4*(6/8)
        "gender_enc":        1,      # Male
        "bmi_enc":           2,      # Overweight
        "disorder_enc":      0,      # None
    }
    result = predict_single(test_record, payload)
    print(f"\n── Тестовый прогноз ──")
    print(f"  Класс:       {result['productivity_label']}")
    print(f"  Уверенность: {result['confidence']*100:.1f}%")
    print(f"  Вероятности: {result['probabilities']}")
