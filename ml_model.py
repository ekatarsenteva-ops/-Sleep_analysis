"""
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

MODEL_PATH = Path("sleep_model.joblib")
RF_PARAMS = {
    "n_estimators": 100,      
    "max_depth":    10,       
    "min_samples_split": 4,   
    "random_state": 42,      
    "class_weight": "balanced",  
    "n_jobs":       -1,       
}

TEST_SIZE   = 0.2  
RANDOM_SEED = 42

def train_model(
    csv_path: Path | None = None,
) -> tuple[RandomForestClassifier, dict, dict]:
   
    kwargs = {"csv_path": csv_path} if csv_path else {}
    X, y, encoders = prepare_features(**kwargs)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = evaluate_model(model, X_test, y_test, y_pred)

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    metrics["cv_mean"]  = round(float(cv_scores.mean()), 4)
    metrics["cv_std"]   = round(float(cv_scores.std()), 4)

    print(f"[ml_model] Обучение завершено.")
    print(f"  Accuracy (test):     {metrics['accuracy']:.4f}")
    print(f"  CV accuracy (5-fold): {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")

    return model, encoders, metrics

def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray | None = None,
) -> dict:
    
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

def save_model(
    model: RandomForestClassifier,
    encoders: dict,
    path: Path = MODEL_PATH,
) -> None:
 
    payload = {
        "model":         model,
        "encoders":      encoders,
        "feature_names": get_feature_names(),
        "class_labels":  CLASS_LABELS,
    }
    joblib.dump(payload, path)
    print(f"[ml_model] Модель сохранена: {path.resolve()}")


def load_model(path: Path = MODEL_PATH) -> dict:
   
    if not path.exists():
        raise FileNotFoundError(
            f"Файл модели не найден: {path.resolve()}\n"
            "Запустите train_model() и save_model() для создания модели."
        )
    payload = joblib.load(path)
    print(f"[ml_model] Модель загружена: {path.name}")
    return payload

def predict_single(record: dict, payload: dict) -> dict:

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

if __name__ == "__main__":
    model, encoders, metrics = train_model()

    print("\n── Матрица ошибок ──")
    cm = np.array(metrics["confusion_matrix"])
    print(cm)
    print("\n── Classification Report ──")
    print(metrics["report"])
    print("── Важность признаков (топ-5) ──")
    for feat, imp in metrics["feature_importance"][:5]:
        print(f"  {feat:<22} {imp:.4f}")

    save_model(model, encoders)

    payload = load_model()
    test_record = {
        "age":               30,
        "sleep_duration":    6.0,
        "sleep_quality":     5,
        "physical_activity": 30,
        "stress_level":      7,
        "heart_rate":        80,
        "daily_steps":       5000,
        "sleep_index":       0.625,  
        "gender_enc":        1,      
        "bmi_enc":           2,      
        "disorder_enc":      0,      
    }
    result = predict_single(test_record, payload)
    print(f"\n── Тестовый прогноз ──")
    print(f"  Класс:       {result['productivity_label']}")
    print(f"  Уверенность: {result['confidence']*100:.1f}%")
    print(f"  Вероятности: {result['probabilities']}")
