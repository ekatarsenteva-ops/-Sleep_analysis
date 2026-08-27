"""
data_processing.py — модуль обработки и подготовки данных.

Функции:
    - load_and_clean()        : загрузка CSV, очистка, переименование колонок
    - encode_features()       : кодирование категориальных признаков (OrdinalEncoder)
    - calc_sleep_index()      : расчёт интегрального индекса качества сна
    - build_target()          : построение целевой переменной (3 класса продуктивности)
    - prepare_features()      : полный pipeline: load → clean → encode → target
    - get_feature_names()     : список числовых признаков для модели
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder

CSV_PATH = Path("Sleep_health_and_lifestyle_dataset.csv")

# ── Константы ──────────────────────────────────────────────────────────────────

# Веса для расчёта sleep_index
WEIGHT_QUALITY  = 0.6   # качество сна (субъективная оценка 1–10)
WEIGHT_DURATION = 0.4   # продолжительность сна (нормирована к 8 ч)
OPTIMAL_SLEEP_H = 8.0   # эталонная продолжительность сна, часов

# Категориальные колонки и их допустимые значения
CAT_GENDER    = ["Male", "Female"]
CAT_BMI       = ["Normal", "Normal Weight", "Overweight", "Obese"]
CAT_DISORDER  = ["None", "Insomnia", "Sleep Apnea"]

# Метки классов продуктивности
CLASS_LABELS = {0: "Низкая", 1: "Средняя", 2: "Высокая"}

# Признаки, используемые ML-моделью
FEATURE_COLS = [
    "age",
    "sleep_duration",
    "sleep_quality",
    "physical_activity",
    "stress_level",
    "heart_rate",
    "daily_steps",
    "sleep_index",
    "gender_enc",
    "bmi_enc",
    "disorder_enc",
]


# ── Шаг 1: Загрузка и очистка ─────────────────────────────────────────────────

def load_and_clean(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """
    Загружает CSV, стандартизирует имена колонок, удаляет дубликаты и строки
    с критически пустыми полями.

    Returns
    -------
    DataFrame с колонками:
        person_id, gender, age, occupation, sleep_duration, sleep_quality,
        physical_activity, stress_level, bmi_category, blood_pressure,
        heart_rate, daily_steps, sleep_disorder
    """
    df = pd.read_csv(csv_path)

    # Стандартизация имён колонок
    rename_map = {
        "Person ID":              "person_id",
        "Gender":                 "gender",
        "Age":                    "age",
        "Occupation":             "occupation",
        "Sleep Duration":         "sleep_duration",
        "Quality of Sleep":       "sleep_quality",
        "Physical Activity Level":"physical_activity",
        "Stress Level":           "stress_level",
        "BMI Category":           "bmi_category",
        "Blood Pressure":         "blood_pressure",
        "Heart Rate":             "heart_rate",
        "Daily Steps":            "daily_steps",
        "Sleep Disorder":         "sleep_disorder",
    }
    df = df.rename(columns=rename_map)

    # Удаление полных дубликатов
    df = df.drop_duplicates()

    # Заполнение пропущенных значений в sleep_disorder
    df["sleep_disorder"] = df["sleep_disorder"].fillna("None")

    # Числовые колонки: принудительное приведение типов
    for col in ["age", "sleep_quality", "physical_activity", "stress_level",
                "heart_rate", "daily_steps"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Удаление строк с NaN в ключевых числовых полях
    required_cols = ["sleep_duration", "sleep_quality", "stress_level",
                     "physical_activity"]
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    # Ограничение допустимых диапазонов
    df = df[
        df["sleep_quality"].between(1, 10) &
        df["stress_level"].between(1, 10) &
        df["sleep_duration"].between(3, 12)
    ].reset_index(drop=True)

    return df


# ── Шаг 2: Расчёт sleep_index ─────────────────────────────────────────────────

def calc_sleep_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет колонку sleep_index — взвешенный нормированный индекс качества сна.

    Формула:
        sleep_index = WEIGHT_QUALITY  * (sleep_quality / 10)
                    + WEIGHT_DURATION * min(sleep_duration / OPTIMAL_SLEEP_H, 1.0)

    Диапазон: [0.0, 1.0].  Чем выше — тем лучше качество и достаточность сна.
    """
    df = df.copy()
    quality_norm   = df["sleep_quality"] / 10
    duration_norm  = (df["sleep_duration"] / OPTIMAL_SLEEP_H).clip(upper=1.0)
    df["sleep_index"] = (
        WEIGHT_QUALITY * quality_norm + WEIGHT_DURATION * duration_norm
    ).round(4)
    return df


# ── Шаг 3: Кодирование категориальных признаков ────────────────────────────────

def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Кодирует категориальные признаки через OrdinalEncoder.
    Возвращает (df_encoded, encoders) — датафрейм с новыми числовыми колонками
    и словарь с обученными энкодерами (для inverse_transform при необходимости).

    Новые колонки:
        gender_enc    : Male→1, Female→0
        bmi_enc       : Normal→0, Overweight→1, Obese→2
        disorder_enc  : None→0, Insomnia→1, Sleep Apnea→2
    """
    df = df.copy()
    encoders = {}

    gender_enc = OrdinalEncoder(categories=[CAT_GENDER])
    df["gender_enc"] = gender_enc.fit_transform(df[["gender"]]).astype(int)
    encoders["gender"] = gender_enc

    bmi_enc = OrdinalEncoder(categories=[CAT_BMI])
    df["bmi_enc"] = bmi_enc.fit_transform(df[["bmi_category"]]).astype(int)
    encoders["bmi"] = bmi_enc

    disorder_enc = OrdinalEncoder(categories=[CAT_DISORDER])
    df["disorder_enc"] = disorder_enc.fit_transform(
        df[["sleep_disorder"]]
    ).astype(int)
    encoders["disorder"] = disorder_enc

    return df, encoders


# ── Шаг 4: Построение целевой переменной ──────────────────────────────────────

def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создаёт целевую переменную productivity_class на основе sleep_index.

    Разбиение по квантилям (0.33 и 0.67):
        0 — низкая продуктивность  (нижняя треть sleep_index)
        1 — средняя продуктивность (средняя треть)
        2 — высокая продуктивность (верхняя треть)

    Квантильное разбиение обеспечивает равномерное распределение классов,
    что важно для обучения классификатора и корректной оценки метрик.
    """
    df = df.copy()
    q33 = df["sleep_index"].quantile(0.33)
    q67 = df["sleep_index"].quantile(0.67)

    conditions = [
        df["sleep_index"] <= q33,
        (df["sleep_index"] > q33) & (df["sleep_index"] <= q67),
        df["sleep_index"] > q67,
    ]
    df["productivity_class"] = np.select(conditions, [0, 1, 2], default=1)
    df["productivity_label"] = df["productivity_class"].map(CLASS_LABELS)

    return df


# ── Полный pipeline ────────────────────────────────────────────────────────────

def prepare_features(csv_path: Path = CSV_PATH) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Полный pipeline обработки данных.

    Returns
    -------
    X         : DataFrame с признаками (FEATURE_COLS)
    y         : Series с целевой переменной (productivity_class)
    encoders  : dict с обученными OrdinalEncoder'ами
    """
    df = load_and_clean(csv_path)
    df = calc_sleep_index(df)
    df, encoders = encode_features(df)
    df = build_target(df)

    X = df[FEATURE_COLS].copy()
    y = df["productivity_class"].copy()

    return X, y, encoders


def get_feature_names() -> list[str]:
    """Возвращает список признаков, используемых ML-моделью."""
    return FEATURE_COLS.copy()


# ── Вспомогательная функция для Streamlit-форм ────────────────────────────────

def encode_single(
    gender: str,
    bmi_category: str,
    sleep_disorder: str,
    encoders: dict,
) -> tuple[int, int, int]:
    """
    Кодирует категориальные признаки одной записи.
    Используется при ручном вводе данных через интерфейс системы.

    Returns
    -------
    (gender_enc, bmi_enc, disorder_enc)
    """
    g = encoders["gender"].transform([[gender]])[0][0]
    b = encoders["bmi"].transform([[bmi_category]])[0][0]
    d = encoders["disorder"].transform([[sleep_disorder]])[0][0]
    return int(g), int(b), int(d)


# ── Быстрая проверка ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    X, y, enc = prepare_features()
    print("Форма матрицы признаков X:", X.shape)
    print("Распределение классов:\n", y.value_counts().sort_index())
    print("\nПервые 3 строки X:")
    print(X.head(3).to_string())
    print("\nПервые 3 значения sleep_index:")
    df_check = load_and_clean()
    df_check = calc_sleep_index(df_check)
    print(df_check[["sleep_duration", "sleep_quality", "sleep_index"]].head(3))
