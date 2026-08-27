"""
database.py — модуль работы с базой данных SQLite.

Функции:
    - init_db()         : создание таблиц (DDL)
    - load_csv_to_db()  : загрузка датасета из CSV в таблицы employees / sleep_records /
                          lifestyle_metrics
    - get_all_records() : выборка всех записей с JOIN для аналитики
    - save_prediction() : сохранение результата ML-прогноза в таблицу predictions
"""

import sqlite3
import pandas as pd
from pathlib import Path
from datetime import date

# ── Конфигурация ──────────────────────────────────────────────────────────────
DB_PATH = Path("sleep_system.db")
CSV_PATH = Path("Sleep_health_and_lifestyle_dataset.csv")

# ── DDL ────────────────────────────────────────────────────────────────────────

DDL = """
CREATE TABLE IF NOT EXISTS employees (
    employee_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    gender        TEXT    NOT NULL,
    age           INTEGER NOT NULL,
    occupation    TEXT    NOT NULL,
    bmi_category  TEXT    NOT NULL,
    blood_pressure TEXT,
    heart_rate    INTEGER NOT NULL,
    daily_steps   INTEGER NOT NULL,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sleep_records (
    record_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id    INTEGER NOT NULL,
    sleep_duration REAL    NOT NULL,
    sleep_quality  INTEGER NOT NULL CHECK(sleep_quality BETWEEN 1 AND 10),
    sleep_disorder TEXT    NOT NULL DEFAULT 'None',
    record_date    DATE    NOT NULL DEFAULT CURRENT_DATE,
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
);

CREATE TABLE IF NOT EXISTS lifestyle_metrics (
    metric_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id       INTEGER NOT NULL,
    physical_activity INTEGER NOT NULL,
    stress_level      INTEGER NOT NULL CHECK(stress_level BETWEEN 1 AND 10),
    sleep_index       REAL    NOT NULL,
    metric_date       DATE    NOT NULL DEFAULT CURRENT_DATE,
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id       INTEGER NOT NULL,
    productivity_class INTEGER NOT NULL CHECK(productivity_class IN (0, 1, 2)),
    confidence_score  REAL    NOT NULL,
    predicted_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
);

CREATE INDEX IF NOT EXISTS idx_sleep_records_emp   ON sleep_records(employee_id);
CREATE INDEX IF NOT EXISTS idx_lifestyle_emp        ON lifestyle_metrics(employee_id);
CREATE INDEX IF NOT EXISTS idx_predictions_emp      ON predictions(employee_id);
"""


def get_connection() -> sqlite3.Connection:
    """Возвращает соединение с БД с включёнными внешними ключами."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Создаёт таблицы и индексы, если они ещё не существуют."""
    with get_connection() as conn:
        conn.executescript(DDL)
    print(f"[database] БД инициализирована: {DB_PATH.resolve()}")


# ── Загрузка CSV ───────────────────────────────────────────────────────────────

def load_csv_to_db(csv_path: Path = CSV_PATH, replace: bool = False) -> int:
    """
    Загружает датасет из CSV-файла в три таблицы: employees, sleep_records,
    lifestyle_metrics.  Возвращает количество загруженных записей.

    Parameters
    ----------
    csv_path : Path  — путь к файлу Sleep_health_and_lifestyle_dataset.csv
    replace  : bool  — если True, перезаписывает существующие данные
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Файл не найден: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)

    # ── Переименование колонок в snake_case ─────────────────────────────
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # ── Расчёт sleep_index (вес: качество 0.6, продолжительность 0.4) ───
    df["sleep_index"] = (
        0.6 * (df["quality_of_sleep"] / 10)
        + 0.4 * df["sleep_duration"].clip(upper=8).div(8)
    ).round(4)

    today = str(date.today())

    with get_connection() as conn:
        if replace:
            conn.executescript("""
                DELETE FROM predictions;
                DELETE FROM lifestyle_metrics;
                DELETE FROM sleep_records;
                DELETE FROM employees;
            """)

        # Проверяем, есть ли уже данные
        existing = conn.execute("SELECT COUNT(*) FROM employees").fetchone()[0]
        if existing > 0 and not replace:
            print(f"[database] Данные уже загружены ({existing} записей). "
                  "Передайте replace=True для перезаписи.")
            return existing

        loaded = 0
        for _, row in df.iterrows():
            # 1. employees
            cur = conn.execute(
                """INSERT INTO employees
                   (gender, age, occupation, bmi_category,
                    blood_pressure, heart_rate, daily_steps)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    row["gender"],
                    int(row["age"]),
                    row["occupation"],
                    row["bmi_category"],
                    row.get("blood_pressure", None),
                    int(row["heart_rate"]),
                    int(row["daily_steps"]),
                ),
            )
            emp_id = cur.lastrowid

            # 2. sleep_records
            conn.execute(
                """INSERT INTO sleep_records
                   (employee_id, sleep_duration, sleep_quality,
                    sleep_disorder, record_date)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    emp_id,
                    float(row["sleep_duration"]),
                    int(row["quality_of_sleep"]),
                    str(row["sleep_disorder"]) if pd.notna(row["sleep_disorder"]) else "None",
                    today,
                ),
            )

            # 3. lifestyle_metrics
            conn.execute(
                """INSERT INTO lifestyle_metrics
                   (employee_id, physical_activity, stress_level,
                    sleep_index, metric_date)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    emp_id,
                    int(row["physical_activity_level"]),
                    int(row["stress_level"]),
                    float(row["sleep_index"]),
                    today,
                ),
            )
            loaded += 1

        conn.commit()

    print(f"[database] Загружено {loaded} записей из {csv_path.name}.")
    return loaded


# ── Выборки ────────────────────────────────────────────────────────────────────

def get_all_records() -> pd.DataFrame:
    """
    Возвращает DataFrame со всеми записями (JOIN трёх таблиц).
    Используется для аналитики и обучения ML-модели.
    """
    sql = """
        SELECT
            e.employee_id,
            e.gender,
            e.age,
            e.occupation,
            e.bmi_category,
            e.heart_rate,
            e.daily_steps,
            sr.sleep_duration,
            sr.sleep_quality,
            sr.sleep_disorder,
            lm.physical_activity,
            lm.stress_level,
            lm.sleep_index
        FROM employees e
        JOIN sleep_records   sr ON sr.employee_id = e.employee_id
        JOIN lifestyle_metrics lm ON lm.employee_id = e.employee_id
        ORDER BY e.employee_id
    """
    with get_connection() as conn:
        return pd.read_sql_query(sql, conn)


def save_prediction(
    employee_id: int,
    productivity_class: int,
    confidence_score: float,
) -> int:
    """Сохраняет прогноз в таблицу predictions. Возвращает prediction_id."""
    sql = """
        INSERT INTO predictions
            (employee_id, productivity_class, confidence_score)
        VALUES (?, ?, ?)
    """
    with get_connection() as conn:
        cur = conn.execute(sql, (employee_id, productivity_class,
                                 confidence_score))
        conn.commit()
        return cur.lastrowid


# ── Быстрая проверка ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    n = load_csv_to_db()
    df = get_all_records()
    print(df.head())
    print(f"\nВсего записей в БД: {len(df)}")
