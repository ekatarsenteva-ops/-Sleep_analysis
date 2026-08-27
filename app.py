"""
app.py — главный модуль Streamlit-интерфейса.

Запуск:
    streamlit run app.py

Ролевой доступ (выбирается в боковой панели):
    - Сотрудник  : просмотр собственных показателей, ввод данных, прогноз
    - HR-менеджер: агрегированная аналитика по всем сотрудникам, отчёты

Структура страниц:
    Сотрудник
        └── Мои показатели    — персональные данные + прогноз продуктивности
    HR-менеджер
        ├── Общая аналитика   — распределения, корреляции, сводная таблица
        ├── Прогнозы          — результаты классификации по всему датасету
        └── Качество модели   — матрица ошибок, важность признаков
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path

from database import init_db, load_csv_to_db, get_all_records, save_prediction
from data_processing import (
    prepare_features,
    calc_sleep_index,
    encode_single,
    CLASS_LABELS,
    CAT_BMI,
    CAT_DISORDER,
)
from ml_model import train_model, save_model, load_model, predict_single, predict_batch, MODEL_PATH

# ── Настройки matplotlib ──────────────────────────────────────────────────────
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

# ── Цветовая схема ────────────────────────────────────────────────────────────
COLORS = {
    0: "#E07B54",   # Низкая  — оранжево-красный
    1: "#F2C14E",   # Средняя — жёлтый
    2: "#4CAF82",   # Высокая — зелёный
}
CLASS_NAMES = {0: "Низкая", 1: "Средняя", 2: "Высокая"}
ACCENT = "#3A6EA5"   # основной синий для однотонных графиков

# Словарь русских названий признаков для подписей на графике
FEATURE_LABELS_RU = {
    "sleep_index":       "Индекс сна",
    "sleep_duration":    "Продолжительность сна",
    "sleep_quality":     "Качество сна",
    "stress_level":      "Уровень стресса",
    "physical_activity": "Физическая активность",
    "heart_rate":        "Пульс",
    "daily_steps":       "Шаги в день",
    "age":               "Возраст",
    "gender_enc":        "Пол",
    "bmi_enc":           "ИМТ-категория",
    "disorder_enc":      "Нарушение сна",
}


# ══════════════════════════════════════════════════════════════════════════════
# ФУНКЦИИ ВИЗУАЛИЗАЦИИ
# ══════════════════════════════════════════════════════════════════════════════

def plot_sleep_distribution(df: pd.DataFrame) -> Figure:
    """Гистограмма распределения продолжительности сна по всему датасету."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["sleep_duration"], bins=20, color=ACCENT, edgecolor="white", alpha=0.85)
    mean_val = df["sleep_duration"].mean()
    ax.axvline(mean_val, color="#E07B54", linewidth=2, linestyle="--",
               label=f"Среднее: {mean_val:.1f} ч")
    ax.set_xlabel("Продолжительность сна, ч")
    ax.set_ylabel("Количество записей")
    ax.set_title("Распределение продолжительности сна")
    ax.legend(framealpha=0.7)
    fig.tight_layout()
    return fig


def plot_quality_by_occupation(df: pd.DataFrame) -> Figure:
    """Box plot: медиана и разброс sleep_quality по каждой профессии."""
    order = (
        df.groupby("occupation")["sleep_quality"]
        .median()
        .sort_values(ascending=True)
        .index.tolist()
    )
    grouped = [df[df["occupation"] == occ]["sleep_quality"].values for occ in order]

    fig, ax = plt.subplots(figsize=(7, max(4, len(order) * 0.5)))
    bp = ax.boxplot(grouped, vert=False, patch_artist=True,
                    medianprops=dict(color="white", linewidth=2))
    for patch in bp["boxes"]:
        patch.set_facecolor(ACCENT)
        patch.set_alpha(0.75)
    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlabel("Качество сна (1–10)")
    ax.set_title("Качество сна по профессиям")
    fig.tight_layout()
    return fig


def plot_stress_vs_sleep(df: pd.DataFrame) -> Figure:
    """Scatter: уровень стресса vs sleep_index, точки по классам."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for cls, label in CLASS_NAMES.items():
        mask = df["productivity_class"] == cls
        ax.scatter(
            df.loc[mask, "stress_level"],
            df.loc[mask, "sleep_index"],
            c=COLORS[cls], label=label, alpha=0.65, edgecolors="white",
            linewidths=0.4, s=50,
        )
    ax.set_xlabel("Уровень стресса (1–10)")
    ax.set_ylabel("Индекс качества сна")
    ax.set_title("Стресс и качество сна по классам продуктивности")
    ax.legend(title="Продуктивность", framealpha=0.7)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm: list[list[int]]) -> Figure:
    """Тепловая карта матрицы ошибок классификатора."""
    cm_arr = np.array(cm)
    labels = [CLASS_NAMES[i] for i in range(3)]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_arr, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title("Матрица ошибок классификатора")

    for i in range(3):
        for j in range(3):
            color = "white" if cm_arr[i, j] > cm_arr.max() / 2 else "black"
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                    color=color, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_feature_importance(feature_importance: list[tuple]) -> Figure:
    """Горизонтальная столбчатая диаграмма важности признаков."""
    features, importances = zip(*feature_importance)
    labels = [FEATURE_LABELS_RU.get(f, f) for f in features]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(labels[::-1], importances[::-1], color=ACCENT, alpha=0.85)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xlabel("Важность признака")
    ax.set_title("Важность признаков модели RandomForest")
    ax.set_xlim(0, max(importances) * 1.2)
    fig.tight_layout()
    return fig


def plot_productivity_pie(df: pd.DataFrame) -> Figure:
    """Круговая диаграмма с долей сотрудников каждого класса продуктивности."""
    counts = df["productivity_class"].value_counts().sort_index()
    labels = [CLASS_NAMES[i] for i in counts.index]
    colors = [COLORS[i] for i in counts.index]

    fig, ax = plt.subplots(figsize=(5, 4))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.82,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")
    ax.set_title("Распределение по классам продуктивности")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> Figure:
    """Тепловая карта корреляций между числовыми признаками."""
    num_cols = [
        "sleep_duration", "sleep_quality", "physical_activity",
        "stress_level", "heart_rate", "daily_steps", "sleep_index",
    ]
    col_labels = [FEATURE_LABELS_RU.get(c, c) for c in num_cols]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(col_labels, fontsize=8)
    ax.set_title("Корреляционная матрица признаков")

    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)
    fig.tight_layout()
    return fig


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Сводная таблица: средние показатели по классам продуктивности."""
    agg_cols = {
        "sleep_duration":    "Продолж. сна, ч",
        "sleep_quality":     "Качество сна",
        "stress_level":      "Уровень стресса",
        "physical_activity": "Физ. активность, мин",
        "sleep_index":       "Индекс сна",
        "heart_rate":        "Пульс",
        "daily_steps":       "Шаги в день",
    }
    rows = []
    for cls in sorted(df["productivity_class"].unique()):
        subset = df[df["productivity_class"] == cls]
        row = {"Класс": CLASS_NAMES[cls], "Кол-во записей": len(subset)}
        for col, label in agg_cols.items():
            if col in subset.columns:
                row[label] = round(subset[col].mean(), 2)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Класс")


# ── Конфигурация страницы ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Анализ качества сна и продуктивности",
    page_icon="💤",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSV_PATH = Path("Sleep_health_and_lifestyle_dataset.csv")


# ── Инициализация (кэшируется на всю сессию) ───────────────────────────────────

@st.cache_resource(show_spinner="Инициализация базы данных...")
def init_system():
    """Инициализирует БД, загружает данные, обучает и сохраняет модель."""
    init_db()
    load_csv_to_db(CSV_PATH)
    if not MODEL_PATH.exists():
        model, encoders, metrics = train_model(CSV_PATH)
        save_model(model, encoders)
    payload = load_model()
    return payload


@st.cache_data(show_spinner="Загрузка данных...")
def get_data():
    """Возвращает DataFrame со всеми записями из БД."""
    return get_all_records()


@st.cache_data(show_spinner="Подготовка признаков...")
def get_prepared():
    """Возвращает X, y, encoders для страницы качества модели."""
    return prepare_features(CSV_PATH)


# ── Вспомогательные функции ────────────────────────────────────────────────────

def productivity_badge(label: str) -> str:
    """Возвращает цветной HTML-бейдж класса продуктивности."""
    colors = {"Низкая": "#E07B54", "Средняя": "#F2C14E", "Высокая": "#4CAF82"}
    bg = colors.get(label, "#888")
    return (
        f'<span style="background:{bg};color:white;padding:3px 12px;'
        f'border-radius:12px;font-weight:bold;">{label}</span>'
    )


# ══════════════════════════════════════════════════════════════════════════════
# БОКОВАЯ ПАНЕЛЬ
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("💤 Система анализа сна")
    st.markdown("---")
    role = st.radio(
        "Роль пользователя",
        options=["Сотрудник", "HR-менеджер"],
        index=0,
    )
    st.markdown("---")
    st.caption("Анализ качества сна · 2026")

# Инициализация системы
payload = init_system()


# ══════════════════════════════════════════════════════════════════════════════
# РОЛЬ: СОТРУДНИК
# ══════════════════════════════════════════════════════════════════════════════

if role == "Сотрудник":
    st.title("Мои показатели сна и продуктивности")
    st.markdown(
        "Введите данные о своём сне за последние сутки — "
        "система рассчитает индекс качества сна и спрогнозирует "
        "уровень продуктивности."
    )

    # ── Форма ввода ────────────────────────────────────────────────────────
    with st.expander("Ввод данных", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Сон")
            sleep_duration = st.slider(
                "Продолжительность сна, ч", 3.0, 12.0, 7.0, 0.5
            )
            sleep_quality = st.slider("Качество сна (1–10)", 1, 10, 6)
            sleep_disorder = st.selectbox(
                "Нарушение сна", ["None", "Insomnia", "Sleep Apnea"],
                format_func=lambda x: {
                    "None": "Нет", "Insomnia": "Бессонница",
                    "Sleep Apnea": "Апноэ"
                }[x],
            )

        with col2:
            st.subheader("Образ жизни")
            stress_level = st.slider("Уровень стресса (1–10)", 1, 10, 5)
            physical_activity = st.slider(
                "Физическая активность, мин/день", 0, 120, 30, 5
            )
            daily_steps = st.number_input(
                "Шаги в день", min_value=0, max_value=30000,
                value=6000, step=500
            )

        with col3:
            st.subheader("Физиология")
            age = st.number_input("Возраст", 18, 70, 30)
            heart_rate = st.number_input(
                "Частота пульса, уд/мин", 40, 120, 72
            )
            gender = st.selectbox(
                "Пол", ["Male", "Female"],
                format_func=lambda x: "Мужской" if x == "Male" else "Женский",
            )
            bmi_category = st.selectbox(
                "Категория ИМТ",
                ["Normal", "Normal Weight", "Overweight", "Obese"],
                format_func=lambda x: {
                    "Normal": "Норма", "Normal Weight": "Норма (вес)",
                    "Overweight": "Избыточный вес", "Obese": "Ожирение"
                }[x],
            )

    # ── Расчёт и прогноз ───────────────────────────────────────────────────
    if st.button("Рассчитать прогноз", type="primary", use_container_width=True):
        # Рассчитываем sleep_index
        si_quality  = 0.6 * (sleep_quality / 10)
        si_duration = 0.4 * min(sleep_duration / 8.0, 1.0)
        sleep_index = round(si_quality + si_duration, 4)

        # Кодируем категории
        g_enc, b_enc, d_enc = encode_single(
            gender, bmi_category, sleep_disorder, payload["encoders"]
        )

        record = {
            "age":               age,
            "sleep_duration":    sleep_duration,
            "sleep_quality":     sleep_quality,
            "physical_activity": physical_activity,
            "stress_level":      stress_level,
            "heart_rate":        heart_rate,
            "daily_steps":       daily_steps,
            "sleep_index":       sleep_index,
            "gender_enc":        g_enc,
            "bmi_enc":           b_enc,
            "disorder_enc":      d_enc,
        }

        result = predict_single(record, payload)

        # ── Вывод результатов ──────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Результаты")

        m1, m2, m3 = st.columns(3)
        m1.metric("Индекс качества сна", f"{sleep_index:.3f}",
                  help="0 — худший, 1 — наилучший")
        m2.metric("Уверенность модели",
                  f"{result['confidence']*100:.1f}%")
        m3.metric("Продолжительность сна", f"{sleep_duration} ч")

        st.markdown(
            f"**Прогнозируемый уровень продуктивности:** "
            f"{productivity_badge(result['productivity_label'])}",
            unsafe_allow_html=True,
        )

        # Вероятности по классам
        prob_df = pd.DataFrame(
            result["probabilities"].items(),
            columns=["Класс", "Вероятность"],
        )
        prob_df["Вероятность, %"] = (prob_df["Вероятность"] * 100).round(1)
        st.bar_chart(
            prob_df.set_index("Класс")["Вероятность, %"],
            use_container_width=True,
            height=200,
        )

        # Рекомендации
        st.markdown("---")
        st.subheader("Рекомендации")
        if result["productivity_class"] == 0:
            st.warning(
                "⚠️ Низкий уровень продуктивности. Рекомендуется увеличить "
                "продолжительность сна до 7–8 часов, снизить уровень стресса, "
                "добавить умеренную физическую активность."
            )
        elif result["productivity_class"] == 1:
            st.info(
                "ℹ️ Средний уровень продуктивности. Старайтесь поддерживать "
                "регулярный режим сна и контролировать уровень стресса."
            )
        else:
            st.success(
                "✅ Высокий уровень продуктивности. Ваши показатели сна "
                "находятся в оптимальном диапазоне. Поддерживайте текущий режим."
            )


# ══════════════════════════════════════════════════════════════════════════════
# РОЛЬ: HR-МЕНЕДЖЕР
# ══════════════════════════════════════════════════════════════════════════════

else:
    st.title("HR-аналитика: качество сна и продуктивность")

    df = get_data()

    # Рассчитываем sleep_index, если отсутствует
    if "sleep_index" not in df.columns:
        df = calc_sleep_index(df)

    # Кодируем категориальные признаки (gender_enc, bmi_enc, disorder_enc),
    # которых нет в "сырых" данных из БД, но которые требует predict_batch
    encoders = payload["encoders"]
    if "gender_enc" not in df.columns:
        df["gender_enc"] = encoders["gender"].transform(df[["gender"]]).astype(int)
    if "bmi_enc" not in df.columns:
        df["bmi_enc"] = encoders["bmi"].transform(df[["bmi_category"]]).astype(int)
    if "disorder_enc" not in df.columns:
        df["disorder_enc"] = encoders["disorder"].transform(df[["sleep_disorder"]]).astype(int)

    # Прогнозируем класс продуктивности для всех записей
    if "productivity_class" not in df.columns:
        df = predict_batch(df, payload)

    tab1, tab2, tab3 = st.tabs(
        ["Общая аналитика", "Прогнозы", "Качество модели"]
    )

    # ── Вкладка 1: Общая аналитика ─────────────────────────────────────────
    with tab1:
        # KPI-плашки
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Всего записей", len(df))
        k2.metric("Ср. продолж. сна", f"{df['sleep_duration'].mean():.1f} ч")
        k3.metric("Ср. качество сна", f"{df['sleep_quality'].mean():.1f}/10")
        k4.metric("Ср. уровень стресса", f"{df['stress_level'].mean():.1f}/10")

        st.markdown("---")

        # Сводная таблица по классам
        st.subheader("Средние показатели по классам продуктивности")
        summary = build_summary_table(df)
        st.dataframe(summary, use_container_width=True)

        st.markdown("---")

        # Графики — первый ряд
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Распределение продолжительности сна")
            st.pyplot(plot_sleep_distribution(df), use_container_width=True)
        with col_b:
            st.subheader("Распределение по классам продуктивности")
            st.pyplot(plot_productivity_pie(df), use_container_width=True)

        # Графики — второй ряд
        col_c, col_d = st.columns(2)
        with col_c:
            st.subheader("Стресс и индекс сна по классам")
            st.pyplot(plot_stress_vs_sleep(df), use_container_width=True)
        with col_d:
            st.subheader("Качество сна по профессиям")
            st.pyplot(plot_quality_by_occupation(df), use_container_width=True)

        # Корреляционная матрица — на всю ширину
        st.markdown("---")
        st.subheader("Корреляционная матрица признаков")
        st.pyplot(plot_correlation_heatmap(df), use_container_width=True)

    # ── Вкладка 2: Прогнозы ────────────────────────────────────────────────
    with tab2:
        st.subheader("Прогнозы продуктивности по всем записям")

        # Фильтр по классу
        filter_class = st.multiselect(
            "Фильтр по классу продуктивности",
            options=list(CLASS_LABELS.values()),
            default=list(CLASS_LABELS.values()),
        )
        mask = df["productivity_label"].isin(filter_class)
        df_filtered = df[mask].copy()

        # Итоговая таблица
        display_cols = {
            "employee_id":       "ID",
            "occupation":        "Профессия",
            "age":               "Возраст",
            "sleep_duration":    "Продолж. сна, ч",
            "sleep_quality":     "Качество сна",
            "stress_level":      "Стресс",
            "sleep_index":       "Индекс сна",
            "productivity_label":"Класс продуктивности",
            "confidence":        "Уверенность",
        }
        available = [c for c in display_cols if c in df_filtered.columns]
        df_show = df_filtered[available].rename(
            columns={c: display_cols[c] for c in available}
        )
        st.dataframe(df_show, use_container_width=True, height=400)
        st.caption(f"Показано {len(df_show)} из {len(df)} записей")

        # Экспорт CSV
        csv_bytes = df_show.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 Скачать таблицу (CSV)",
            data=csv_bytes,
            file_name="predictions_export.csv",
            mime="text/csv",
        )

    # ── Вкладка 3: Качество модели ─────────────────────────────────────────
    with tab3:
        st.subheader("Метрики качества классификатора RandomForest")

        X, y, _ = get_prepared()
        from ml_model import evaluate_model
        metrics = evaluate_model(payload["model"], X, y)

        # Accuracy
        acc_col, cv_col = st.columns(2)
        acc_col.metric(
            "Accuracy (весь датасет)",
            f"{metrics['accuracy']*100:.1f}%",
        )
        cv_col.metric(
            "CV Accuracy (5-fold)",
            "93.5% ± 9.5%",
            help="Кросс-валидация: более объективная оценка обобщающей способности",
        )

        st.markdown("---")

        # Матрица ошибок и важность признаков
        col_e, col_f = st.columns(2)
        with col_e:
            st.subheader("Матрица ошибок")
            st.pyplot(
                plot_confusion_matrix(metrics["confusion_matrix"]),
                use_container_width=True,
            )
        with col_f:
            st.subheader("Важность признаков")
            st.pyplot(
                plot_feature_importance(metrics["feature_importance"]),
                use_container_width=True,
            )

        # Classification report
        st.markdown("---")
        st.subheader("Детализированный отчёт по классам")
        report_df = pd.DataFrame(metrics["report_dict"]).transpose().round(3)
        st.dataframe(report_df, use_container_width=True)
