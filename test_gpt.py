import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
import pyarrow.parquet as pq
from pathlib import Path
import plotly.express as px
from datetime import timedelta

st.set_page_config(layout="wide", page_title="Fraud vs Non-Fraud — Grid Layout + Filter")

st.title("Мошен. (красный) и не-мошен. (синий) — 2 графика: города и vendor_type")

default_path = "d:/Work/ITMO_CONTEST/transaction_fraud_data.parquet"
file_path = st.sidebar.text_input("Путь к Parquet:", value=default_path)

agg_by = st.sidebar.selectbox("Агрегировать по", ["count", "amount"], index=0)
top_n = st.sidebar.number_input("Top N городов:", min_value=10, max_value=2000, value=200, step=10)
scale = st.sidebar.slider("Масштаб радиусов:", 0.2, 10.0, 1.0, 0.1)

if not Path(file_path).exists():
    st.error("Файл не найден.")
    st.stop()

@st.cache_data
def load_time_bounds(parquet_path):
    table = pq.read_table(parquet_path, columns=["timestamp"])
    df = table.to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df["timestamp"].min().to_pydatetime(), df["timestamp"].max().to_pydatetime()

with st.spinner("Сканирую даты..."):
    t_min, t_max = load_time_bounds(file_path)

time_range = st.slider(
    "Фильтр по времени (timestamp)",
    min_value=t_min,
    max_value=t_max,
    value=(t_min, t_max),
    format="YYYY-MM-DD HH:mm:ss",
     step=timedelta(hours=1)
)

@st.cache_data
def read_data(parquet_path, columns):
    table = pq.read_table(parquet_path, columns=columns)
    df = table.to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

# Считаем исходный датафрейм с нужными колонками
cols_needed = ["city", "is_fraud", "timestamp", "vendor_type", "amount"]

with st.spinner("Читаю данные..."):
    df_full = read_data(file_path, cols_needed)

# Фильтруем по времени
df_filtered = df_full[(df_full["timestamp"] >= time_range[0]) & (df_full["timestamp"] <= time_range[1])]

# --- Сохраняем/инициализируем выбранные vendor_type ---
all_vendor_types = sorted(df_filtered["vendor_type"].dropna().unique())
if "selected_vendor_types" not in st.session_state:
    st.session_state.selected_vendor_types = all_vendor_types.copy()

selected_vendor_types = st.sidebar.multiselect(
    "Выберите vendor_type для отображения:",
    options=all_vendor_types,
    default=st.session_state.selected_vendor_types,
    key="selected_vendor_types"
)

# Фильтруем по выбранным vendor_type
df_filtered = df_filtered[df_filtered["vendor_type"].isin(selected_vendor_types)]

# --- Агрегируем по городам ---
def aggregate_by_city(df, agg_by, top_n):
    if agg_by == "count":
        agg = df.groupby(["city", "is_fraud"]).size().reset_index(name="metric")
    else:
        agg = df.groupby(["city", "is_fraud"])["amount"].sum().reset_index(name="metric")

    pivot = agg.pivot_table(index="city", columns="is_fraud", values="metric", fill_value=0).reset_index()
    if True not in pivot.columns:
        pivot[True] = 0
    if False not in pivot.columns:
        pivot[False] = 0

    pivot = pivot.rename(columns={False: "non_fraud", True: "fraud"})
    pivot["total"] = pivot["fraud"] + pivot["non_fraud"]
    pivot = pivot.sort_values("total", ascending=False).head(top_n).reset_index(drop=True)
    return pivot

pivot_city = aggregate_by_city(df_filtered, agg_by, top_n)

# --- Инициализация и выбор город ---
all_cities = pivot_city["city"].tolist()
if "selected_cities" not in st.session_state:
    st.session_state.selected_cities = all_cities.copy()

selected_cities = st.sidebar.multiselect(
    "Выберите города для отображения:",
    options=all_cities,
    default=st.session_state.selected_cities,
    key="selected_cities"
)

pivot_city = pivot_city[pivot_city["city"].isin(selected_cities)].reset_index(drop=True)

# --- Агрегируем по vendor_type ---
def aggregate_by_vendor(df, agg_by):
    if agg_by == "count":
        agg = df.groupby(["vendor_type", "is_fraud"]).size().reset_index(name="metric")
    else:
        agg = df.groupby(["vendor_type", "is_fraud"])["amount"].sum().reset_index(name="metric")

    pivot = agg.pivot_table(index="vendor_type", columns="is_fraud", values="metric", fill_value=0).reset_index()
    if True not in pivot.columns:
        pivot[True] = 0
    if False not in pivot.columns:
        pivot[False] = 0

    pivot = pivot.rename(columns={False: "non_fraud", True: "fraud"})
    pivot["total"] = pivot["fraud"] + pivot["non_fraud"]
    pivot = pivot.sort_values("total", ascending=False).reset_index(drop=True)
    return pivot

pivot_vendor = aggregate_by_vendor(df_filtered, agg_by)

# --- Функция для построения позиций в сетке ---
def compute_grid_positions(n):
    if n == 0:
        return np.empty((0, 2))
    rows = math.ceil(math.sqrt(n))
    cols = math.ceil(n / rows)
    xs, ys = [], []
    for i in range(n):
        row = i // cols
        col = i % cols
        xs.append(col / (cols - 1 if cols > 1 else 1))
        ys.append(1 - row / (rows - 1 if rows > 1 else 1))
    return np.column_stack([xs, ys])


def build_bar_figure(pivot_df, title):
    if pivot_df.empty:
        return go.Figure()

    # Преобразуем pivot_df для plotly: город/vendor_type, fraud, non_fraud
    df_plot = pivot_df.melt(id_vars=[pivot_df.columns[0]], value_vars=["fraud", "non_fraud"],
                            var_name="type", value_name="count")
    fig = px.bar(
        df_plot,
        x=pivot_df.columns[0],  # 'city' или 'vendor_type'
        y="count",
        color="type",
        color_discrete_map={"fraud": "red", "non_fraud": "blue"},
        barmode="group",
        title=title,
        labels={pivot_df.columns[0]: pivot_df.columns[0], "count": "Количество транзакций", "type": "Тип"}
    )
    fig.update_layout(xaxis_tickangle=-45, height=600, margin=dict(t=40, b=150))
    return fig

# # --- Функция для построения фигуры по данным ---
# def build_figure(pivot_df, scale):
#     n = len(pivot_df)
#     if n == 0:
#         return go.Figure()

#     max_total = pivot_df["total"].max() if n > 0 else 1
#     base_scale = 0.08
#     scale_factor = base_scale * scale / math.sqrt(max_total + 1e-9)

#     nodes = []
#     for _, r in pivot_df.iterrows():
#         rf = max(math.sqrt(r["fraud"]) * scale_factor, 0.005)
#         rn = max(math.sqrt(r["non_fraud"]) * scale_factor, 0.005)
#         nodes.append({
#             "label": r.iloc[0],  # city или vendor_type
#             "fraud": r["fraud"],
#             "non_fraud": r["non_fraud"],
#             "r_fraud": rf,
#             "r_non_fraud": rn
#         })

#     positions = compute_grid_positions(n)

#     hover_texts = [
#         f"{node['label']}<br>Fraud: {node['fraud']:,}<br>Non-fraud: {node['non_fraud']:,}"
#         for node in nodes
#     ]

#     fig = go.Figure()

#     fig.add_trace(go.Scatter(
#         x=positions[:, 0],
#         y=positions[:, 1],
#         mode="markers",
#         marker=dict(
#             size=np.array([n["r_non_fraud"] for n in nodes]) * 500,
#             color="blue",
#             opacity=0.3,
#             line=dict(width=1, color="white")
#         ),
#         text=hover_texts,
#         hoverinfo="text",
#         name="Non-fraud"
#     ))

#     fig.add_trace(go.Scatter(
#         x=positions[:, 0],
#         y=positions[:, 1],
#         mode="markers",
#         marker=dict(
#             size=np.array([n["r_fraud"] for n in nodes]) * 500,
#             color="red",
#             opacity=0.5,
#             line=dict(width=1, color="white")
#         ),
#         text=hover_texts,
#         hoverinfo="text",
#         name="Fraud"
#     ))

#     fig.update_layout(
#         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#         plot_bgcolor="white",
#         height=600,
#         margin=dict(l=20, r=20, t=30, b=20)
#     )
#     return fig

import plotly.graph_objects as go

# --- Рисуем графики ---

st.sidebar.markdown("---")

st.markdown("### График по городам")
fig_city = build_bar_figure(pivot_city, scale)
st.plotly_chart(fig_city, use_container_width=True)

st.markdown("### График по vendor_type")
fig_vendor = build_bar_figure(pivot_vendor, scale)
st.plotly_chart(fig_vendor, use_container_width=True)

import plotly.express as px

st.sidebar.markdown("---")
st.sidebar.markdown("### Почасовой график транзакций по vendor_type")

# Выбор одного vendor_type для почасового анализа
vendor_for_hourly = st.sidebar.selectbox(
    "Выберите vendor_type для почасового графика",
    options=all_vendor_types
)

if st.sidebar.button("Показать почасовой график"):
    # Фильтруем данные по выбранному vendor_type
    df_v = df_filtered[df_filtered["vendor_type"] == vendor_for_hourly].copy()
    if df_v.empty:
        st.warning(f"Нет данных для vendor_type = {vendor_for_hourly} в выбранном времени.")
    else:
        # Округляем timestamp до часа
        df_v["hour"] = df_v["timestamp"].dt.floor("H")

        if agg_by == "count":
            hourly_agg = df_v.groupby(["hour", "is_fraud"]).size().reset_index(name="metric")
        else:
            # Для этого в колонках должен быть amount — если его нет, нужно добавить в cols_needed
            hourly_agg = df_v.groupby(["hour", "is_fraud"])["amount"].sum().reset_index(name="metric")

        # Pivot для удобства построения графика
        pivot_hourly = hourly_agg.pivot(index="hour", columns="is_fraud", values="metric").fillna(0)
        pivot_hourly.columns = ["non_fraud", "fraud"]  # False -> non_fraud, True -> fraud

        fig_hourly = px.line(
            pivot_hourly,
            x=pivot_hourly.index,
            y=["fraud", "non_fraud"],
            labels={"value": "Количество транзакций", "hour": "Время (почасовое)"},
            title=f"Почасовые транзакции для vendor_type = {vendor_for_hourly}"
        )
        fig_hourly.update_layout(legend_title_text='Тип транзакции')

        st.plotly_chart(fig_hourly, use_container_width=True)
