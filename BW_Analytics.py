import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
import pyarrow.parquet as pq
from pyarrow import Table
from pathlib import Path
import plotly.express as px
from datetime import timedelta

st.set_page_config(layout="wide", page_title="Fraud vs Non-Fraud — Grid Layout + Filter")

st.title("Свободное исследование данных BW")

default_path = "d:/Work/ITMO_CONTEST/transaction_fraud_data.parquet"
file_path = st.sidebar.text_input("Путь к Parquet:", value=default_path)

agg_by = st.sidebar.selectbox("Агрегировать по", ["count", "amount"], index=0)


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
def read_data(parquet_path, columns=None):
    table = pq.read_table(parquet_path, columns=columns)
    df = table.to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

# Считаем исходный датафрейм с нужными колонками
cols_needed = ["is_fraud"]

with st.spinner("Читаю данные..."):
    pf = pq.ParquetFile(file_path) 
    first_ten_rows = next(pf.iter_batches(batch_size = 10)) 
    df_col = Table.from_batches([first_ten_rows]).to_pandas()
cols = df_col.columns.to_list()




selected_column = st.selectbox("Выберите vendor_type для отображения:",
    options=cols,
    index=None)
if selected_column != None:
    with st.spinner("Читаю данные...", show_time=True):
        df_full = read_data(file_path, [selected_column, 'timestamp', 'is_fraud', 'amount', 'customer_id'])
    df_filtered = df_full[(df_full["timestamp"] >= time_range[0]) & (df_full["timestamp"] <= time_range[1])]
    # # --- Сохраняем/инициализируем выбранные vendor_type ---
    all_types = sorted(df_filtered[selected_column].dropna().unique())
    # if "selected_vendor_types" not in st.session_state:
    st.session_state.selected_vendor_types = all_types.copy()

    selected_types = st.sidebar.multiselect(
        "Выберите " + selected_column + " для отображения:",
        options=all_types,
        default=st.session_state.selected_vendor_types,
        key="selected_types"
    )

    # # Фильтруем по выбранным vendor_type
    df_filtered = df_filtered[df_filtered[selected_column].isin(selected_types)]

    def aggregate_by(col_name, df, agg_by):
        if agg_by == "count":
            agg = df.groupby([col_name, "is_fraud"]).size().reset_index(name="metric")
        else:
            agg = df.groupby([col_name, "is_fraud"])["amount"].sum().reset_index(name="metric")

        pivot = agg.pivot_table(index=col_name, columns="is_fraud", values="metric", fill_value=0).reset_index()
        if True not in pivot.columns:
            pivot[True] = 0
        if False not in pivot.columns:
            pivot[False] = 0

        pivot = pivot.rename(columns={False: "non_fraud", True: "fraud"})
        pivot["total"] = pivot["fraud"] + pivot["non_fraud"]
        pivot = pivot.sort_values("total", ascending=False).reset_index(drop=True)
        return pivot



    pivot_city = aggregate_by(selected_column, df_filtered, agg_by)


    def build_bar_figure(pivot_df, agg_type):
        if pivot_df.empty:
            return go.Figure()

        # Преобразуем pivot_df для plotly: город/vendor_type, fraud, non_fraud
        df_plot = pivot_df.melt(id_vars=[pivot_df.columns[0]], value_vars=["fraud", "non_fraud"],
                                var_name="type", value_name="count")
        fig = px.bar(
            df_plot,
            x=pivot_df.columns[0], 
            y="count",
            color="type",
            color_discrete_map={"fraud": "red", "non_fraud": "blue"},
            barmode="group",
            labels={pivot_df.columns[0]: pivot_df.columns[0], "count": agg_type, "type": "Тип"}
        )
        fig.update_layout(xaxis_tickangle=-45, height=600, margin=dict(t=40, b=150))
        return fig


    import plotly.graph_objects as go

    # # --- Рисуем графики ---

    st.sidebar.markdown("---")

    st.markdown("### График по " + selected_column)
    fig_city = build_bar_figure(pivot_city, agg_by)
    st.plotly_chart(fig_city, use_container_width=True)

    vendor_for_hourly = st.sidebar.selectbox(
        "Выберите " + selected_column + " для почасового графика",
        options=all_types
    )

    if st.sidebar.button("Показать почасовой график", use_container_width=True):
        # Фильтруем данные по выбранному vendor_type
        df_v = df_filtered[df_filtered[selected_column] == vendor_for_hourly].copy()
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
                color_discrete_map={"fraud": "red", "non_fraud": "blue"},
                labels={"value": "Количество транзакций", "hour": "Время (почасовое)"},
                title=f"Почасовые транзакции для vendor_type = {vendor_for_hourly}"
            )
            fig_hourly.update_layout(legend_title_text='Тип транзакции')

            st.plotly_chart(fig_hourly, use_container_width=True)

    st.sidebar.markdown("---")
    if st.sidebar.button("Kmeans", use_container_width=True):
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.cluster import KMeans
        # Генерация случайных данных
        np.random.seed(42)
        X = df_filtered[['amount', 'customer_id']]
        X['amount'] = X['amount'].astype(int)
        X['customer_id'] = X['customer_id'].str[5:]
        X['customer_id'] = X['customer_id'].astype(int)
        X = X.to_numpy()

        kmeans = KMeans(n_clusters=2)
        kmeans.fit(X)
        # Получаем результаты кластеризации
        labels = kmeans.labels_ # Массив меток кластеров
        centroids = kmeans.cluster_centers_ # Центры кластеров
        fig = go.Figure()
        # 1. Добавляем точки данных с цветами по кластерам
        fig.add_trace(go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode='markers',
            marker=dict(
                color=labels,
                size=8,
                colorscale=['orange', 'purple'],
                opacity=0.7
            ),
            text=df_filtered['is_fraud'],
            hoverinfo="x+y+text",
            name='Точки данных'
        ))

        fig.update_layout(
            title='Результаты кластеризации методом K-means',
            xaxis_title='Сумма транзакции',
            yaxis_title='ID клиента',
            legend_title='Легенда',
            hovermode='closest'
        )

        st.plotly_chart(fig, use_container_width=True)
