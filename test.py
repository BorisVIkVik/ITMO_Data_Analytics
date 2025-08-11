import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from textwrap import fill

# 1. Загрузка данных с оптимизацией
def load_data():
    # Загружаем только необходимые колонки
    cols = ['city', 'country', 'timestamp', 'amount', 'is_fraud']
    transactions = pd.read_parquet('transaction_fraud_data.parquet', columns=cols)
    
    # Преобразуем даты и оптимизируем типы
    transactions['date'] = pd.to_datetime(transactions['timestamp']).dt.date
    transactions['city_country'] = transactions['city'] + ', ' + transactions['country']
    transactions.drop(columns=['timestamp'], inplace=True)
    
    return transactions

# Загрузка данных
print("Загрузка данных...")
transactions = load_data()

# 2. Подготовка временного диапазона
min_date = transactions['date'].min()
max_date = transactions['date'].max()
date_range = pd.date_range(start=min_date, end=max_date, freq='D')

# 3. Функция для агрегации данных
def prepare_city_data(df, end_date):
    # Фильтрация по дате
    filtered = df[df['date'] <= end_date.date()]
    
    # Агрегация данных
    agg = filtered.groupby(['city_country', 'is_fraud']).agg(
        count=('amount', 'size'),
        amount=('amount', 'sum')
    ).unstack(fill_value=0)
    
    # Формируем результат
    result = []
    for city_country, row in agg.iterrows():
        result.append({
            'city_country': city_country,
            'fraud_count': row[('count', True)],
            'fraud_amount': row[('amount', True)],
            'legit_count': row[('count', False)],
            'legit_amount': row[('amount', False)],
            'total_count': row[('count', True)] + row[('count', False)]
        })
    
    return pd.DataFrame(result)

# 4. Создаем базовую визуализацию
def create_base_figure():
    fig = go.Figure()
    
    fig.update_layout(
        title='<b>Мошеннические и легитимные операции по городам</b>',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 100]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 100]),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True,
        legend=dict(x=0.5, y=-0.1, orientation='h'),
        hovermode='closest',
        height=800,
        width=1200,
        margin=dict(l=50, r=50, b=100, t=100)
    )
    
    # Добавляем легенду
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='red'),
        name='Мошеннические'
    ))
    
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='blue'),
        name='Легитимные'
    ))
    
    return fig

# 5. Создаем анимированную визуализацию
def create_visualization():
    fig = create_base_figure()
    
    # Получаем список всех городов
    all_cities = transactions['city_country'].unique()
    num_cities = len(all_cities)
    
    # Создаем позиции для городов в виде сетки
    grid_size = int(np.ceil(np.sqrt(num_cities)))
    city_positions = {}
    
    for i, city in enumerate(all_cities):
        row = i // grid_size
        col = i % grid_size
        city_positions[city] = {
            'x': 5 + (90/grid_size) * col,
            'y': 5 + (90/grid_size) * row
        }
    
    # Создаем кадры анимации (возьмем каждый 3-й день для примера)
    frames = []
    
    for date in date_range[::3]:
        city_data = prepare_city_data(transactions, date)
        
        # Создаем данные для кадра
        fraud_traces = []
        legit_traces = []
        labels = []
        
        for _, row in city_data.iterrows():
            pos = city_positions.get(row['city_country'], {'x': 50, 'y': 50})
            
            # Рассчитываем радиусы
            fraud_radius = np.sqrt(row['fraud_count']) * 0.5
            legit_radius = np.sqrt(row['legit_count']) * 0.5
            
            # Добавляем круги
            fraud_traces.append(
                go.Scatter(
                    x=[pos['x']],
                    y=[pos['y']],
                    mode='markers',
                    marker=dict(
                        size=fraud_radius,
                        color='red',
                        opacity=0.7,
                        line=dict(width=0)
                    ),
                    name='Мошеннические',
                    text=f"{row['city_country']}<br>Мошеннических: {row['fraud_count']}",
                    hoverinfo='text'
                )
            )
            
            legit_traces.append(
                go.Scatter(
                    x=[pos['x']],
                    y=[pos['y']],
                    mode='markers',
                    marker=dict(
                        size=legit_radius,
                        color='blue',
                        opacity=0.5,
                        line=dict(width=0)
                    ),
                    name='Легитимные',
                    text=f"{row['city_country']}<br>Легитимных: {row['legit_count']}",
                    hoverinfo='text'
                )
            )
            
            # Добавляем подпись города
            labels.append(
                go.Scatter(
                    x=[pos['x']],
                    y=[pos['y'] - 3],
                    mode='text',
                    text=[row['city_country'].split(',')[0]],
                    textposition='top center',
                    textfont=dict(size=8),
                    hoverinfo='none'
                )
            )
        
        # Собираем все trace'ы для кадра
        frame_data = fraud_traces + legit_traces + labels
        
        # Создаем кадр
        frames.append(
            go.Frame(
                data=frame_data,
                name=str(date.date()),
                layout=go.Layout(
                    annotations=[dict(
                        text=f"Дата: {date.strftime('%Y-%m-%d')}",
                        x=0.5,
                        y=1.05,
                        showarrow=False,
                        font=dict(size=14)
                    )]
                )
            )
        )
    
    # Добавляем кадры к фигуре
    fig.frames = frames
    
    # Настраиваем слайдер
    fig.update_layout(
        sliders=[{
            'active': len(frames)-1,
            'steps': [
                {
                    'args': [[f.name], {'frame': {'duration': 300, 'redraw': True}, 'mode': 'immediate'}],
                    'label': f.date.strftime('%Y-%m-%d'),
                    'method': 'animate'
                }
                for i, f in enumerate(frames)
            ],
            'transition': {'duration': 300},
            'x': 0.1,
            'y': 0,
            'currentvalue': {'prefix': 'Дата: ', 'font': {'size': 14}}
        }]
    )
    
    return fig

# 6. Создаем и сохраняем визуализацию
print("Создание визуализации...")
fig = create_visualization()
fig.write_html('fraud_circles_final.html')
print("Готово! Результат сохранен в fraud_circles_final.html")