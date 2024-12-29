[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/2AJIyfdd)
## Dataset

https://www.kaggle.com/code/brightezeoha/bike-sales-in-europe-eda

## Plan

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
data = pd.read_csv('/content/Sales.csv')

# 1. Очистка данных
# Преобразование типа данных для столбца "Date"
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Удаление дубликатов
data = data.drop_duplicates()

# Проверка и обработка пропущенных значений
missing_data = data.isnull().sum()
# Если пропущенные значения есть, заполним их средними значениями для числовых данных
data.fillna(data.mean(numeric_only=True), inplace=True)

# Обработка категориальных пропущенных значений
categorical_cols = data.select_dtypes(include='object').columns
for col in categorical_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Удаление выбросов (метод IQR)
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_columns:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

# Проверка данных после очистки
print("Первые строки данных после очистки:")
print(data.head())

# 2. Исследовательский анализ данных (EDA)
# Описательная статистика
print("\nОписательная статистика числовых данных:")
print(data.describe())

# Анализ категориальных данных
for col in categorical_cols:
    print(f"\nЧастоты для столбца {col}:")
    print(data[col].value_counts())

# Визуализация распределения числовых данных
plt.figure(figsize=(10, 6))
sns.histplot(data['Revenue'], bins=30, kde=True)
plt.title('Распределение дохода (Revenue)')
plt.xlabel('Revenue')
plt.ylabel('Frequency')
plt.show()

# Корреляционная матрица: выбираем только числовые столбцы
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Проверяем наличие неожиданных строковых значений
for col in numeric_data.columns:
    if data[col].dtype == 'object':
        print(f"Столбец {col} содержит некорректные значения: {data[col].unique()}")

# Строим корреляционную матрицу только для числовых данных
plt.figure(figsize=(12, 8))
corr_matrix = numeric_data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Корреляционная матрица')
plt.show()

# Histogram of Order Quantity
plt.figure(figsize=(8, 5))
plt.hist(data['Order_Quantity'], bins=30, edgecolor='k', alpha=0.7)
plt.title('Distribution of Order Quantity')
plt.xlabel('Order Quantity')
plt.ylabel('Frequency')
plt.show()

# Bar plot for Product Categories
plt.figure(figsize=(8, 5))
data['Product_Category'].value_counts().plot(kind='bar')
plt.title('Product Category Distribution')
plt.xlabel('Product Category')
plt.ylabel('Frequency')
plt.show()


# 3. Преобразование данных и проектирование признаков
# Создание нового признака "Profit Margin" (рентабельность)
data['Profit_Margin'] = data['Profit'] / data['Revenue']

# Преобразование категориальных данных в числовые
data['Gender_Code'] = data['Customer_Gender'].map({'M': 0, 'F': 1})

# Фильтрация данных
filtered_data = data[data['Revenue'] > 100]  # Фильтр по доходу

# 4. Сохранение итогового результата
data.to_csv('Cleaned_Sales.csv', index=False)

# Итоговое сообщение
print("\nОчистка данных и анализ завершены. Итоговый файл сохранен как 'Cleaned_Sales.csv'.")
