import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

try:
    data = pd.read_csv("Mall_Customers.csv")
    print("Дані успішно завантажені!")
except FileNotFoundError:
    print("Файл не знайдено. Перевірте назву файлу та його місцезнаходження.")

# Первинний аналіз (EDA)
print("\nПерша пара рядків:\n", data.head())
print("\nІнформація про датасет:\n")
data.info()

# Перевірка на пропущені значення
missing_values = data.isnull().sum()
print("\nКількість пропущених значень у кожному стовпці:\n", missing_values)

# Побудова гістограм
plt.style.use('seaborn-v0_8-pastel')

data_points = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
for data_point in data_points:
    plt.hist(data[data_point], bins=20, color='lightgreen', edgecolor='black')
    plt.title(f'Distribution of {data_point}')
    plt.xlabel(data_point)
    plt.ylabel('Frequency')
    plt.show()

# Основні статистичні показники
stats = data.describe()
print("\nОсновні статистичні показники:\n", stats)

# Стандартизація даних
scaler = StandardScaler()
numerical_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
scaled_data = scaler.fit_transform(data[numerical_features])

# Ставимо назад у DataFrame
scaled_df = pd.DataFrame(scaled_data, columns=numerical_features)
print("\nСтандартизовані дані (перші рядки):\n", scaled_df.head())

# Задача: Визначення оптимальної кількості кластерів методом ліктя
# Тестуватимемо від 1 до 10 кластерів
inertia_values = []
range_clusters = range(1, 11)

# Вивчаємо, як змінюється інерція при різній кількості кластерів
for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_df)
    inertia_values.append(kmeans.inertia_)

# Візуалізація методу ліктя
plt.figure(figsize=(8, 5))
plt.plot(range_clusters, inertia_values, marker='o', linestyle='--', color='b')
plt.title("Метод ліктя: вибір оптимальної кількості кластерів")
plt.xlabel("Кількість кластерів")
plt.ylabel("Інерція")
plt.grid(True)
plt.show()

# Аналіз: Коефіцієнт силуету
silhouette_scores = []

# Визначаємо силует для кількості кластерів від 2 до 10
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_df)
    score = silhouette_score(scaled_df, cluster_labels)
    silhouette_scores.append(score)

# Візуалізація залежності коефіцієнта силуету від кількості кластерів
plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='--', color='g')
plt.title("Коефіцієнт силуету: оцінка якості кластеризації")
plt.xlabel("Кількість кластерів")
plt.ylabel("Коефіцієнт силуету")
plt.grid(True)
plt.show()

# Вибір оптимальної кількості кластерів:
optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"Оптимальна кількість кластерів за коефіцієнтом силуету: {optimal_clusters}")
