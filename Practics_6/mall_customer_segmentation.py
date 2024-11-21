import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Завантаження датасету
try:
    data = pd.read_csv("Mall_Customers.csv")
    print("Дані успішно завантажено!")
except FileNotFoundError:
    print("Помилка: Файл не знайдено. Перевірте місцезнаходження та назву файлу.")

# Перевірка перших кількох рядків і типів даних
print("\nПерші рядки датасету:")
print(data.head())

print("\nОпис датасету:")
print(data.info())

# Перевірка пропущених значень
missing_values = data.isnull().sum()
print("\nПропущені значення в кожному стовпці:")
print(missing_values)

# Кодування категоріальних змінних
# Оскільки у нас є стовпець 'Gender', закодуємо його
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])
print("\nПеретворення 'Gender' на числові значення (Male=1, Female=0).")

# Стандартизація числових ознак
numerical_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[numerical_features])

# Збереження стандартизованих даних у DataFrame
scaled_df = pd.DataFrame(data_scaled, columns=numerical_features)
print("\nДані після стандартизації (перші рядки):")
print(scaled_df.head())

# Застосування PCA
pca = PCA()
pca_data = pca.fit_transform(data_scaled)

# Пояснена дисперсія для кожної компоненти
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

# Візуалізація кумулятивної поясненої дисперсії
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.title("Кумулятивна пояснена дисперсія (PCA)")
plt.xlabel("Кількість компонент")
plt.ylabel("Кумулятивна дисперсія")
plt.grid(True)
plt.show()

# Визначення оптимальної кількості компонент
optimal_components = np.argmax(cumulative_variance >= 0.9) + 1
print(f"\nОптимальна кількість компонент (90% дисперсії): {optimal_components}")

# PCA з оптимальною кількістю компонент
pca_final = PCA(n_components=optimal_components)
pca_reduced_data = pca_final.fit_transform(data_scaled)

# Візуалізація у 2D-просторі
plt.figure(figsize=(8, 6))
plt.scatter(pca_reduced_data[:, 0], pca_reduced_data[:, 1], c='lightgreen', s=50, alpha=0.7)
plt.title("PCA: Візуалізація у 2D-просторі")
plt.xlabel("Головна компонента 1")
plt.ylabel("Головна компонента 2")
plt.grid(True)
plt.show()

# t-SNE з базовими параметрами
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
tsne_results = tsne.fit_transform(data_scaled)

# Візуалізація результатів t-SNE
plt.figure(figsize=(8, 6))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c='purple', s=50, alpha=0.7)
plt.title("t-SNE: Візуалізація у 2D-просторі")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.grid(True)
plt.show()

# Експерименти з параметрами t-SNE
for perplexity in [5, 30, 50]:
    tsne_experiment = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_experiment_results = tsne_experiment.fit_transform(data_scaled)

    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_experiment_results[:, 0], tsne_experiment_results[:, 1], s=50, alpha=0.7)
    plt.title(f"t-SNE з perplexity={perplexity}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True)
    plt.show()

print("\nПорівняння PCA і t-SNE:")
print("- PCA дозволяє зменшити розмірність із збереженням більшості інформації.")
print("- t-SNE краще виявляє локальні структури, але не гарантує збереження глобальних патернів.")

# Кластеризація після PCA
kmeans_pca = KMeans(n_clusters=optimal_components, random_state=42)
pca_clusters = kmeans_pca.fit_predict(pca_reduced_data)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_reduced_data[:, 0], y=pca_reduced_data[:, 1], hue=pca_clusters, palette="Set2", s=50, alpha=0.8)
plt.title("Кластеризація після PCA")
plt.xlabel("Головна компонента 1")
plt.ylabel("Головна компонента 2")
plt.legend(title="Кластери")
plt.grid(True)
plt.show()

# Кластеризація після t-SNE
kmeans_tsne = KMeans(n_clusters=optimal_components, random_state=42)
tsne_clusters = kmeans_tsne.fit_predict(tsne_results)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=tsne_clusters, palette="Set1", s=50, alpha=0.8)
plt.title("Кластеризація після t-SNE")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(title="Кластери")
plt.grid(True)
plt.show()

# Аналіз кластерів (приклад простого виводу)
print("\nІнтерпретація кластерів:")
for cluster in np.unique(pca_clusters):
    print(f"- Кластер {cluster}: Ця група клієнтів має унікальні ознаки (засновані на PCA).")
    print(f"  Пропозиція: спеціальні акції для клієнтів із групи {cluster}.")

# Порівняння кластеризації на PCA, t-SNE та оригінальних даних
kmeans_original = KMeans(n_clusters=optimal_components, random_state=42)
original_clusters = kmeans_original.fit_predict(data_scaled)

# Оцінка якості кластеризації для кожного методу
from sklearn.metrics import silhouette_score

silhouette_pca = silhouette_score(pca_reduced_data, pca_clusters)
silhouette_tsne = silhouette_score(tsne_results, tsne_clusters)
silhouette_original = silhouette_score(data_scaled, original_clusters)

print("\nОцінка якості кластеризації (Silhouette Score):")
print(f"- Оригінальні дані: {silhouette_original:.2f}")
print(f"- PCA-зменшення: {silhouette_pca:.2f}")
print(f"- t-SNE-зменшення: {silhouette_tsne:.2f}")

# Аналіз середніх значень кожного кластеру (приклад для PCA)
data['PCA_Cluster'] = pca_clusters

pca_cluster_analysis = data.groupby('PCA_Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
print("\nСередні значення показників для кожного кластеру (PCA):")
print(pca_cluster_analysis)

# Пропозиції на основі кластерів
print("\nМаркетингові стратегії:")
for cluster in pca_cluster_analysis.index:
    print(f"- Кластер {cluster}:")
    if pca_cluster_analysis.loc[cluster, 'Spending Score (1-100)'] > 75:
        print("  Рекомендація: створити програми лояльності преміум-класу для клієнтів цього кластеру.")
    elif pca_cluster_analysis.loc[cluster, 'Annual Income (k$)'] < 50:
        print("  Рекомендація: пропонувати акції та знижки для залучення клієнтів із нижчою купівельною спроможністю.")
    else:
        print("  Рекомендація: спеціальні пропозиції для утримання клієнтів із середнім рівнем витрат.")

# Узагальнення результатів
print("\nВисновки:")
print("- PCA виявляє основні глобальні тренди в даних, що дозволяє створювати узагальнені стратегії.")
print("- Кластеризація на основі PCA забезпечує чіткі межі між кластерами, що добре підходить для маркетингових стратегій.")
print("- t-SNE підходить для виявлення локальних патернів, які можуть бути корисними для вивчення унікальних клієнтських сегментів.")
print("- t-SNE показує цікаві локальні особливості, але результати можуть бути менш стійкими.")
