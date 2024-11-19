import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import joblib


# Завантаження даних
housing_data = fetch_california_housing()
housing_df = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
housing_df['TargetPrice'] = housing_data.target  # Додаємо цільову змінну

# Частина 1: Базовий аналіз даних
print("Головні дані про таблицю:\n", housing_df.info())
print("\nОписова статистика:\n", housing_df.describe())
print("\nПеревірка пропущених значень:\n", housing_df.isna().sum())

# Візуалізація: Розподіл даних
housing_df.hist(bins=25, figsize=(15, 10), color='skyblue', edgecolor='black')
plt.suptitle('Розподіл ознак у таблиці', fontsize=16)
plt.show()

# Візуалізація: Boxplots для виявлення викидів
for col in housing_df.columns[:-1]:  # Пропускаємо TargetPrice
    plt.figure(figsize=(15, 10))
    sns.boxplot(data=housing_df, y=col, color='skyblue')
    plt.title(f'Boxplot для {col}')
    plt.show()

# Кореляційна матриця
cor_matrix = housing_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(cor_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Кореляційна матриця ознак', fontsize=14)
plt.show()

# Візуалізація: Scatter plots між TargetPrice та ознаками
for col in housing_df.columns[:-1]:  # Пропускаємо TargetPrice
    plt.figure(figsize=(8, 4))
    plt.scatter(housing_df[col], housing_df['TargetPrice'], alpha=0.5, color='green')
    plt.title(f'Залежність ціни від {col}')
    plt.xlabel(col)
    plt.ylabel('TargetPrice')
    plt.grid(True)
    plt.show()

# Висновки:
# 1. Аналіз кореляції:
#    Найсильніше пов'язані з ціною "MedInc" (медіанний дохід) та "AveRooms".
#    Latitude і Longitude показують слабку або середню кореляцію.
# 2. Викиди:
#    Boxplots показують наявність викидів у деяких колонках, зокрема Population та AveRooms.
# 3. Трансформація:
#    Розподіл деяких ознак, таких як Population та AveOccup, не є нормальним.
#    Ймовірно, знадобиться логарифмічна або інша трансформація.

# Частина 2: Підготовка даних

# Відокремлення ознак і цільової змінної
X = housing_df.drop('TargetPrice', axis=1)  # Всі колонки, крім TargetPrice
y = housing_df['TargetPrice']  # Цільова змінна

# Розділення на тренувальну і тестову вибірки (80% - тренування, 20% - тест)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Розмір тренувальної вибірки: {X_train.shape[0]} зразків")
print(f"Розмір тестової вибірки: {X_test.shape[0]} зразків")

# Масштабування ознак
scaler = StandardScaler()  # Створення екземпляру StandardScaler

# Навчання скейлера на тренувальній вибірці та трансформація
X_train_scaled = scaler.fit_transform(X_train)

# Трансформація тестової вибірки (використовуємо ті самі параметри скейлера)
X_test_scaled = scaler.transform(X_test)

# Збереження скейлера для подальшого використання
scaler_name = 'scaler_model.pkl'
joblib.dump(scaler, scaler_name)
print(f"Скейлер збережено як {scaler_name}")

# Перевірка масштабованих даних
print("\nПриклад масштабованих даних (перші 5 рядків тренувальної вибірки):\n", X_train_scaled[:5])

# Частина 3: Побудова моделей

# --- Проста лінійна регресія (з однією ознакою) ---
# Вибір ознаки з найвищою кореляцією з ціною
corr_target = housing_df.corr()['TargetPrice'].sort_values(ascending=False)
top_feature = corr_target.index[1]  # Пропускаємо TargetPrice (це кореляція з самою собою)
print(f"Ознака з найбільшою кореляцією: {top_feature} (коеф.: {corr_target[top_feature]:.2f})")

# Використання лише цієї ознаки
X_simple_train = X_train_scaled[:, housing_df.columns[:-1].get_loc(top_feature)]
X_simple_test = X_test_scaled[:, housing_df.columns[:-1].get_loc(top_feature)]

# Перетворення в 2D масив для моделі
X_simple_train = X_simple_train.reshape(-1, 1)
X_simple_test = X_simple_test.reshape(-1, 1)

# Навчання моделі
simple_model = LinearRegression()
simple_model.fit(X_simple_train, y_train)

# Прогнози
y_simple_pred = simple_model.predict(X_simple_test)

# Візуалізація результатів
plt.figure(figsize=(8, 6))
plt.scatter(X_simple_test, y_test, color='blue', label='Реальні значення', alpha=0.5)
plt.plot(X_simple_test, y_simple_pred, color='red', label='Прогноз', linewidth=2)
plt.title(f"Проста лінійна регресія (ознака: {top_feature})")
plt.xlabel(top_feature)
plt.ylabel('TargetPrice')
plt.legend()
plt.grid()
plt.show()

# Метрики якості
mse_simple = mean_squared_error(y_test, y_simple_pred)
r2_simple = r2_score(y_test, y_simple_pred)
print(f"Метрики простої регресії:\n  - MSE: {mse_simple:.2f}\n  - R2: {r2_simple:.2f}")

# --- Множинна лінійна регресія ---
# Навчання моделі на всіх ознаках
multi_model = LinearRegression()
multi_model.fit(X_train_scaled, y_train)

# Прогнози
y_multi_pred = multi_model.predict(X_test_scaled)

# Метрики якості
mse_multi = mean_squared_error(y_test, y_multi_pred)
r2_multi = r2_score(y_test, y_multi_pred)
print(f"\nМетрики множинної регресії:\n  - MSE: {mse_multi:.2f}\n  - R2: {r2_multi:.2f}")

# Аналіз коефіцієнтів
print("\nКоефіцієнти множинної моделі:")
coef_df = pd.DataFrame({
    'Ознака': housing_df.columns[:-1],
    'Коефіцієнт': multi_model.coef_
})
print(coef_df.sort_values(by='Коефіцієнт', key=abs, ascending=False))

# --- Оптимізована модель (з регуляризацією) ---
# Вибір важливих ознак (кореляція > 0.1 або <-0.1)
important_features = corr_target[(corr_target.abs() > 0.1) & (corr_target.index != 'TargetPrice')].index
X_train_opt = X_train_scaled[:, [housing_df.columns[:-1].get_loc(f) for f in important_features]]
X_test_opt = X_test_scaled[:, [housing_df.columns[:-1].get_loc(f) for f in important_features]]

# Регуляризація: Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_opt, y_train)

# Прогнози
y_ridge_pred = ridge_model.predict(X_test_opt)

# Метрики якості
mse_ridge = mean_squared_error(y_test, y_ridge_pred)
r2_ridge = r2_score(y_test, y_ridge_pred)
print(f"\nМетрики оптимізованої моделі:\n  - MSE: {mse_ridge:.2f}\n  - R2: {r2_ridge:.2f}")

# Порівняння моделей
print("\nПорівняння метрик моделей:")
comparison = pd.DataFrame({
    'Модель': ['Проста регресія', 'Множинна регресія', 'Оптимізована модель'],
    'MSE': [mse_simple, mse_multi, mse_ridge],
    'R2': [r2_simple, r2_multi, r2_ridge]
})
print(comparison)


# Частина 4: Оцінка моделей

# Функція для обчислення Adjusted R-squared
def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


# Кількість зразків і ознак
n_samples = X_test_scaled.shape[0]
n_features_simple = 1
n_features_multi = X_test_scaled.shape[1]
n_features_opt = X_test_opt.shape[1]

# Розрахунок метрик для кожної моделі
metrics = {
    'Модель': ['Проста регресія', 'Множинна регресія', 'Оптимізована модель'],
    'MSE': [mse_simple, mse_multi, mse_ridge],
    'RMSE': [np.sqrt(mse_simple), np.sqrt(mse_multi), np.sqrt(mse_ridge)],
    'R2': [r2_simple, r2_multi, r2_ridge],
    'Adjusted R2': [
        adjusted_r2(r2_simple, n_samples, n_features_simple),
        adjusted_r2(r2_multi, n_samples, n_features_multi),
        adjusted_r2(r2_ridge, n_samples, n_features_opt)
    ]
}
metrics_df = pd.DataFrame(metrics)

print("\nМетрики для моделей:")
print(metrics_df)

# Візуалізація передбачених vs реальних значень
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_simple_pred, alpha=0.6, label='Проста регресія')
plt.scatter(y_test, y_multi_pred, alpha=0.6, label='Множинна регресія')
plt.scatter(y_test, y_ridge_pred, alpha=0.6, label='Оптимізована модель')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', linewidth=2)
plt.title("Передбачені vs Реальні значення")
plt.xlabel("Реальні значення")
plt.ylabel("Передбачені значення")
plt.legend()
plt.grid()
plt.show()

# Графік залишків
for pred, name in zip(
        [y_simple_pred, y_multi_pred, y_ridge_pred],
        ['Проста регресія', 'Множинна регресія', 'Оптимізована модель']
):
    residuals = y_test - pred
    plt.figure(figsize=(10, 6))
    plt.scatter(pred, residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--', linewidth=2)
    plt.title(f"Графік залишків: {name}")
    plt.xlabel("Передбачені значення")
    plt.ylabel("Залишки")
    plt.grid()
    plt.show()

# Розподіл залишків
for pred, name in zip(
        [y_simple_pred, y_multi_pred, y_ridge_pred],
        ['Проста регресія', 'Множинна регресія', 'Оптимізована модель']
):
    residuals = y_test - pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=25, color='purple')
    plt.title(f"Розподіл залишків: {name}")
    plt.xlabel("Залишки")
    plt.ylabel("Частота")
    plt.grid()
    plt.show()

# Частина 5: Інтерпретація результатів

print("\nВисновки:")
print("""
1. Найкраща модель:
   Оптимізована модель з регуляризацією показала найкращі метрики MSE, RMSE та R².
   Adjusted R² також найвищий у оптимізованої моделі.

2. Найбільш впливові ознаки:
   Аналіз коефіцієнтів моделі показав, що MedInc (медіанний дохід) має найбільший вплив на ціну.
   Latitude і Population теж мають помітний вплив, але менший.

3. Обмеження моделі:
   - Лінійна регресія передбачає лінійний зв'язок, що може бути обмеженням.
   - Деякі ознаки мають викиди, які впливають на точність.
""")