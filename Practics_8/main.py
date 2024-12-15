
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error


ticker = "VOO"
data = yf.download(ticker, start="2023-11-01", end="2024-10-31")

# Первинний аналіз даних
print("\nПерші кілька рядків датасету:")
print(data.head())

# Перевірка на пропущені значення
missing_values = data.isnull().sum()
print("\nПропущені значення у кожному стовпці:")
print(missing_values)

# Побудова графіка зміни ціни закриття
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label="Ціна закриття")
plt.title(f"Графік зміни ціни закриття акцій {ticker}")
plt.xlabel("Дата")
plt.ylabel("Ціна закриття ($)")
plt.legend()
plt.grid()
plt.show()

# Базова описова статистика
print("\nОписова статистика:")
print(data.describe())

from statsmodels.tsa.seasonal import seasonal_decompose

# Виділення тренду за допомогою ковзного середнього (30 днів)
data['Trend'] = data['Close'].rolling(window=30).mean()

# Декомпозиція часового ряду
decomposition = seasonal_decompose(data['Close'], model='additive', period=30)

# Виділення компонент
data['Seasonal'] = decomposition.seasonal
data['Residual'] = decomposition.resid

# Побудова графіків
plt.figure(figsize=(12, 8))

# Оригінальний ряд
plt.subplot(4, 1, 1)
plt.plot(data['Close'], label="Оригінальний ряд")
plt.legend()

# Тренд
plt.subplot(4, 1, 2)
plt.plot(data['Trend'], label="Тренд", color='orange')
plt.legend()

# Сезонність
plt.subplot(4, 1, 3)
plt.plot(data['Seasonal'], label="Сезонність", color='green')
plt.legend()

# Випадкова компонента
plt.subplot(4, 1, 4)
plt.plot(data['Residual'], label="Випадкова компонента", color='teal')
plt.legend()

plt.tight_layout()
plt.show()

# Прості ковзні середні (SMA 7 і 30 днів)
data['SMA_7'] = data['Close'].rolling(window=7).mean()
data['SMA_30'] = data['Close'].rolling(window=30).mean()


# Розрахунок RSI (Relative Strength Index)
# Формула RSI: RSI = 100 - (100 / (1 + RS)), де RS = середній приріст / середній спад

def calculate_rsi(data, window=14):
    # Різниця між сусідніми значеннями
    delta = data['Close'].diff()

    # Розрахунок приростів і спадів
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Середній приріст і середній спад
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    # Розрахунок RS і RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


# Додаємо RSI до даних
data['RSI'] = calculate_rsi(data, window=14)

# Розрахунок 30-денної волатильності (стандартне відхилення)
data['Volatility'] = data['Close'].rolling(window=30).std()

# Побудова графіків технічних індикаторів
plt.figure(figsize=(12, 10))

# Графік SMA
plt.subplot(3, 1, 1)
plt.plot(data['Close'], label="Ціна закриття", color='blue')
plt.plot(data['SMA_7'], label="SMA (7 днів)", color='orange')
plt.plot(data['SMA_30'], label="SMA (30 днів)", color='green')
plt.title("Прості ковзні середні (7 і 30 днів)")
plt.legend()

# Графік RSI
plt.subplot(3, 1, 2)
plt.plot(data['RSI'], label="RSI", color='purple')
plt.axhline(70, linestyle='--', color='red', alpha=0.7, label="Перекупленість")
plt.axhline(30, linestyle='--', color='blue', alpha=0.7, label="Перепроданість")
plt.title("Відносна сила (RSI)")
plt.legend()

# Графік волатильності
plt.subplot(3, 1, 3)
plt.plot(data['Volatility'], label="Волатильність (30 днів)", color='brown')
plt.title("30-денна волатильність")
plt.legend()

plt.tight_layout()
plt.show()

# Розділення даних на навчальну і тестову вибірки
train = data['Close'][:int(0.8 * len(data))]
test = data['Close'][int(0.8 * len(data)):]

# Прогнозування за допомогою експоненційного згладжування
model_es = ExponentialSmoothing(train, trend="add", seasonal=None, seasonal_periods=12)
model_es_fit = model_es.fit()
forecast_es = model_es_fit.forecast(len(test))

# Прогнозування за допомогою ARIMA
model_arima = ARIMA(train, order=(5, 1, 0))
model_arima_fit = model_arima.fit()
forecast_arima = model_arima_fit.forecast(len(test))

# Оцінка якості прогнозу
mse_es = mean_squared_error(test, forecast_es)
mae_es = mean_absolute_error(test, forecast_es)
mse_arima = mean_squared_error(test, forecast_arima)
mae_arima = mean_absolute_error(test, forecast_arima)

print("\nОцінка якості прогнозу:")
print(f"- Експоненційне згладжування: MSE={mse_es:.2f}, MAE={mae_es:.2f}")
print(f"- ARIMA: MSE={mse_arima:.2f}, MAE={mae_arima:.2f}")

# Візуалізація прогнозу
plt.figure(figsize=(12, 6))
plt.plot(train, label="Навчальні дані")
plt.plot(test, label="Тестові дані", color='orange')
plt.plot(test.index, forecast_es, label="Прогноз (ES)", linestyle='--', color='green')
plt.plot(test.index, forecast_arima, label="Прогноз (ARIMA)", linestyle='--', color='red')
plt.title(f"Прогноз цін закриття акцій {ticker}")
plt.xlabel("Дата")
plt.ylabel("Ціна закриття ($)")
plt.legend()
plt.grid()
plt.show()
