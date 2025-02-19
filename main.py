import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

#  Crypto Volatility Detector

# Load historical crypto data from CSV
def get_crypto_data(file_path=r"C:\Rishee javascript react\Bhumi1\Bitcoin BEP2.csv"):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    return df

# Calculate volatility indicators
def calculate_volatility(df):
    df['returns'] = df['Close'].pct_change()
    df['std_dev'] = df['returns'].rolling(window=10).std()
    df['bollinger_high'] = df['Close'].rolling(window=20).mean() + (df['std_dev'] * 2)
    df['bollinger_low'] = df['Close'].rolling(window=20).mean() - (df['std_dev'] * 2)
    df['ATR'] = (df['High'] - df['Low']).rolling(window=14).mean()
    return df

# Train an ML model to predict volatility
def train_volatility_model(df):
    df = df.dropna()
    X = df[['Close', 'std_dev', 'ATR']]
    y = df['std_dev'].shift(-1)  # Predict next-day volatility
    X_train, X_test, y_train, y_test = train_test_split(X[:-1], y[:-1], test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f'MAE: {mean_absolute_error(y_test, predictions):.5f}')
    return model

# Visualize volatility trends
def plot_volatility(df):
    plt.figure(figsize=(12,6))
    plt.plot(df['Date'], df['Close'], label='Closing Price', color='blue')
    plt.plot(df['Date'], df['bollinger_high'], label='Bollinger High', linestyle='dashed', color='red')
    plt.plot(df['Date'], df['bollinger_low'], label='Bollinger Low', linestyle='dashed', color='green')
    plt.fill_between(df['Date'], df['bollinger_low'], df['bollinger_high'], color='gray', alpha=0.3)
    plt.legend()
    plt.title('Crypto Volatility Detection')
    plt.show()

# Run the pipeline
data = get_crypto_data()
data = calculate_volatility(data)
plot_volatility(data)
model = train_volatility_model(data)
