import ccxt
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report  # Keep this line as it is
from sklearn.exceptions import UndefinedMetricWarning  # Import UndefinedMetricWarning from sklearn.exceptions
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
import logging
import joblib
import os
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Ignore UserWarnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)  # Ignore UndefinedMetricWarnings

# Model file path
model_file_path = 'trading_model.pkl'

# Set up logging
logging.basicConfig(level=logging.INFO, filename='trading_bot.log', format='%(asctime)s - %(levelname)s - %(message)s')

# Set up connection
exchange = ccxt.binance()

# Configuration
symbol = 'BTC/USDT'   # Trading pair
timeframe = '1m'      # Time interval
training_limit = 10000 # Historical data points for training
initial_balance = 100000  # Starting virtual USD balance
trade_size = 0.1  # Fraction of balance to use per trade

# Paper trading variables
balance = initial_balance
btc_balance = 1
trade_log = []

# Step 1: Fetch historical data
def fetch_data():
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=training_limit)
        data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        if data.empty:
            print("No data fetched. Check connection or symbol/timeframe validity.")

        return data
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame or handle as needed

# Step 2: Add indicators
def add_indicators(data):
    data['SMA_50'] = SMAIndicator(data['close'], window=50).sma_indicator()
    data['RSI'] = RSIIndicator(data['close'], window=14).rsi()
    
    # Bollinger Bands
    bb = BollingerBands(data['close'], window=20, window_dev=2)
    data['BB_Middle'] = bb.bollinger_mavg()
    data['BB_Upper'] = bb.bollinger_hband()
    data['BB_Lower'] = bb.bollinger_lband()
    
    data.dropna(inplace=True)  # Drop rows with NaN values
    return data

# Step 3: Prepare training data
def prepare_training_data(data):
    data['Target'] = (data['close'].shift(-1) > data['close']).astype(int)
    features = data[['SMA_50', 'RSI', 'BB_Middle', 'BB_Upper', 'BB_Lower']].values  # Include Bollinger Bands
    labels = data['Target'].values

    # Normalize features
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)
    return X_train, y_train, X_val, y_val, scaler

# Step 4: Train the machine learning model
def train_model(X_train, y_train, X_val, y_val):
    model = Sequential([
        Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=2)

    # Evaluate on validation set
    y_pred = (model.predict(X_val) > 0.5).astype(int)
    logging.info("Model performance on validation set:\n" + classification_report(y_val, y_pred))
    print("Shapes - X_train:", X_train.shape, "y_train:", y_train.shape, "X_val:", X_val.shape, "y_val:", y_val.shape)
    return model

# Initialize cumulative_data at the beginning of your script
cumulative_data = pd.DataFrame()  # Initialize as an empty DataFrame

# Step 5: Simulate paper trading
def paper_trade(model, scaler):
    global balance, btc_balance, trade_log, cumulative_data  # Ensure cumulative_data is global

    # Fetch live data
    live_data = fetch_data()
    if live_data.empty:
        return  # Skip if data fetching failed

    live_data = add_indicators(live_data)

    # Update cumulative dataset with new live data
    cumulative_data = pd.concat([cumulative_data, live_data], ignore_index=True).drop_duplicates()

    # Prepare features for prediction using the cumulative dataset
    if len(cumulative_data) < 2:
        return  # Need at least two data points to predict

    # Prepare training data from the cumulative dataset
    X_train, y_train, X_val, y_val, scaler = prepare_training_data(cumulative_data)

    # Retrain the model with the cumulative data
    model = train_model(X_train, y_train, X_val, y_val)

    # Prepare features for the latest prediction
    live_features = scaler.transform(cumulative_data[['SMA_50', 'RSI', 'BB_Middle', 'BB_Upper', 'BB_Lower']].values[-1].reshape(1, -1))
    prediction = model.predict(live_features)[0][0]

    # Get the latest price
    current_price = live_data['close'].iloc[-1]

    # Simulate trades based on prediction
    if prediction > 0.6 and balance > 0:  # BUY
        btc_to_buy = min((balance * trade_size) / current_price, balance / current_price)
        balance -= btc_to_buy * current_price
        btc_balance += btc_to_buy
        trade_log.append({'action': 'BUY', 'price': current_price, 'btc': btc_to_buy, 'balance': balance})
        logging.info(f"BUY at {current_price:.2f}, BTC Balance: {btc_balance:.6f}, USD Balance: {balance:.2f}")

    elif prediction <= 0.4 and btc_balance > 0:  # SELL
        usd_to_add = btc_balance * current_price * trade_size
        btc_to_sell = btc_balance * trade_size
        balance += usd_to_add
        btc_balance -= btc_to_sell
        trade_log.append({'action': 'SELL', 'price': current_price, 'btc': btc_to_sell, 'balance': balance})
        logging.info(f"SELL at {current_price:.2f}, BTC Balance: {btc_balance:.6f}, USD Balance: {balance:.2f}")

    # Display overall balance in USD
    overall_balance_usd = balance + (btc_balance * current_price)
    print(f"Overall Balance in USD: {overall_balance_usd:.2f}")

    # Save the model after each trade
    joblib.dump(model, model_file_path)
    print("Prediction:", prediction)



# Step 6: Display results
def show_results():
    print("\n--- Trading Summary ---")
    for trade in trade_log:
        print(f"{trade['action']} | Price: {trade['price']:.2f} | BTC: {trade['btc']:.6f} | Balance: {trade['balance']:.2f}")
    print(f"\nFinal USD Balance: {balance:.2f}")
    print(f"Final BTC Balance: {btc_balance:.6f}")

# Main script
if __name__ == "__main__":
    print("Fetching historical data...")
    historical_data = fetch_data()
    historical_data = add_indicators(historical_data)

    print("Preparing training data...")
    X_train, y_train, X_val, y_val, scaler = prepare_training_data(historical_data)

    # Load existing model if it exists
    if os.path.exists(model_file_path):
        model = joblib.load(model_file_path)
        print("Loaded existing model.")
    else:
        print("Training new model...")
        model = train_model(X_train, y_train, X_val, y_val)

    print("Starting paper trading...")
    try:
        for _ in range(60):  # Run for 60 minutes
            paper_trade(model, scaler)
            time.sleep(60)  # Wait 1 minute for the next trade
    except Exception as e:
        logging.error(f"An error occurred during paper trading: {e}")
        print(f"An error occurred: {e}")
