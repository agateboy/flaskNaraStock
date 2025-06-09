from flask import Flask, request, jsonify, abort
from flask_cors import CORS
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import numpy as np
import json
import ta
import os
import joblib
from joblib import load
from flask_caching import Cache

app = Flask(__name__)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})

# Konfigurasi cache: gunakan SimpleCache (in-memory)
cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': 300})

START_DATE = datetime.strptime('2015-01-01', '%Y-%m-%d')
END_DATE = datetime.today()
WINDOW_SIZE_REG = 270
FEATURES_PRICE = ['Close', 'High', 'Low', 'Open']
FEATURES_FIBO = ['FiboDist_0', 'FiboDist_23.6', 'FiboDist_38.2', 'FiboDist_50', 'FiboDist_61.8', 'FiboDist_100']
FEATURES_TECH = [
    'SMA_10', 'SMA_20', 'SMA_50',
    'EMA_10', 'EMA_20', 'EMA_50',
    'RSI_14', 'MACD', 'MACD_signal',
    'Stochastic_k', 'Stochastic_d', 'CCI',
    'ATR_14', 'Bollinger_High', 'Bollinger_Low', 'Bollinger_Mid', 'StdDev_20',
    'Log_Return', 'ROC_5',
    'Body', 'Upper_Shadow', 'Lower_Shadow', 'Bullish_Engulfing'
]

# List of symbols to download data for and cache
SYMBOLS = ['EURUSD=X', 'NZDUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X']  # Extend or change as needed

def download_and_save_data(symbols):
    for symbol in symbols:
        try:
            data = yf.download(symbol, start=START_DATE.strftime('%Y-%m-%d'), end=END_DATE.strftime('%Y-%m-%d'))
            if not data.empty:
                filename = f"{symbol.replace('=', '_')}_data.json"
                data.reset_index(inplace=True)
                data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
                data.to_json(filename, orient='records', date_format='iso')
                print(f"Data saved for {symbol} at {filename}")
            else:
                print(f"No data downloaded for {symbol}")
        except Exception as e:
            print(f"Error downloading data for {symbol}: {e}")

# Download and save data at startup
download_and_save_data(SYMBOLS)

def add_technical_indicators(data):
    data = data.sort_index()
    data['SMA_10'] = ta.trend.sma_indicator(data['Close'], window=10)
    data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
    data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
    data['EMA_10'] = ta.trend.ema_indicator(data['Close'], window=10)
    data['EMA_20'] = ta.trend.ema_indicator(data['Close'], window=20)
    data['EMA_50'] = ta.trend.ema_indicator(data['Close'], window=50)
    data['RSI_14'] = ta.momentum.rsi(data['Close'], window=14)
    data['MACD'] = ta.trend.macd(data['Close'])
    data['MACD_signal'] = ta.trend.macd_signal(data['Close'])
    data['Stochastic_k'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'], window=14)
    data['Stochastic_d'] = ta.momentum.stoch_signal(data['High'], data['Low'], data['Close'], window=14)
    data['CCI'] = ta.trend.cci(data['High'], data['Low'], data['Close'], window=20)
    data['ATR_14'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=14)
    data['Bollinger_High'] = ta.volatility.bollinger_hband(data['Close'], window=20)
    data['Bollinger_Low'] = ta.volatility.bollinger_lband(data['Close'], window=20)
    data['Bollinger_Mid'] = ta.volatility.bollinger_mavg(data['Close'], window=20)
    data['StdDev_20'] = data['Close'].rolling(window=20).std()
    data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
    data['ROC_5'] = data['Close'].pct_change(5)
    data['Body'] = abs(data['Close'] - data['Open'])
    data['Upper_Shadow'] = data['High'] - data[['Close', 'Open']].max(axis=1)
    data['Lower_Shadow'] = data[['Close', 'Open']].min(axis=1) - data['Low']
    data['Bullish_Engulfing'] = ((data['Close'] > data['Open'].shift(1)) & (data['Open'] < data['Close'].shift(1))).astype(int)
    data.fillna(method='bfill', inplace=True)
    data.fillna(method='ffill', inplace=True)
    return data

def dynamic_fibo_levels(row):
    if pd.isna(row['High_prev1']) or pd.isna(row['Low_prev1']):
        return None
    high = row['High_prev1']
    low = row['Low_prev1']
    trend = row.get('TrendYesterday', 0)  # Default to 0 if not available
    start, end = (low, high) if trend == 1 else (high, low)
    levels = {
        0: start,
        23.6: start + 0.236 * (end - start),
        38.2: start + 0.382 * (end - start),
        50.0: start + 0.5 * (end - start),
        61.8: start + 0.618 * (end - start),
        100: end
    }
    return levels

def fibo_distance_features(row):
    if pd.isna(row['Close']) or row['FiboLevels'] is None:
        return pd.Series([np.nan]*6)
    close = row['Close']
    levels = row['FiboLevels']
    return pd.Series([abs(close - levels[lvl]) for lvl in [0, 23.6, 38.2, 50.0, 61.8, 100]])

@app.route('/predict', methods=['GET'])
def predict():
    symbol = request.args.get('symbol', 'NZDUSD=X')
    model_path = f'pred/{symbol}/model_reg_best.h5'
    
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model tidak ditemukan.'}), 404

    model_reg = load_model(model_path)
    scaler_price = load(f'pred/{symbol}/scaler_price.save')
    scaler_fibo = load(f'pred/{symbol}/scaler_fibo.save')
    scaler_tech = load(f'pred/{symbol}/scaler_tech.save')

    # Use cached JSON file for price data
    filename = f"{symbol.replace('=', '_')}_data.json"
    if not os.path.isfile(filename):
        return jsonify({'error': 'Data tidak tersedia.'}), 404

    try:
        datap = pd.read_json(filename)
    except Exception:
        return jsonify({'error': 'Gagal membaca data cached JSON.'}), 500

    datap = datap[['Close', 'High', 'Low', 'Open', 'Volume']]

    # Drop volume jika semua 0
    if 'Volume' in datap.columns and datap['Volume'].nunique() == 1 and datap['Volume'].iloc[0] == 0:
        datap.drop('Volume', axis=1, inplace=True)

    # Feature engineering
    datap['Target'] = (datap['Close'].shift(-1) > datap['Close']).astype(int)
    datap['High_prev1'] = datap['High'].shift(1)
    datap['Low_prev1'] = datap['Low'].shift(1)
    datap['Close_prev1'] = datap['Close'].shift(1)
    datap['Close_prev2'] = datap['Close'].shift(2)
    datap['TrendYesterday'] = (datap['Close_prev1'] > datap['Close_prev2']).astype(int)

    datap = add_technical_indicators(datap)
    datap['FiboLevels'] = datap.apply(dynamic_fibo_levels, axis=1)
    fibo_dists = datap.apply(fibo_distance_features, axis=1)
    fibo_dists.columns = ['FiboDist_0', 'FiboDist_23.6', 'FiboDist_38.2', 'FiboDist_50', 'FiboDist_61.8', 'FiboDist_100']
    datap = pd.concat([datap, fibo_dists], axis=1)

    # Pastikan tidak ada NaN pada fitur input
    required_cols = FEATURES_PRICE + FEATURES_FIBO + FEATURES_TECH
    datap = datap.dropna(subset=required_cols)

    if len(datap) < WINDOW_SIZE_REG:
        return jsonify({'error': f'Data tidak cukup untuk membuat sequence, butuh minimal {WINDOW_SIZE_REG} baris.'}), 400

    # Ambil window terakhir untuk prediksi
    input_data = datap[-WINDOW_SIZE_REG:]

    # Scaling
    scaled_price = scaler_price.transform(input_data[FEATURES_PRICE])
    scaled_fibo = scaler_fibo.transform(input_data[FEATURES_FIBO])
    scaled_tech = scaler_tech.transform(input_data[FEATURES_TECH])

    X_input = np.hstack([scaled_price, scaled_fibo, scaled_tech])
    X_input = X_input.reshape(1, WINDOW_SIZE_REG, X_input.shape[1])

    # Prediksi scaled output
    y_pred_scaled = model_reg.predict(X_input)

    # Buat dummy kolom open untuk inverse transform (scaler_price expect 4 fitur)
    dummy_open = np.zeros((y_pred_scaled.shape[0], 1))
    y_pred_full = np.hstack([y_pred_scaled, dummy_open])

    # Inverse transform untuk price prediction (close, high, low)
    y_pred = scaler_price.inverse_transform(y_pred_full)[:, :3]

    pred_close, pred_high, pred_low = y_pred[0]

    return jsonify({
        'predicted_close': float(pred_close),
        'predicted_high': float(pred_high),
        'predicted_low': float(pred_low),
    })

@app.route('/weekly', methods=['GET'])
@cache.cached(query_string=True)
def weekly():
    model_name = request.args.get('symbol')
    if not model_name:
        return jsonify({'error': 'Parameter symbol harus diberikan.'}), 400

    filename = f"{model_name}_prediksi.json"
    file_path = os.path.join(os.getcwd(), filename)

    if not os.path.isfile(file_path):
        abort(404, description=f"File {filename} tidak ditemukan.")

    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return jsonify(data)
    except json.JSONDecodeError:
        abort(500, description="Format JSON tidak valid dalam file.")

@app.route('/api/forex')
@cache.cached(query_string=True)
def get_forex():
    symbol = request.args.get('symbol', 'EURUSD=X')
    period = request.args.get('period', '4d')  # default 4 days
    interval = request.args.get('interval', '1h')  # default 1 hour
    # Fetch data from yfinance
    data = yf.download(symbol, period=period, interval=interval)
    if data.empty:
        return jsonify({'error': 'Data tidak tersedia.'}), 404
    data.reset_index(inplace=True)
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    # Ensure datetime column is string
    if 'Datetime' in data.columns:
        data['Datetime'] = data['Datetime'].astype(str)
    elif 'Date' in data.columns:
        data['Datetime'] = data['Date'].astype(str)
    else:
        return jsonify({'error': 'Kolom tanggal tidak ditemukan.'}), 500
    for col in ['Open', 'High', 'Low', 'Close']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data[['Datetime', 'Open', 'High', 'Low', 'Close']].dropna()
    return jsonify(data.to_dict(orient='records'))
@app.route('/api/zones')
@cache.cached(query_string=True)
def get_zones():
    symbol = request.args.get('symbol', 'EURUSD=X')
    filename = f"{symbol.replace('=', '_')}_data.json"
    
    if not os.path.isfile(filename):
        return jsonify({'error': 'Data tidak tersedia.'}), 404

    try:
        with open(filename, 'r') as file:
            df = pd.read_json(file)
    except json.JSONDecodeError:
        return jsonify({'error': 'Format JSON tidak valid dalam file.'}), 500

    # Debug: Print the columns of the DataFrame
    print("DataFrame columns:", df.columns.tolist())

    # Check if 'Datetime' or 'Date' exists
    if 'Datetime' not in df.columns:
        if 'Date' in df.columns:
            df.rename(columns={'Date': 'Datetime'}, inplace=True)
        else:
            return jsonify({'error': 'Kolom tanggal tidak ditemukan.'}), 500

    df['Datetime'] = df['Datetime'].astype(str)

    zones = []
    for i in range(2, len(df) - 2):
        if df['Low'][i] < df['Low'][i-1] and df['Low'][i] < df['Low'][i+1]:
            zones.append({'type': 'FBuy', 'Datetime': df['Datetime'][i], 'Price': float(df['Low'][i])})
        if df['High'][i] > df['High'][i-1] and df['High'][i] > df['High'][i+1]:
            zones.append({'type': 'FSell', 'Datetime': df['Datetime'][i], 'Price': float(df['High'][i])})

    return jsonify(zones)


@app.route('/api/zones/counts')
@cache.cached(query_string=True)
def get_zone_counts():
    symbol = request.args.get('symbol', 'EURUSD=X')
    filename = f"{symbol.replace('=', '_')}_data.json"
    
    if not os.path.isfile(filename):
        return jsonify({})

    try:
        with open(filename, 'r') as file:
            df = pd.read_json(file)
    except json.JSONDecodeError:
        return jsonify({})

    counts = {'FBuy': 0, 'FSell': 0, 'DBuy': 0, 'DSell': 0}

    for i in range(2, len(df) - 2):
        if df['Low'][i] < df['Low'][i-1] and df['Low'][i] < df['Low'][i+1]:
            counts['FBuy'] += 1
        if df['High'][i] > df['High'][i-1] and df['High'][i] > df['High'][i+1]:
            counts['FSell'] += 1
        if df['Low'][i] < df['Low'][i-2] and df['Low'][i] < df['Low'][i+2]:
            counts['DBuy'] += 1
        if df['High'][i] > df['High'][i-2] and df['High'][i] > df['High'][i+2]:
            counts['DSell'] += 1

    total = sum(counts.values())
    if total == 0:
        total = 1  # avoid division by zero

    percentages = {k + '_pct': round((v / total) * 100, 2) for k, v in counts.items()}
    max_count = max(counts.values()) if counts.values() else 1
    if max_count == 0:
        max_count = 1

    strengths = {k + '_strength': round((v / max_count) * 100, 2) for k, v in counts.items()}

    return jsonify({**counts, **percentages, **strengths})

@app.route('/api/indicator')
@cache.cached(query_string=True)
def get_indicator():
    symbol = request.args.get('symbol', 'EURUSD=X')
    ind_type = request.args.get('type', 'SMA')
    period = request.args.get('period', '4d')
    interval = request.args.get('interval', '1h')  # default 1 hour

    # Fetch data from yfinance
    df = yf.download(symbol, period=period, interval=interval)
    if df.empty:
        return jsonify([])

    df.reset_index(inplace=True)
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    if 'Datetime' not in df.columns:
        if 'Date' in df.columns:
            df.rename(columns={'Date': 'Datetime'}, inplace=True)
        else:
            return jsonify([])

    df['Datetime'] = df['Datetime'].astype(str)

    days = 0
    try:
        days = int(period.lower().replace('d', ''))
    except Exception:
        days = 4  # default fallback

    # Set window size proportional to days
    window = max(2, min(14, days * 6))  # e.g., days*6 hours (1h interval), limited 2-14

    if ind_type.upper() == 'SMA':
        df['Value'] = df['Close'].rolling(window=window).mean()
    elif ind_type.upper() == 'EMA':
        df['Value'] = df['Close'].ewm(span=window, adjust=False).mean()
    else:
        return jsonify([])

    df.dropna(inplace=True)
    result = df[['Datetime', 'Value']].to_dict(orient='records')

    return jsonify(result)


@app.route('/')
@cache.cached(query_string=True)
def index():
    return 'Forex API Server is running'

@app.route('/favicon.ico')
@cache.cached(query_string=True)
def favicon():
    return '', 204

if __name__ == '__main__':
    app.run()

