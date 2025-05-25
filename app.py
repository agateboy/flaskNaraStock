from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from tensorflow import keras
import joblib
import os

app = Flask(__name__)
# CORS(app)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})

symbol = 'AUDUSD=X'
start_date = datetime.strptime('2015-01-01', '%Y-%m-%d')
end_date = datetime.today()

print("📥 Mengambil data dari Yahoo Finance...")
df = yf.download(symbol, start=start_date, end=end_date)
df.reset_index(inplace=True)
df['Date'] = pd.to_datetime(df['Date'])

df['price range'] = df['Close'] - df['Open']
df['highlow range'] = df['High'] - df['Low']
df['MA_5'] = df['Close'].rolling(window=5).mean()

def calculate_rsi(data, window):
    diff = data.diff()
    up, down = diff.copy(), diff.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    avg_gain = up.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = down.abs().ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI_14'] = calculate_rsi(df['Close'], 14)

def get_last_available_trading_day(df, target_date):
    date = target_date
    available_dates = set(df['Date'].dt.date)
    while date.date() not in available_dates:
        date -= timedelta(days=1)
        if date.date() < min(available_dates):
            return None
    return date.date()

# --- Load scaler dinamis dari path (bisa disesuaikan namingnya) ---
def load_scaler(scaler_path):
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    return None

# --- Endpoint prediksi dinamis berdasarkan model dan scaler path dari query param ---
@app.route('/predict', methods=['GET'])
def predict():
    model_name = request.args.get('model')
    if not model_name:
        return jsonify({'error': 'Parameter model harus diberikan.'}), 400

    # Contoh path model dan scaler, sesuaikan dengan lokasi file kamu
    model_path = f'{model_name}.h5'
    scaler_path = f'{model_name}_scaler.save'

    if not os.path.exists(model_path):
        return jsonify({'error': f'Model file {model_path} tidak ditemukan.'}), 404
    if not os.path.exists(scaler_path):
        return jsonify({'error': f'Scaler file {scaler_path} tidak ditemukan.'}), 404

    try:
        model = keras.models.load_model(model_path)
    except Exception as e:
        return jsonify({'error': f'Gagal memuat model: {str(e)}'}), 500

    try:
        scaler = load_scaler(scaler_path)
        if scaler is None:
            return jsonify({'error': 'Scaler tidak ditemukan atau gagal dimuat.'}), 500
    except Exception as e:
        return jsonify({'error': f'Gagal memuat scaler: {str(e)}'}), 500

    # Gunakan tanggal hari ini untuk prediksi otomatis
    today = datetime.today()
    prediction_reference_date = today - timedelta(days=1)
    reference_day = get_last_available_trading_day(df, prediction_reference_date)

    if not reference_day:
        return jsonify({'error': 'Tidak ada data tersedia sebelum tanggal hari ini.'}), 404

    ref_data = df[df['Date'].dt.date == reference_day]
    features = ['price range', 'highlow range', 'MA_5', 'RSI_14']

    if not all(feature in ref_data.columns for feature in features):
        return jsonify({'error': 'Fitur tidak lengkap dalam data.'}), 500

    input_data = ref_data[features].iloc[0].values.reshape(1, -1)
    try:
        input_scaled = scaler.transform(input_data)
        input_reshaped = input_scaled.reshape((1, 1, input_scaled.shape[1]))
        prediction_prob = model.predict(input_reshaped)[0][0]
        prediction_label = 'naik' if prediction_prob >= 0.5 else 'turun'

        return jsonify({
            'input_date': today.strftime('%Y-%m-%d'),
            'reference_date': reference_day.strftime('%Y-%m-%d'),
            'model_used': model_name,
            'prediction': prediction_label,
            'probability_up': round(float(prediction_prob), 4)
        })

    except Exception as e:
        return jsonify({'error': f'Error saat scaling atau prediksi: {str(e)}'}), 500


def period_to_str(days):
    return f"{days}d"

@app.route('/api/forex')
def get_forex():
    symbol = request.args.get('symbol', 'EURUSD=X')
    period = request.args.get('period', '4d')  # default 4 hari
    data = yf.download(symbol, period=period, interval='1h')
    data.reset_index(inplace=True)

    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    data['Datetime'] = data['Datetime'].astype(str)

    for col in ['Open', 'High', 'Low', 'Close']:
        data[col] = data[col].astype(float)

    data = data[['Datetime', 'Open', 'High', 'Low', 'Close']]
    return jsonify(data.to_dict(orient='records'))

@app.route('/api/zones')
def get_zones():
    symbol = request.args.get('symbol', 'EURUSD=X')
    period = request.args.get('period', '4d')
    df = yf.download(symbol, period=period, interval='1h')
    df.reset_index(inplace=True)

    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df['Datetime'] = df['Datetime'].astype(str)

    zones = []
    for i in range(2, len(df) - 2):
        if df['Low'][i] < df['Low'][i-1] and df['Low'][i] < df['Low'][i+1]:
            zones.append({'type': 'FBuy', 'Datetime': df['Datetime'][i], 'Price': df['Low'][i]})
        if df['High'][i] > df['High'][i-1] and df['High'][i] > df['High'][i+1]:
            zones.append({'type': 'FSell', 'Datetime': df['Datetime'][i], 'Price': df['High'][i]})

    return jsonify(zones)

@app.route('/api/zones/counts')
def get_zone_counts():
    symbol = request.args.get('symbol', 'EURUSD=X')
    period = request.args.get('period', '4d')
    df = yf.download(symbol, period=period, interval='1h')
    df.reset_index(inplace=True)

    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

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

    total = sum(counts.values()) or 1
    percentages = {k + '_pct': round((v / total) * 100, 2) for k, v in counts.items()}
    strengths = {k + '_strength': round((v / max(counts.values() or [1])) * 100, 2) for k, v in counts.items()}

    return jsonify({**counts, **percentages, **strengths})

@app.route('/api/indicator')
def get_indicator():
    symbol = request.args.get('symbol', 'EURUSD=X')
    ind_type = request.args.get('type', 'SMA')
    period = request.args.get('period', '4d')

    df = yf.download(symbol, period=period, interval='1h')
    df.reset_index(inplace=True)

    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df['Datetime'] = df['Datetime'].astype(str)

    # Sesuaikan window SMA/EMA dengan panjang period dalam jam
    days = int(period.replace('d', ''))
    window = min(14, max(2, days * 24 // 4))  # contoh: window = days * 6 jam, minimal 2

    if ind_type == 'SMA':
        df['Value'] = df['Close'].rolling(window=window).mean()
    elif ind_type == 'EMA':
        df['Value'] = df['Close'].ewm(span=window, adjust=False).mean()
    else:
        return jsonify([])

    df.dropna(inplace=True)
    result = df[['Datetime', 'Value']].to_dict(orient='records')
    return jsonify(result)

@app.route('/')
def index():
    return 'Forex API Server is running'

@app.route('/favicon.ico')
def favicon():
    return '', 204


if __name__ == '__main__':
    app.run()


