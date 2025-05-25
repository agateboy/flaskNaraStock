from flask import Flask, jsonify, request
import yfinance as yf
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
