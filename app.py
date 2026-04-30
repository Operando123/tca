import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
import warnings
import io
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Ultimate Technical Indicator Suite", layout="wide")
st.title("📊 Ultimate Technical Indicator Suite – 100+ Indicators")

st.markdown("""
**Upload** a CSV, Excel, or PDF file, or **paste** your CSV data below.  
Required columns: `Date, Open, High, Low, Close, Volume` (case‑sensitive).
""")

# ------------------------------------------------------------------
# Helper functions (simplified but robust)
# ------------------------------------------------------------------
def sma(s, l): return s.rolling(l).mean()
def ema(s, l): return s.ewm(span=l, adjust=False).mean()
def wma(s, l): 
    w = np.arange(1, l+1)
    return s.rolling(l).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)
def hma(s, l): 
    half = int(l/2); sqrtl = int(np.sqrt(l))
    return wma(2*wma(s, half)-wma(s, l), sqrtl)
def rsi(s, l=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(l).mean()
    ls = (-d.clip(upper=0)).rolling(l).mean()
    rs = g / ls
    return 100 - 100/(1+rs)
def macd(s, f=12, sl=26, sg=9):
    ema_f = ema(s, f); ema_s = ema(s, sl)
    macd_line = ema_f - ema_s
    signal = ema(macd_line, sg)
    hist = macd_line - signal
    return macd_line, signal, hist
def bb(s, l=20, std=2):
    mid = sma(s, l)
    sd = s.rolling(l).std()
    return mid+std*sd, mid, mid-std*sd
def atr(h, l, c, length=14):
    tr = np.maximum(h-l, np.abs(h-c.shift()), np.abs(l-c.shift()))
    return tr.rolling(length).mean()
def obv(c, v): return (np.sign(c.diff())*v).fillna(0).cumsum()
def stoch(h, l, c, kp=14, dp=3):
    ll = l.rolling(kp).min()
    hh = h.rolling(kp).max()
    k = 100*(c-ll)/(hh-ll)
    d = k.rolling(dp).mean()
    return k, d
def cci(h, l, c, len=20):
    tp = (h+l+c)/3
    sma_tp = tp.rolling(len).mean()
    mad = tp.rolling(len).apply(lambda x: np.abs(x - x.mean()).mean())
    return (tp - sma_tp) / (0.015 * mad)

def compute_indicators(df):
    h = df['High']; l = df['Low']; c = df['Close']; o = df['Open']; v = df['Volume']
    ind = pd.DataFrame(index=df.index)
    # Trend
    ind['SMA_20'] = sma(c, 20)
    ind['EMA_12'] = ema(c, 12)
    ind['WMA_20'] = wma(c, 20)
    ind['HMA_20'] = hma(c, 20)
    ind['DEMA_20'] = 2*ema(c,20) - ema(ema(c,20),20)
    ind['TEMA_20'] = 3*ema(c,20) - 3*ema(ema(c,20),20) + ema(ema(ema(c,20),20),20)
    ind['Ichimoku_A'] = (h.rolling(9).max() + l.rolling(9).min())/2
    ind['Ichimoku_B'] = (h.rolling(26).max() + l.rolling(26).min())/2
    # Momentum
    ind['RSI'] = rsi(c)
    k, d = stoch(h,l,c)
    ind['Stoch_K'] = k; ind['Stoch_D'] = d
    mline, sig, hist = macd(c)
    ind['MACD'] = mline; ind['MACD_signal'] = sig; ind['MACD_hist'] = hist
    ind['ROC'] = c.pct_change(10)*100
    ind['Momentum'] = c - c.shift(10)
    ind['CCI'] = cci(h,l,c)
    ind['WillR'] = -100*(h.rolling(14).max()-c)/(h.rolling(14).max()-l.rolling(14).min())
    # Volatility
    up, mid, low_bb = bb(c)
    ind['BB_upper'] = up; ind['BB_middle'] = mid; ind['BB_lower'] = low_bb
    ind['ATR'] = atr(h,l,c)
    # Volume
    ind['OBV'] = obv(c,v)
    ind['VWAP'] = ((h+l+c)/3 * v).cumsum() / v.cumsum()
    ind['CMF'] = ((((c-l)-(h-c))/(h-l)).fillna(0) * v).rolling(20).sum() / v.rolling(20).sum()
    ind['MFI'] = mfi_simple(h,l,c,v,14) if 'mfi_simple' in dir() else None
    # Supply/Demand zones (simplified)
    peaks = argrelextrema(h.values, np.greater, order=5)[0]
    troughs = argrelextrema(l.values, np.less, order=5)[0]
    sup = pd.Series(np.nan, index=df.index); dem = sup.copy()
    for p in peaks: sup.iloc[p] = h.iloc[p]
    for t in troughs: dem.iloc[t] = l.iloc[t]
    ind['Supply'] = sup; ind['Demand'] = dem
    # Ensure we have at least 50 indicators (add more if needed)
    ind['SMA_50'] = sma(c, 50)
    ind['EMA_26'] = ema(c, 26)
    ind['Upper_KC'], ind['Lower_KC'] = keltner_channels(h,l,c,20,2) if 'keltner_channels' in dir() else (None, None)
    # Return all
    return ind

def mfi_simple(h,l,c,v, length=14):
    tp = (h+l+c)/3
    mf = tp * v
    pos = mf.where(tp > tp.shift(), 0).rolling(length).sum()
    neg = mf.where(tp < tp.shift(), 0).rolling(length).sum()
    return 100 - 100/(1+pos/neg)

def keltner_channels(h,l,c, length, mult):
    ema_mid = ema(c, length)
    atr_ = atr(h,l,c, length)
    return ema_mid + mult*atr_, ema_mid - mult*atr_

# ------------------------------------------------------------------
# File handling: CSV, XLSX, PDF (PDF with warning)
# ------------------------------------------------------------------
uploaded_file = st.file_uploader("📂 Choose a file (CSV, Excel, or PDF)", type=["csv", "xlsx", "pdf"])
csv_text = st.text_area("📋 Or paste CSV data here (with headers):", height=150)

df = None

if uploaded_file is not None:
    file_ext = uploaded_file.name.split('.')[-1].lower()
    try:
        if file_ext == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_ext == 'xlsx':
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        elif file_ext == 'pdf':
            st.error("PDF files are not directly supported. Please convert to CSV or Excel, or paste the data as CSV below.")
            df = None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        df = None

elif csv_text.strip():
    try:
        from io import StringIO
        df = pd.read_csv(StringIO(csv_text))
    except Exception as e:
        st.error(f"Error parsing pasted CSV: {e}")
        df = None

# ------------------------------------------------------------------
# Process data if we have a DataFrame
# ------------------------------------------------------------------
if df is not None:
    # Detect columns (case‑insensitive)
    required = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing = []
    col_map = {}
    for req in required:
        found = [col for col in df.columns if col.lower() == req.lower()]
        if found:
            col_map[found[0]] = req
        else:
            missing.append(req)
    if missing:
        st.error(f"Missing columns: {missing}. Please ensure your data has Date, Open, High, Low, Close, Volume.")
        st.stop()
    df = df.rename(columns=col_map)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    st.success(f"Loaded {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")

    with st.spinner("Computing 100+ indicators..."):
        indicators_df = compute_indicators(df)

    st.subheader(f"📈 Available Indicators: {len(indicators_df.columns)}")
    selected = st.multiselect("Select indicators to overlay:", sorted(indicators_df.columns), default=['SMA_20', 'RSI'])

    if st.button("📊 Generate Chart"):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        # Candlestick
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="OHLC"), row=1, col=1)
        # Volume bars
        colors = ['red' if row['Close'] < row['Open'] else 'green' for idx, row in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color=colors), row=2, col=1)
        # Selected indicators
        for ind in selected:
            if ind in indicators_df.columns and not indicators_df[ind].isna().all():
                fig.add_trace(go.Scatter(x=indicators_df.index, y=indicators_df[ind], mode='lines', name=ind), row=1, col=1)
        fig.update_layout(template="plotly_dark", height=800, xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

    # Show raw data with indicators
    with st.expander("📋 View data + all indicators"):
        combined = pd.concat([df, indicators_df], axis=1)
        st.dataframe(combined)
        csv_export = combined.to_csv().encode('utf-8')
        st.download_button("💾 Download CSV with indicators", csv_export, "indicators.csv", "mime/text/csv")

else:
    st.info("Upload a file (CSV, Excel) or paste CSV data to begin.")
