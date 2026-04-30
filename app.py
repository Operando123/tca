import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta

st.set_page_config(page_title="Technical Indicator Pro", layout="wide")
st.title("📈 Technical Indicator Suite – 100+ Indicators")
st.markdown("""
Upload a CSV file with **OHLCV** data (Date, Open, High, Low, Close, Volume).  
The app computes **100+ technical indicators** using `pandas_ta` and lets you overlay them on interactive charts.
""")

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def load_data(uploaded_file):
    """Load CSV, parse dates, and ensure correct column names."""
    df = pd.read_csv(uploaded_file)
    # Try to find date column (case-insensitive)
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    if date_cols:
        df['Date'] = pd.to_datetime(df[date_cols[0]])
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        st.error("No date column found. Please include a column named 'Date'.")
        return None
    # Ensure required OHLCV columns
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}. Please rename columns to: Open, High, Low, Close, Volume.")
        return None
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df

def compute_indicators(df):
    """Compute all available indicators from pandas_ta."""
    # Use pandas_ta to compute all indicators (it returns a DataFrame)
    # We'll do it step by step to avoid huge memory, but we can compute a wide set.
    # For demonstration, we compute a comprehensive list of popular ones.
    indicators = {}
    
    # Overlap (Trend) Indicators
    indicators['SMA_20'] = ta.sma(df['Close'], length=20)
    indicators['SMA_50'] = ta.sma(df['Close'], length=50)
    indicators['EMA_12'] = ta.ema(df['Close'], length=12)
    indicators['EMA_26'] = ta.ema(df['Close'], length=26)
    indicators['WMA_20'] = ta.wma(df['Close'], length=20)
    indicators['HMA_20'] = ta.hma(df['Close'], length=20)
    indicators['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    
    # Momentum Indicators
    indicators['RSI_14'] = ta.rsi(df['Close'], length=14)
    indicators['StochK'] = ta.stoch(df['High'], df['Low'], df['Close']).iloc[:,0] if ta.stoch(df['High'], df['Low'], df['Close']) is not None else None
    indicators['StochD'] = ta.stoch(df['High'], df['Low'], df['Close']).iloc[:,1] if ta.stoch(df['High'], df['Low'], df['Close']) is not None else None
    indicators['WilliamsR'] = ta.willr(df['High'], df['Low'], df['Close'], length=14)
    indicators['MFI_14'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
    # MACD
    macd = ta.macd(df['Close'])
    if macd is not None:
        indicators['MACD'] = macd.iloc[:,0]
        indicators['MACD_signal'] = macd.iloc[:,1]
        indicators['MACD_hist'] = macd.iloc[:,2]
    else:
        indicators['MACD'] = indicators['MACD_signal'] = indicators['MACD_hist'] = None
        
    # Bollinger Bands
    bb = ta.bbands(df['Close'], length=20, std=2)
    if bb is not None:
        indicators['BB_upper'] = bb.iloc[:,0]
        indicators['BB_middle'] = bb.iloc[:,1]
        indicators['BB_lower'] = bb.iloc[:,2]
    else:
        indicators['BB_upper'] = indicators['BB_middle'] = indicators['BB_lower'] = None
    
    # Volume Indicators
    indicators['OBV'] = ta.obv(df['Close'], df['Volume'])
    indicators['CMF_20'] = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=20)
    
    # Volatility Indicators
    indicators['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    indicators['BB_width'] = (indicators['BB_upper'] - indicators['BB_lower']) / indicators['BB_middle'] if indicators['BB_middle'] is not None else None
    
    # Add more to reach 100+ : we'll add many from pandas_ta's full suite
    # Use a loop to generate all available indicators from ta (but careful about performance)
    # For safety, we'll add a set of prominent ones.
    # Let's add another 50+ using ta's built-in methods
    
    # Trend indicators
    indicators['ADX_14'] = ta.adx(df['High'], df['Low'], df['Close'], length=14).iloc[:,0] if ta.adx(df['High'], df['Low'], df['Close']) is not None else None
    indicators['PSAR'] = ta.psar(df['High'], df['Low'], close=df['Close'])
    indicators['KC_upper'], indicators['KC_lower'] = None, None
    kc = ta.kc(df['High'], df['Low'], df['Close'], length=20)
    if kc is not None:
        indicators['KC_upper'] = kc.iloc[:,0]
        indicators['KC_lower'] = kc.iloc[:,2]
    # More momentum
    indicators['CCI_20'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)
    indicators['TSI'] = ta.tsi(df['Close'])
    indicators['UO'] = ta.uo(df['High'], df['Low'], df['Close'])
    # Volume
    indicators['ADI'] = ta.ad(df['High'], df['Low'], df['Close'], df['Volume'])  # Accumulation/Distribution
    indicators['EOM'] = ta.eom(df['High'], df['Low'], df['Close'], df['Volume'])  # Ease of Movement
    # Other
    indicators['KAMA_20'] = ta.kama(df['Close'], length=20)
    indicators['ZLEMA_20'] = ta.zlma(df['Close'], length=20)
    indicators['FRAMA_20'] = ta.frama(df['Close'], length=20)
    
    # Add all indicators from pandas_ta's "indicators" list (includes many more)
    # This is a safe way to compute everything without overloading
    # But we'll only compute the ones that are numeric series
    all_ta = ta.indicators()
    # For each indicator, we could compute, but to keep performance, we'll just list the ones we have.
    # Actually we already have ~40 distinct. Let's use the full ta extension: df.ta.indicators()
    # However, that may be heavy. We'll do a selective bulk.
    # Instead, we can use df.ta.strategy to compute many at once.
    # Let's use the built-in "All" strategy.
    try:
        # This computes dozens of indicators quickly
        df_with_indicators = df.ta.strategy("All")
        # Extract all columns that were added (they start with prefixes like "RSI_", "MACD_", etc.)
        # But to avoid duplicate work, we'll just merge with our existing.
        for col in df_with_indicators.columns:
            if col not in df.columns and not col.startswith('date'):
                indicators[col] = df_with_indicators[col]
    except Exception as e:
        st.warning(f"Some indicators could not be computed automatically: {e}")
    
    # Remove any None entries
    indicators = {k: v for k, v in indicators.items() if v is not None}
    # Convert to DataFrame for easier handling
    ind_df = pd.DataFrame(indicators, index=df.index)
    return ind_df

def plot_chart(df, indicators_df, selected_indicators):
    """Create interactive Plotly chart with price, volume, and selected indicators."""
    # Main price and volume subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, 
                        row_heights=[0.7, 0.3])
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                  low=df['Low'], close=df['Close'], name="OHLC"),
                  row=1, col=1)
    
    # Volume bars
    colors = ['red' if row['Close'] < row['Open'] else 'green' for idx, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color=colors),
                  row=2, col=1)
    
    # Add selected indicators
    for ind in selected_indicators:
        if ind in indicators_df.columns:
            fig.add_trace(go.Scatter(x=indicators_df.index, y=indicators_df[ind],
                                     mode='lines', name=ind, line=dict(width=1.5)),
                          row=1, col=1)
    
    # Layout
    fig.update_layout(title="Technical Analysis Chart", xaxis_title="Date",
                      yaxis_title="Price", template="plotly_dark", height=800)
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload CSV (Date, Open, High, Low, Close, Volume)", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is not None:
        st.success(f"Loaded {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")
        
        with st.spinner("Computing 100+ indicators... (might take a moment)"):
            indicators_df = compute_indicators(df)
        
        st.subheader(f"📊 Available Indicators: {len(indicators_df.columns)}")
        st.caption("The following indicators have been computed. You can select any to overlay on the chart.")
        
        # Multi-select for indicators
        selected = st.multiselect("Select indicators to display on chart:",
                                  options=sorted(indicators_df.columns),
                                  default=['SMA_20', 'SMA_50'])
        
        if st.button("📈 Refresh Chart"):
            plot_chart(df, indicators_df, selected)
        
        # Show data table
        with st.expander("🔍 View Raw Data + Indicators"):
            combined = pd.concat([df, indicators_df], axis=1)
            st.dataframe(combined)
        
        # Download button
        csv = combined.to_csv().encode('utf-8')
        st.download_button("💾 Download Data with Indicators", csv, "technical_indicators.csv", "text/csv")
else:
    # Provide sample data
    st.info("No file uploaded. Use a sample CSV with columns: Date, Open, High, Low, Close, Volume.")
    if st.button("📥 Download Sample CSV"):
        sample = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=100),
            "Open": np.random.randn(100).cumsum() + 100,
            "High": np.random.randn(100).cumsum() + 102,
            "Low": np.random.randn(100).cumsum() + 98,
            "Close": np.random.randn(100).cumsum() + 100,
            "Volume": np.random.randint(1000, 10000, 100)
        })
        sample['High'] = sample[['Open','Close','High']].max(axis=1)
        sample['Low'] = sample[['Open','Close','Low']].min(axis=1)
        csv = sample.to_csv(index=False)
        st.download_button("Download sample CSV", csv, "sample_ohlc.csv", "text/csv")
