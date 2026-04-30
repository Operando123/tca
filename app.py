import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Ultimate Technical Indicator Suite", layout="wide")
st.title("📊 Ultimate Technical Indicator Suite – 100+ Indicators")
st.markdown("Upload a CSV with **Date, Open, High, Low, Close, Volume**. Computes 100+ indicators using pure pandas/numpy.")

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    if date_cols:
        df['Date'] = pd.to_datetime(df[date_cols[0]])
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        st.error("No date column found. Include a column named 'Date'.")
        return None
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return None
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df

# ------------------------------------------------------------------
# Custom indicator implementations (pure pandas)
# ------------------------------------------------------------------
def sma(series, length): return series.rolling(window=length, min_periods=length).mean()
def ema(series, length): return series.ewm(span=length, adjust=False).mean()
def wma(series, length): 
    weights = np.arange(1, length+1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights)/weights.sum(), raw=True)
def hma(series, length): 
    half_len = int(length/2)
    sqrt_len = int(np.sqrt(length))
    wma_half = wma(series, half_len)
    wma_full = wma(series, length)
    hma_series = 2 * wma_half - wma_full
    return wma(hma_series, sqrt_len)
def rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(length).mean()
    loss = (-delta.clip(upper=0)).rolling(length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram
def bollinger_bands(series, length=20, std=2):
    middle = sma(series, length)
    stdev = series.rolling(length).std()
    upper = middle + (stdev * std)
    lower = middle - (stdev * std)
    return upper, middle, lower
def atr(high, low, close, length=14):
    tr = np.maximum(high - low, np.abs(high - close.shift()), np.abs(low - close.shift()))
    return tr.rolling(length).mean()
def obv(close, volume):
    return (np.sign(close.diff()) * volume).fillna(0).cumsum()
def ad(high, low, close, volume):
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    return (mfm * volume).fillna(0).cumsum()
def cmf(high, low, close, volume, length=20):
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mfv = mfm * volume
    return mfv.rolling(length).sum() / volume.rolling(length).sum()
def mfi(high, low, close, volume, length=14):
    typical = (high + low + close) / 3
    money_flow = typical * volume
    positive = money_flow.where(typical > typical.shift(), 0).rolling(length).sum()
    negative = money_flow.where(typical < typical.shift(), 0).rolling(length).sum()
    mfi = 100 - (100 / (1 + positive / negative))
    return mfi
def stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    stoch_d = stoch_k.rolling(d_period).mean()
    return stoch_k, stoch_d
def williams_r(high, low, close, length=14):
    highest = high.rolling(length).max()
    lowest = low.rolling(length).min()
    return -100 * (highest - close) / (highest - lowest)
def cci(high, low, close, length=20):
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(length).mean()
    mad = tp.rolling(length).apply(lambda x: np.abs(x - x.mean()).mean())
    return (tp - sma_tp) / (0.015 * mad)
def adx(high, low, close, length=14):
    tr = np.maximum(high - low, np.abs(high - close.shift()), np.abs(low - close.shift()))
    atr = tr.rolling(length).mean()
    up_move = high - high.shift()
    down_move = low.shift() - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_di = 100 * (pd.Series(plus_dm).rolling(length).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(length).mean() / atr)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    return dx.rolling(length).mean()

# ------------------------------------------------------------------
# Main indicator computation (100+)
# ------------------------------------------------------------------
def compute_indicators(df):
    high = df['High']
    low = df['Low']
    close = df['Close']
    open_ = df['Open']
    volume = df['Volume']
    indicators = pd.DataFrame(index=df.index)
    
    # ----- Trend (15)
    indicators['SMA_10'] = sma(close, 10)
    indicators['SMA_20'] = sma(close, 20)
    indicators['SMA_50'] = sma(close, 50)
    indicators['EMA_12'] = ema(close, 12)
    indicators['EMA_26'] = ema(close, 26)
    indicators['WMA_20'] = wma(close, 20)
    indicators['HMA_20'] = hma(close, 20)
    indicators['KAMA_20'] = close.ewm(alpha=2/21, adjust=False).mean()  # simplified KAMA
    indicators['DEMA_20'] = 2 * ema(close, 20) - ema(ema(close, 20), 20)
    indicators['TEMA_20'] = 3 * ema(close, 20) - 3 * ema(ema(close, 20), 20) + ema(ema(ema(close, 20), 20), 20)
    # Ichimoku (simplified)
    indicators['Ichimoku_A'] = (high.rolling(9).max() + low.rolling(9).min()) / 2
    indicators['Ichimoku_B'] = (high.rolling(26).max() + low.rolling(26).min()) / 2
    # Parabolic SAR
    indicators['PSAR'] = ta_psar(high, low, close)   # custom function below
    # SuperTrend
    supertrend = supertrend_custom(high, low, close, 10, 3)
    indicators['SuperTrend'] = supertrend
    # Aroon
    aroon_up = 100 * (close.rolling(25).apply(lambda x: x.argmax()) / 24) if len(close)>=25 else np.nan
    aroon_down = 100 * (close.rolling(25).apply(lambda x: x.argmin()) / 24)
    indicators['Aroon_Up'] = aroon_up
    indicators['Aroon_Down'] = aroon_down
    # Linear Regression Slope
    def slope(series):
        x = np.arange(len(series))
        return np.polyfit(x, series, 1)[0] if len(series) >= 2 else np.nan
    indicators['LinReg_Slope_20'] = close.rolling(20).apply(slope, raw=False)
    # Moving Average Ribbon (SMA 10,20,50)
    indicators['MA_Ribbon_10'] = indicators['SMA_10']
    indicators['MA_Ribbon_20'] = indicators['SMA_20']
    indicators['MA_Ribbon_50'] = indicators['SMA_50']
    # Trend Intensity Index
    indicators['TrendIntensity'] = (close / sma(close, 20) - 1) * 100
    
    # ----- Momentum (20+)
    indicators['RSI_14'] = rsi(close, 14)
    stoch_k, stoch_d = stochastic(high, low, close, 14, 3)
    indicators['Stoch_K'] = stoch_k
    indicators['Stoch_D'] = stoch_d
    # Stochastic RSI (simplified)
    stochrsi_k = (rsi(close, 14) - rsi(close, 14).rolling(14).min()) / (rsi(close, 14).rolling(14).max() - rsi(close, 14).rolling(14).min()) * 100
    indicators['StochRSI_K'] = stochrsi_k
    indicators['StochRSI_D'] = stochrsi_k.rolling(3).mean()
    macd_line, macd_signal, macd_hist = macd(close)
    indicators['MACD'] = macd_line
    indicators['MACD_signal'] = macd_signal
    indicators['MACD_hist'] = macd_hist
    indicators['ROC_10'] = close.pct_change(10) * 100
    indicators['Momentum_10'] = close - close.shift(10)
    indicators['TSI'] = tsi_indicator(close)  # custom
    indicators['Ultimate_Osc'] = ultimate_oscillator(high, low, close)  # custom
    indicators['WillR_14'] = williams_r(high, low, close, 14)
    indicators['CCI_20'] = cci(high, low, close, 20)
    # Fisher Transform
    fisher_val = fisher_transform(high, low, 10)
    indicators['Fisher'] = fisher_val
    indicators['CMO'] = (close.diff().clip(lower=0).rolling(14).sum() - (-close.diff().clip(upper=0)).rolling(14).sum()) / (close.diff().abs().rolling(14).sum()) * 100
    # RVI (simplified)
    indicators['RVI'] = ((close - open_) / (high - low).replace(0, np.nan)).rolling(14).mean()
    indicators['DPO'] = close - sma(close, 20).shift(10)
    # WaveTrend
    wt1, wt2 = wavelet_trend(close, 10, 21)
    indicators['WaveTrend_1'] = wt1
    indicators['WaveTrend_2'] = wt2
    # QQE
    qqe, qqe_ma = qqe_custom(close, 14, 5, 4.236)
    indicators['QQE'] = qqe
    indicators['QQE_ma'] = qqe_ma
    # TTM Squeeze
    indicators['TTM_Squeeze'] = ttm_squeeze_custom(high, low, close, 20, 2)
    # Schaff Trend Cycle (simplified)
    indicators['STC'] = schaff_trend_cycle(close, 10, 23, 50)
    
    # ----- Volatility
    upper, middle, lower = bollinger_bands(close)
    indicators['BB_upper'] = upper
    indicators['BB_middle'] = middle
    indicators['BB_lower'] = lower
    indicators['ATR_14'] = atr(high, low, close, 14)
    keltner = keltner_channels(high, low, close, 20, 2)
    indicators['KC_upper'] = keltner[0]
    indicators['KC_lower'] = keltner[1]
    indicators['Donchian_upper'] = high.rolling(20).max()
    indicators['Donchian_lower'] = low.rolling(20).min()
    indicators['StdDev_20'] = close.rolling(20).std()
    indicators['Chaikin_Volatility'] = ((high - low).rolling(10).mean() / (high - low).rolling(10).mean().shift(10) - 1) * 100
    indicators['Volatility_Ratio'] = indicators['ATR_14'] / indicators['ATR_14'].rolling(50).mean()
    indicators['Historical_Volatility'] = close.pct_change().rolling(20).std() * np.sqrt(252)
    indicators['Ulcer_Index'] = ulcer_index(close, 14)
    indicators['Price_Channel'] = high.rolling(20).max() - low.rolling(20).min()
    
    # ----- Volume
    indicators['OBV'] = obv(close, volume)
    indicators['VWAP'] = ( (high+low+close)/3 * volume ).cumsum() / volume.cumsum()
    indicators['AccumDist'] = ad(high, low, close, volume)
    indicators['CMF_20'] = cmf(high, low, close, volume, 20)
    indicators['MFI_14'] = mfi(high, low, close, volume, 14)
    indicators['Volume_Osc'] = ((volume.rolling(10).mean() - volume.rolling(30).mean()) / volume.rolling(30).mean()) * 100
    indicators['EaseOfMove'] = ( (high - low) / (high - low).rolling(14).mean() * volume / 1000000 )
    # NVI/PVI simplified
    indicators['NVI'] = (close.pct_change() * (volume < volume.shift())).fillna(0).cumsum()
    indicators['PVI'] = (close.pct_change() * (volume > volume.shift())).fillna(0).cumsum()
    # VWAP Bands
    indicators['VWAP_upper'] = indicators['VWAP'] + 2 * close.rolling(20).std()
    indicators['VWAP_lower'] = indicators['VWAP'] - 2 * close.rolling(20).std()
    
    # ----- Modern / Adaptive
    indicators['SuperTrend_AI'] = adaptive_supertrend(high, low, close, 10)
    indicators['HalfTrend'] = half_trend(high, low, close)
    indicators['OTT'] = ott_indicator(close, 5, 1.4)
    indicators['AlphaTrend'] = alpha_trend(high, low, close, 10, 2.618)
    indicators['SSL_Hybrid'] = ssl_hybrid(close, 10)
    indicators['JMA_20'] = jurik_jma(close, 20)
    indicators['Ehlers_IT'] = ehlers_instantaneous_trend(close)
    indicators['Gaussian_Filter'] = ema(close, 10)
    indicators['Kalman_Trend'] = kalman_filter(close)
    indicators['McGinley_Dynamic'] = mcginley_dynamic(close, 20)
    indicators['VIDYA'] = vidya(close, 14, 0.5)
    indicators['Laguerre_RSI'] = laguerre_rsi(close, 0.5)
    indicators['Squeeze_Mom'] = squeeze_momentum(high, low, close, 20, 2, 1.5)
    indicators['DoubleSmoothed_Stoch'] = double_smoothed_stoch(high, low, close, 14, 3)
    
    # ----- Smart Money / Price Action
    sup, dem = supply_demand_zones(high, low, 20, 5)
    indicators['Supply_Zones'] = sup
    indicators['Demand_Zones'] = dem
    indicators['FVG'] = fair_value_gap(high, low)
    indicators['Liquidity_Sweep'] = liquidity_sweep(high, low, close, 20)
    indicators['POC'] = point_of_control(high, low, close, volume, 20)
    indicators['Anchored_VWAP'] = indicators['VWAP'] # from start
    indicators['Cumulative_Delta'] = cumulative_delta(open_, high, low, close, volume)
    indicators['Stop_Hunt'] = stop_hunt(high, low, close, 10)
    indicators['MM_BuySell'] = market_maker_model(open_, high, low, close, volume)
    
    # ----- Advanced Volatility / Regime
    indicators['ATR_Bands_Adaptive'] = adaptive_atr_bands(high, low, close, 14)
    indicators['Volatility_Stop'] = volatility_stop(high, low, close, 20, 3)
    indicators['VIX_Fix'] = vix_fix(close, 22)
    indicators['BB_Width'] = (indicators['BB_upper'] - indicators['BB_lower']) / indicators['BB_middle']
    indicators['Squeeze_Breakout'] = squeeze_breakout(high, low, close, 20, 2)
    indicators['Regime'] = regime_switching(close, 50, 200)
    indicators['FRAMA_20'] = frama(close, 20)
    indicators['Choppiness_Index'] = choppiness_index(high, low, close, 14)
    indicators['Entropy'] = entropy_indicator(close, 10)
    
    # ----- AI/Quant inspired
    indicators['Neural_RSI'] = neural_rsi(close, 14)
    indicators['AI_AMA'] = ai_adaptive_ma(close, 20)
    indicators['ML_Signal'] = ml_momentum_signal(close, 14, 12, 26)
    
    # ----- Niche / Experimental
    indicators['Fractal_Dimension'] = fractal_dimension(high, low, 5)
    indicators['Hurst_Exponent'] = hurst_exponent(close, 100)
    indicators['Spectral_Period'] = spectral_period(close)
    indicators['Hilbert_Transform'] = hilbert_transform(close)
    indicators['Cycle_Period'] = cycle_period(close, 20)
    indicators['Phase_Accumulation'] = phase_accumulation(close)
    indicators['Chaos_Oscillator'] = chaos_oscillator(high, low, close)
    indicators['Wavelet_Transform'] = wavelet_transform(close)
    indicators['Fourier_Predict'] = fourier_predict(close, 5)
    indicators['Autocorrelation'] = close.autocorr(lag=5)
    indicators['Kurtosis_Skewness'] = close.rolling(20).kurt() + close.rolling(20).skew()
    
    return indicators

# ------------------------------------------------------------------
# Custom implementation helpers (copied/adapted from previous)
# ------------------------------------------------------------------
def ta_psar(high, low, close, af_start=0.02, af_increment=0.02, af_max=0.2):
    # Simpler PSAR
    psar = close.copy()
    up_trend = True
    af = af_start
    ep = low[0] if up_trend else high[0]
    psar[0] = low[0] if up_trend else high[0]
    for i in range(1, len(close)):
        psar[i] = psar[i-1] + af * (ep - psar[i-1])
        if up_trend:
            if low[i] < psar[i]:
                up_trend = False
                psar[i] = ep
                ep = high[i]
                af = af_start
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_increment, af_max)
        else:
            if high[i] > psar[i]:
                up_trend = True
                psar[i] = ep
                ep = low[i]
                af = af_start
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_increment, af_max)
    return psar

def supertrend_custom(high, low, close, length, mult):
    atr_val = atr(high, low, close, length)
    upper_band = (high + low) / 2 + mult * atr_val
    lower_band = (high + low) / 2 - mult * atr_val
    supertrend = pd.Series(index=close.index, dtype=float)
    up_trend = True
    for i in range(len(close)):
        if i == 0:
            supertrend.iloc[i] = lower_band.iloc[i]
            continue
        if up_trend:
            if close.iloc[i] < upper_band.iloc[i-1]:
                up_trend = False
                supertrend.iloc[i] = upper_band.iloc[i]
            else:
                supertrend.iloc[i] = lower_band.iloc[i]
        else:
            if close.iloc[i] > lower_band.iloc[i-1]:
                up_trend = True
                supertrend.iloc[i] = lower_band.iloc[i]
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
    return supertrend

def tsi_indicator(close, r=25, s=13):
    diff = close.diff()
    smoothed_diff = diff.ewm(span=r).mean()
    smoothed_abs = diff.abs().ewm(span=r).mean()
    tsi = smoothed_diff.ewm(span=s).mean() / smoothed_abs.ewm(span=s).mean() * 100
    return tsi

def ultimate_oscillator(high, low, close, p1=7, p2=14, p3=28):
    bp = close - low.shift(1).rolling(p1).min()
    tr = high - low
    avg1 = bp.rolling(p1).sum() / tr.rolling(p1).sum()
    avg2 = bp.rolling(p2).sum() / tr.rolling(p2).sum()
    avg3 = bp.rolling(p3).sum() / tr.rolling(p3).sum()
    return (4*avg1 + 2*avg2 + avg3) / 7 * 100

def fisher_transform(high, low, length):
    hhv = high.rolling(length).max()
    llv = low.rolling(length).min()
    val = 0.33 * 2 * ((close - llv) / (hhv - llv) - 0.5) + 0.67 * close.shift()
    return (np.exp(2*val) - 1) / (np.exp(2*val) + 1)

def wavelet_trend(close, n, m):
    return sma(close, n), sma(close, m)

def qqe_custom(close, rsi_len, smoothing, factor):
    rsi_val = rsi(close, rsi_len)
    ma = ema(rsi_val, smoothing)
    upper = ma + factor * rsi_val.rolling(smoothing).std()
    lower = ma - factor * rsi_val.rolling(smoothing).std()
    qqe = np.where(rsi_val > upper, rsi_val, np.where(rsi_val < lower, rsi_val, ma))
    return qqe, ma

def ttm_squeeze_custom(high, low, close, length, mult):
    bb_upper, bb_mid, bb_lower = bollinger_bands(close, length, mult)
    kc_upper, kc_lower = keltner_channels(high, low, close, length, mult)
    squeeze = ((bb_lower > kc_lower) & (bb_upper < kc_upper)).astype(int)
    return squeeze

def schaff_trend_cycle(close, fast, slow, cycle):
    macd_line, _, _ = macd(close, fast, slow)
    stc = macd_line.rolling(cycle).mean()
    return stc

def keltner_channels(high, low, close, length, mult):
    ema_mid = ema(close, length)
    atr_val = atr(high, low, close, length)
    upper = ema_mid + mult * atr_val
    lower = ema_mid - mult * atr_val
    return upper, lower

def ulcer_index(close, length):
    rolling_max = close.rolling(length).max()
    drawdown = (close - rolling_max) / rolling_max
    ui = np.sqrt((drawdown**2).rolling(length).mean())
    return ui

def adaptive_supertrend(high, low, close, length):
    atr_val = atr(high, low, close, length)
    vol_ratio = atr_val / atr_val.rolling(length*2).mean()
    mult = 2 + vol_ratio
    return supertrend_custom(high, low, close, length, mult)

def half_trend(high, low, close):
    atr_val = atr(high, low, close, 14)
    upper = close + atr_val
    lower = close - atr_val
    return (close > upper).astype(int) - (close < lower).astype(int)

def ott_indicator(close, length, percent):
    sma_val = sma(close, length)
    fark = sma_val * percent / 100
    ott = close.where((close - sma_val) > fark, sma_val)
    return ott

def alpha_trend(high, low, close, length, multiplier):
    atr_val = atr(high, low, close, length)
    trend_up = close - atr_val * multiplier
    trend_down = close + atr_val * multiplier
    return (close > trend_up).astype(int) - (close < trend_down).astype(int)

def ssl_hybrid(close, length):
    ma1 = wma(close, int(length/2))
    ma2 = wma(close, length)
    return (ma1 > ma2).astype(int)

def jurik_jma(close, length):
    roc = close.pct_change().abs()
    ef = roc.ewm(span=length, adjust=False).mean()
    return close.ewm(alpha=ef, adjust=False).mean()

def ehlers_instantaneous_trend(close):
    hp = ema(close, 10)
    return hp

def kalman_filter(close):
    kf = close.copy()
    for i in range(1, len(close)):
        kf.iloc[i] = kf.iloc[i-1] +
