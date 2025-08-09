# app.py
import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
import plotly.graph_objects as go

from src.data_loader import fetch_ohlcv
from src.indicators import add_technical_indicators
from src.sentiment import fetch_news_sentiment
from src.model import train_and_save, load_model, predict_next, FEATURES

# Load .env for local dev
load_dotenv()

def get_newsapi_key():
    # Priority: Streamlit secrets -> environment variable
    try:
        key = st.secrets["NEWSAPI_KEY"]
        if key:
            return key
    except Exception:
        pass
    return os.getenv("NEWSAPI_KEY")

st.set_page_config(page_title="Crypto Price Predictor", layout="wide")
st.title("📈 Crypto Price Predictor (Linear Regression + Indicators + Sentiment)")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    symbol_display = st.selectbox("Cryptocurrency", ["Bitcoin (BTC-USD)", "Ethereum (ETH-USD)", "Solana (SOL-USD)"])
    symbol = symbol_display.split("(")[1].strip(")")
    days = st.slider("History window (days)", min_value=90, max_value=1500, value=365, step=30)
    horizon = st.selectbox("Prediction horizon (days ahead)", [1])  # beginner version supports 1 day
    retrain = st.button("Retrain model now")
    st.write("---")
    st.markdown("**API keys**: set `NEWSAPI_KEY` in `.env` (local) or Streamlit secrets.")

# Caching data fetch
@st.cache_data(ttl=600)
def load_data(symbol, days):
    return fetch_ohlcv(symbol, period_days=days)

df = load_data(symbol, days)
if df is None or df.empty:
    st.error("No data fetched. Try increasing the history window or check your internet.")
    st.stop()

st.subheader(f"Historical prices for {symbol} (last {len(df)} rows)")
st.dataframe(df.tail(5), use_container_width=True)

# Indicators
df_ind = add_technical_indicators(df)

# Sentiment (cached)
@st.cache_data(ttl=600)
def get_sentiment(sym):
    key = get_newsapi_key()
    map_q = {"BTC-USD":"bitcoin", "ETH-USD":"ethereum", "SOL-USD":"solana"}
    q = map_q.get(sym, sym.split("-")[0])
    return fetch_news_sentiment(q, api_key=key, n_articles=25)

sentiment_score = get_sentiment(symbol)
st.metric("News Sentiment (avg VADER compound)", f"{sentiment_score:.3f}")

# Broadcast sentiment into df
df_ind['Sentiment'] = sentiment_score

# Plot price + indicators
st.subheader("Price & indicators")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_ind['Date'], y=df_ind['Close'], name='Close', line=dict(color='royalblue')))
fig.add_trace(go.Scatter(x=df_ind['Date'], y=df_ind['MA7'], name='MA7', line=dict(color='orange')))
fig.add_trace(go.Scatter(x=df_ind['Date'], y=df_ind['EMA14'], name='EMA14', line=dict(color='green')))
fig.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20))
st.plotly_chart(fig, use_container_width=True)

st.write("Latest indicators")
st.table(df_ind[['Date','Close','MA7','EMA14','RSI','MACD_diff','BB_width','Sentiment']].tail(3).set_index('Date'))

# Load or train model
model = load_model()
metric = None
if model is None or retrain:
    with st.spinner("Training Linear Regression model (this may take ~30-60s)..."):
        try:
            model, metric = train_and_save(df_ind, model_path="models/crypto_lr.pkl", horizon=horizon)
            st.success("Model trained and saved.")
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()
else:
    st.info("Loaded existing model from models/crypto_lr.pkl")

if metric:
    st.write(f"Cross-validated MAE (avg): {metric:.2f}")

# Predict next day
latest = df_ind.iloc[-1]
latest_feat = latest[FEATURES].fillna(0)
pred_price = predict_next(model, latest_feat)

# Next-day date
last_date = pd.to_datetime(df_ind['Date'].iloc[-1])
next_date = last_date + pd.Timedelta(days=1)

st.metric(f"Predicted Close price for {symbol} on {next_date.date()}", f"${pred_price:,.2f}")

# Show predicted point
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df_ind['Date'], y=df_ind['Close'], name='Close'))
fig2.add_trace(go.Scatter(x=[next_date], y=[pred_price], mode='markers+text', name='Predicted', text=[f"${pred_price:,.0f}"], textposition="top center"))
fig2.update_layout(height=450, margin=dict(l=20,r=20,t=30,b=20))
st.plotly_chart(fig2, use_container_width=True)

st.write("Feature values used for prediction")
st.table(latest_feat[FEATURES].to_frame(name='value').T)

st.caption("Linear Regression baseline — consider RandomForest/XGBoost for better performance.")
