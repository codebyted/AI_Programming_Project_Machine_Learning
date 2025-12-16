import os
import datetime as dt
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# =========================================================
# Data loading
# =========================================================

@st.cache_data(show_spinner=False)
def load_price_data(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    Download historical OHLCV data for a ticker using yfinance.
    Ensures simple columns: Date, Open, High, Low, Close, Volume.
    """
    df = yf.download(ticker, start=start, end=end)

    if df.empty:
        return df

    df = df.reset_index()

    # If yfinance returns MultiIndex columns, flatten to first level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Make sure we have a Close column
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    return df


@st.cache_data(show_spinner=False)
def get_company_name(ticker: str) -> str:
    """
    Try to get a nice human-readable company name from yfinance.
    """
    try:
        info = yf.Ticker(ticker).info
        return info.get("shortName") or info.get("longName") or ticker
    except Exception:
        return ticker


# =========================================================
# Feature engineering & model
# =========================================================

def create_features(df: pd.DataFrame, n_lags: int = 5) -> pd.DataFrame:
    """
    Create lag and rolling features for the Close price.
    """
    df = df.copy().sort_values("Date")

    for i in range(1, n_lags + 1):
        df[f"lag_{i}"] = df["Close"].shift(i)

    df["return_1d"] = df["Close"].pct_change()
    df["rolling_mean_5"] = df["Close"].rolling(window=5).mean()
    df["rolling_std_5"] = df["Close"].rolling(window=5).std()

    df = df.dropna().reset_index(drop=True)
    return df


def train_model(df: pd.DataFrame):
    """
    Train a RandomForestRegressor on the engineered features.
    """
    feature_cols = [c for c in df.columns if c.startswith("lag_")] + [
        "return_1d",
        "rolling_mean_5",
        "rolling_std_5",
    ]

    X = df[feature_cols]
    y = df["Close"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)

    return model, X_test, y_test, y_pred_test, feature_cols


def forecast_future(
    df: pd.DataFrame,
    model: RandomForestRegressor,
    feature_cols,
    n_days: int = 5,
) -> pd.DataFrame:
    """
    Predict the next n_days business days. Uses iterative forecasting.
    """
    df = df.copy().sort_values("Date")
    closes = list(df["Close"].values)

    future_dates = pd.bdate_range(
        start=df["Date"].iloc[-1] + pd.Timedelta(days=1),
        periods=n_days,
    )

    future_rows = []

    for date in future_dates:
        temp = {}

        # lags from most recent closes
        for i in range(1, 6):
            temp[f"lag_{i}"] = closes[-i]

        temp_series = pd.Series(closes)
        temp["return_1d"] = (temp_series.iloc[-1] / temp_series.iloc[-2]) - 1
        temp["rolling_mean_5"] = temp_series.iloc[-5:].mean()
        temp["rolling_std_5"] = temp_series.iloc[-5:].std()

        X_future = pd.DataFrame([temp])[feature_cols]
        pred_price = model.predict(X_future)[0]

        closes.append(pred_price)
        future_rows.append({"Date": date, "Predicted_Close": pred_price})

    future_df = pd.DataFrame(future_rows)
    return future_df


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)  # for older sklearn versions
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


# =========================================================
# News fetching (easy UI level)
# =========================================================

@st.cache_data(show_spinner=False)
def fetch_news(company_query: str, max_articles: int = 5):
    """
    Fetch latest news about the company using NewsAPI (or similar).
    """
    # 🔴 Put your real key string here:
    api_key = "pub_4e0029b54c44d7aa944b45169c840a2f"
    if not api_key:
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": company_query,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": max_articles,
        "apiKey": api_key,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            # Optional: show status to debug
            st.warning(f"News API error: {resp.status_code}")
            return []
        data = resp.json()
        articles = data.get("articles", [])
        cleaned = []
        for art in articles:
            cleaned.append(
                {
                    "title": art.get("title"),
                    "source": art.get("source", {}).get("name"),
                    "url": art.get("url"),
                    "published_at": art.get("publishedAt"),
                }
            )
        return cleaned
    except Exception as e:
        st.warning(f"News fetch failed: {e}")
        return []



# =========================================================
# UI Styling
# =========================================================

def inject_css():
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        body, .block-container {
            background-color: #0f172a;
        }
        [data-testid="stAppViewContainer"] > .main {
            background: radial-gradient(circle at top left, #0b1020 0%, #020617 55%, #020617 100%);
        }
        .block-container {
            padding-top: 1.5rem;
        }

        .app-title {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            font-weight: 700;
            font-size: 2.1rem;
            color: #f9fafb;
        }
        .app-subtitle {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            font-size: 0.95rem;
            color: #9ca3af;
        }

        section[data-testid="stSidebar"] {
            background-color: #020617;
            border-right: 1px solid #1f2937;
        }
        section[data-testid="stSidebar"] * {
            color: #e5e7eb;
        }

        .metric-card {
            background-color: #020617;
            border-radius: 0.75rem;
            padding: 0.8rem 1rem;
            border: 1px solid #1f2937;
        }

        .download-btn > button {
            width: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# Main app
# =========================================================

def main():
    st.set_page_config(
        page_title="Stock Price Predictor",
        page_icon="📈",
        layout="wide",
    )

    inject_css()

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown("<div class='app-title' style='text-align:center;'>📈 Stock Price Predictor</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='app-subtitle' style='text-align:center;'>"
            "Educational demo: train a simple ML model, forecast a few days ahead, and view recent news. "
            "Not financial advice."
            "</div>",
            unsafe_allow_html=True,
        )

    # ---------------- Sidebar controls ----------------
        # ---------------- Controls (main area instead of sidebar) ----------------
    st.markdown("### ⚙️ Settings")

    c1, c2, c3, c4 = st.columns([2, 2, 2, 1])

    favourite_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NFLX", "NVDA"]

    with c1:
        base_choice = st.selectbox(
            "Quick select ticker",
            options=favourite_tickers,
            index=0,
        )

    with c2:
        custom_ticker = st.text_input(
            "Or type any Yahoo Finance symbol",
            value=base_choice,
        )

    ticker = custom_ticker.upper().strip()

    today = dt.date.today()
    default_start = today - dt.timedelta(days=365 * 2)

    with c3:
        start_date = st.date_input("Start date", value=default_start)

    with c4:
        end_date = st.date_input("End date", value=today)

    n_future_days = st.slider("Days to predict into the future", 1, 10, 5)

    st.caption("Tip: switch tickers to see how the model behaves for different stocks.")

    if start_date >= end_date:
        st.error("Start date must be before end date.")
        return

    # ---------------- Load price data ----------------
    with st.spinner(f"Loading data for {ticker}..."):
        data = load_price_data(ticker, start_date, end_date)

    if data.empty:
        st.error("No data returned. Check the ticker symbol or date range.")
        return

    company_name = get_company_name(ticker)

    st.subheader(f"{company_name} ({ticker}) – Historical Prices")
    price_col1, price_col2 = st.columns([2, 1])

    with price_col1:
        fig = px.line(data, x="Date", y="Close", title="Close Price")
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=360)
        st.plotly_chart(fig, use_container_width=True)

    with price_col2:
        st.write("Recent data:")
        st.dataframe(data.tail(10), use_container_width=True, height=360)

    # ---------------- Features & model ----------------
    with st.spinner("Building features and training model..."):
        df_feat = create_features(data, n_lags=5)

        if len(df_feat) < 30:
            st.error("Not enough data after feature engineering. Try a larger date range.")
            return

        model, X_test, y_test, y_pred_test, feature_cols = train_model(df_feat)

    # ---------------- Evaluation ----------------
    st.subheader("Model evaluation (on most recent data)")

    mae, rmse, r2 = compute_metrics(y_test, y_pred_test)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("MAE", f"{mae:,.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
    with m2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("RMSE", f"{rmse:,.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
    with m3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("R² score", f"{r2:,.3f}")
        st.markdown("</div>", unsafe_allow_html=True)

    eval_df = pd.DataFrame(
        {
            "Date": df_feat["Date"].iloc[-len(y_test):],
            "Actual": y_test.values,
            "Predicted": y_pred_test,
        }
    )
    fig_eval = px.line(
        eval_df,
        x="Date",
        y=["Actual", "Predicted"],
        title="Actual vs Predicted (Test set)",
    )
    fig_eval.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=360)
    st.plotly_chart(fig_eval, use_container_width=True)

    # ---------------- Forecast ----------------
    st.subheader(f"Forecast for next {n_future_days} business days")

    future_df = forecast_future(df_feat[["Date", "Close"]], model, feature_cols, n_days=n_future_days)

    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        history_tail = (
            data[["Date", "Close"]]
            .rename(columns={"Close": "Price"})
            .tail(60)
            .copy()
        )
        history_tail["Type"] = "History"

        forecast_plot = future_df.rename(columns={"Predicted_Close": "Price"}).copy()
        forecast_plot["Type"] = "Forecast"

        combined = pd.concat([history_tail, forecast_plot], ignore_index=True)

        fig_future = px.line(
            combined,
            x="Date",
            y="Price",
            color="Type",
            title="Last 60 days + forecast",
        )
        fig_future.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=360)
        st.plotly_chart(fig_future, use_container_width=True)

    with col_f2:
        st.write("Forecast values:")
        st.dataframe(future_df, use_container_width=True)

        csv_bytes = future_df.to_csv(index=False).encode("utf-8")
        st.markdown("<div class='download-btn'>", unsafe_allow_html=True)
        st.download_button(
            label="⬇️ Download forecast as CSV",
            data=csv_bytes,
            file_name=f"{ticker}_forecast_{n_future_days}d.csv",
            mime="text/csv",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- News panel ----------------
    st.subheader(f"Latest news about {company_name}")

    articles = fetch_news(company_name)

    if not articles:
        st.info(
            "No news could be loaded. "
            "If you want live headlines, create a free NewsAPI key at newsapi.org and set "
            "the environment variable NEWS_API_KEY before running this app."
        )
    else:
        for art in articles:
            title = art.get("title") or "(no title)"
            source = art.get("source") or "Unknown source"
            url = art.get("url") or "#"
            published = art.get("published_at") or ""
            st.markdown(
                f"**[{title}]({url})**  \n"
                f"*{source} – {published[:10]}*",
            )
            st.markdown("---")

    st.info(
        "This is a simplified educational model using only past prices. "
        "Real trading decisions should use deeper analysis and proper risk management."
    )

st.markdown(
    "_MAE ≈ average error in price units; RMSE ≈ like MAE but punishes big errors; "
    "R² ≈ how much of the price movement the model explains (1.0 is perfect)._"
)


if __name__ == "__main__":
    main()
