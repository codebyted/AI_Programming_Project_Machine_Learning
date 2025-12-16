# 📈 Stock Price Predictor (ML + Visualization App)

An **interactive stock analysis and forecasting web application** built with **Streamlit**, **yFinance**, **Scikit-learn**, and **Plotly**.
The app downloads historical stock prices, trains a **Random Forest regression model**, evaluates its performance, forecasts future prices, and displays **latest related news**.

> ⚠️ **Educational demo only. Not financial advice.**

---

## 🚀 Key Features

* Download real stock price data from Yahoo Finance
* Interactive ticker selection (AAPL, MSFT, TSLA, etc.)
* Feature engineering with lag, return, and rolling indicators
* Machine Learning model training using Random Forest
* Model evaluation (MAE, RMSE, R²)
* Short-term future price forecasting
* Interactive Plotly charts
* Latest company news via NewsAPI
* CSV download of forecasted prices
* Clean, dark-themed UI

---

## 🧠 Application Workflow

1. User selects a stock ticker and date range
2. Historical OHLCV data is downloaded via `yfinance`
3. Features are engineered from closing prices
4. A Random Forest model is trained
5. Model performance is evaluated on recent data
6. Future prices are predicted iteratively
7. Results are visualized and downloadable
8. Recent news articles are displayed

---

## 🗂️ Project Structure

```
stock-price-predictor/
│
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # Documentation
```

---

## 🧩 Requirements

### 1️⃣ Python Version

* **Python 3.9 – 3.11 (recommended)**

Check version:

```bash
python --version
```

---

### 2️⃣ Required Libraries

| Library      | Purpose                   |
| ------------ | ------------------------- |
| streamlit    | Web application framework |
| pandas       | Data handling             |
| numpy        | Numerical computations    |
| plotly       | Interactive charts        |
| yfinance     | Stock price data          |
| scikit-learn | Machine learning          |
| requests     | API calls                 |
| datetime     | Date handling             |

---

### 3️⃣ Installation

#### Step 1: (Optional) Create Virtual Environment

```bash
python -m venv venv
```

Activate:

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

---

#### Step 2: Install Dependencies

```bash
pip install streamlit pandas numpy plotly yfinance scikit-learn requests
```

Or via `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 📊 Machine Learning Details

### Model Used

* **RandomForestRegressor**
* 200 trees
* Trained on historical price features
* Uses time-aware split (no shuffling)

### Features Engineered

* Lag features (`lag_1` to `lag_5`)
* Daily return
* 5-day rolling mean
* 5-day rolling standard deviation

### Target Variable

* **Close price**

---

## 📈 Model Evaluation Metrics

| Metric | Meaning                        |
| ------ | ------------------------------ |
| MAE    | Average prediction error       |
| RMSE   | Penalizes large errors         |
| R²     | How much variance is explained |

Displayed directly in the UI as metric cards.

---

## 🔮 Forecasting Logic

* Predicts **1–10 future business days**
* Uses **iterative forecasting**
* Each prediction feeds into the next day
* Results shown in:

  * Line chart (history + forecast)
  * Downloadable CSV table

---

## 📰 News Integration

* Fetches latest articles related to the company
* Uses **NewsAPI**
* Displays headline, source, date, and link

### 🔑 News API Key Setup (Optional but Recommended)

Create a free API key at:

```
https://newsapi.org
```

Then set it as an environment variable:

**Windows**

```bash
set NEWS_API_KEY=your_api_key_here
```

**macOS / Linux**

```bash
export NEWS_API_KEY=your_api_key_here
```

---

## 🎨 UI & Styling

* Dark mode theme
* Custom CSS injection
* Plotly interactive charts
* Responsive layout
* Hidden Streamlit branding

---

## ⚠️ Limitations

* Uses only historical prices (no fundamentals)
* Not suitable for real trading
* Forecast horizon is short
* News API may rate-limit
* Random Forest is not time-series specific

---

## 🔮 Possible Enhancements

* Add LSTM / Prophet models
* Add technical indicators (RSI, MACD)
* Add portfolio comparison
* Add user authentication
* Add FastAPI backend
* Deploy to cloud (Railway, Render, AWS)

---

## 👨‍🎓 Intended Audience

* Data science learners
* Machine learning students
* Finance analytics demos
* Portfolio projects
* Hackathons & coursework

---

## 📜 License

Open-source. Free for educational and personal use.

---

## ✨ Author Notes

This project demonstrates:

* Real-world data ingestion
* Feature engineering
* ML model training & evaluation
* Forecasting logic
* Clean visualization practices
* Practical Streamlit UI design

---

