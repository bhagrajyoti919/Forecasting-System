# Video Presentation Script (3.5 Minutes Max)

Use this condensed guide to stay within the 3.5-minute time limit for your submission.

---

## 🎙️ 1. Intro & Goal (30s)
*Visual: GitHub Repository or Project root.*

"Hi, I’m [Your Name]. I've built an automated, production-ready Sales Forecasting System. The goal was to take weekly sales data, automatically find the best model for each state, and serve 8-week forecasts through a REST API. It’s designed as a modular backend service, not just a notebook."

---

## 🎙️ 2. Architecture & Data Prep (45s)
*Visual: Open `project structure.md` and quickly show `clean_data.py`.*

"The system is modular. I handle raw Excel data by cleaning missing values and ensuring a continuous weekly frequency for every state. 

For feature engineering, I created:
- **Lag Features** (t-1, t-7, t-30) to capture history.
- **Rolling Mean & Std** to track local trends.
- **Holiday Flags** to handle seasonal spikes.
I used a chronological split—the last 8 weeks for validation—to ensure zero data leakage."

---

## 🎙️ 3. The Models & Selection (60s)
*Visual: Scroll through `models/` folder and open `compare_models.py`.*

"I implemented and compared four mandatory models:
1. **SARIMA**: Optimized for weekly seasonality (s=52).
2. **Prophet**: For robust trend and holiday handling.
3. **XGBoost**: A regression approach using our engineered features.
4. **LSTM**: For deep learning non-linear patterns.

The `compare_models.py` script evaluates these using RMSE and MAE, then automatically selects the top-performing model for **each individual state**. This ensures maximum accuracy across different regions."

---

## 🎙️ 4. Execution & API Demo (60s)
*Visual: Terminal showing `python run.py` output, then browser at `/docs`.*

"The entire pipeline is orchestrated by `run.py`. Running this single script handles the full flow from ingestion to model saving. 

On the serving side, I used **FastAPI**. If we call the `/forecast/Texas` endpoint, the API:
- Identifies the best model for Texas.
- Loads the latest features.
- Returns a precise 8-week forecast in JSON. 
We also have a `/train` endpoint to refresh the system with new data."

---

## 🎙️ 5. Conclusion (15s)
*Visual: Show `README.md` checklist.*

"The system is fully automated, documented, and meets all production requirements. It handles seasonality, missing data, and model selection seamlessly. Thanks for watching!"

---

### ⏱️ Timing Breakdown:
- **0:00 - 0:30**: Intro
- **0:30 - 1:15**: Data & Features
- **1:15 - 2:15**: Models & Comparison
- **2:15 - 3:15**: Running the System & API
- **3:15 - 3:30**: Wrap up
