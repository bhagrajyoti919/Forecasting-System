# Video Presentation Script: End-to-End Sales Forecasting System

Use this script as a guide for your presentation video. It follows the logical flow of the project from data ingestion to API serving.

---

## 🎙️ Section 1: Introduction (30-45 seconds)
*Visual: Show the GitHub repository or the project structure in the IDE.*

"Hello, my name is [Your Name], and today I’m presenting my End-to-End Sales Forecasting System. The objective of this project was to build a production-ready backend service that can ingest weekly sales data, automatically train and compare multiple forecasting models, and serve the best predictions through a REST API."

---

## 🎙️ Section 2: Project Structure & Modular Design (1 minute)
*Visual: Open `project structure.md` and expand the folders in the sidebar.*

"I’ve designed the system with a clean, modular architecture:
- The **Data Layer** handles raw and processed states.
- The **Preprocessing and Feature Engineering** modules ensure our data is clean and rich with temporal features.
- The **Models** folder contains implementations of four distinct algorithms: SARIMA, Prophet, XGBoost, and LSTM.
- And finally, the **API Layer** built with FastAPI allows for real-time inference."

---

## 🎙️ Section 3: Feature Engineering & Data Prep (1-2 minutes)
*Visual: Open `feature_engineering/build_features.py` and `preprocessing/clean_data.py`.*

"To handle the complexities of time-series data, I implemented:
- **Missing Value Handling**: We use aggregate grouping and forward-filling to ensure a continuous weekly frequency for every state.
- **Advanced Features**: In the `build_features.py` script, I created lag features for historical context, rolling statistics to capture local trends, and integrated a US holiday calendar to account for seasonal sales spikes.
- **Time-Series Split**: We use a chronological split (the last 8 weeks for validation) to prevent data leakage."

---

## 🎙️ Section 4: Model Training & Comparison (2 minutes)
*Visual: Open `models/arima_model.py` and `evaluation/compare_models.py`.*

"One of the core requirements was to compare multiple models:
1. **SARIMA**: Optimized for weekly data with a seasonal order of 52.
2. **Prophet**: For robust trend handling.
3. **XGBoost**: Using our engineered features for regression-based forecasting.
4. **LSTM**: For capturing deep non-linear dependencies.

Once trained, the `compare_models.py` script automatically evaluates every model using RMSE and MAE. It then selects the top-performing model for **each individual state**, ensuring we always provide the most accurate forecast possible."

---

## 🎙️ Section 5: Running the System (1 minute)
*Visual: Open a terminal and run `python run.py` (or show the logs of a previous run).*

"Running the entire system is simple. By executing `run.py`, the orchestration service triggers the full pipeline—cleaning, engineering, training, and selection—in one go. All results are saved in the `evaluation/` folder and trained models are stored as `.pkl` files."

---

## 🎙️ Section 6: API Demonstration (1.5 minutes)
*Visual: Open a browser at `http://127.0.0.1:8000/docs` or use a tool like Postman.*

"Finally, let's look at the API. I used FastAPI for its speed and automatic documentation.
- We have a `/health` check.
- A `/train` endpoint to upload new data and retrain the whole system.
- And the main `/forecast/{state_name}` endpoint.

When we request a forecast for a state like 'Texas', the API identifies the best model (in this case, XGBoost), loads the latest features, and returns an 8-week prediction in JSON format."

---

## 🎙️ Section 7: Conclusion (30 seconds)
*Visual: Show the `README.md` and the final checklist.*

"In summary, this system is fully automated, handles seasonality and missing data, and provides a scalable way to serve predictions. All code is documented and refactored for professional standards. Thank you for your time!"

---

### 💡 Presentation Tips:
- **Be clear and steady**: Don't rush through the code.
- **Focus on the 'Why'**: Explain *why* you used a specific feature or model (e.g., 's=52 for weekly data').
- **Highlight the automation**: The company wants to see that you've built a *system*, not just a one-off notebook.
