forecasting-system/
│
├── api/
│   └── main.py                 # FastAPI application and endpoints
│
├── data/
│   ├── raw/
│   │   └── sales_data.xlsx     # Original input dataset
│   │
│   └── processed/
│       ├── cleaned_sales.csv   # Aggregated and cleaned data
│       ├── featured_sales.csv  # Data with lag and rolling features
│       ├── train.csv           # Training split
│       └── validation.csv      # Validation split
│
├── evaluation/
│   ├── metrics.py              # Standardized metric calculations
│   ├── compare_models.py       # Logic to select best model per state
│   └── *_results.csv           # Performance metrics for each model type
│
├── feature_engineering/
│   └── build_features.py       # Lag, rolling, and holiday features
│
├── forecasting/
│   └── forecast.py             # Inference logic for the API
│
├── models/
│   ├── arima_model.py          # SARIMA training implementation
│   ├── prophet_model.py        # Facebook Prophet training implementation
│   ├── xgboost_model.py        # XGBoost training implementation
│   └── lstm_model.py           # Deep Learning (LSTM) implementation
│
├── notebooks/
│   └── eda.ipynb               # Exploratory Data Analysis (placeholder)
│
├── preprocessing/
│   ├── load_data.py            # Data ingestion from Excel
│   ├── clean_data.py           # Missing value and duplicate handling
│   └── split_data.py           # Time-series aware train/test splitting
│
├── saved_models/               # Directory for trained .pkl model files
│   ├── arima/
│   ├── prophet/
│   ├── xgboost/
│   └── best_model/
│
├── services/
│   └── pipeline_service.py     # Orchestration of the full end-to-end pipeline
│
├── training/
│   └── train_all_models.py     # Utility to trigger the full training process
│
├── requirements.txt            # Project dependencies
├── README.md                   # System documentation
└── run.py                      # Main entry point for the training pipeline
