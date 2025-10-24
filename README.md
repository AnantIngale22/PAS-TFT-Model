# PAS-TFT-Model

A production-ready Temporal Fusion Transformer (TFT) implementation for PAS (Purchase Agreement System) time series forecasting.

## Project Structure

```
PAS-TFT-Model/
├── api/                    # FastAPI server and visualization endpoints
│   ├── fastapi_server.py   # Main API server
│   └── visualization.py    # Data visualization utilities
├── data/                   # Data processing and database layer
│   ├── database_connector.py # Database connection and queries
│   └── train_model_with_limited_data.py # Training utilities
├── models/                 # Core ML models and feature engineering
│   ├── tft_model.py        # TFT model implementation
│   └── feature_engineering.py # Feature processing pipeline
├── scripts/                # Utility scripts and data processing
│   ├── add_manufacturer_data.py
│   ├── create_readable_forecasts_fixed.py
│   ├── dummy_pas_data.py
│   ├── fix_forecast_errors.py
│   └── quick_start.py
├── tests/                  # Test suite
│   ├── test_connection.py
│   ├── debug_database.py
│   └── test_features.py
├── outputs/                # Generated forecasts and visualizations
├── logs/                   # Application and training logs
├── main.py                 # Main application entry point
├── config.py               # Configuration management
└── requirements.txt        # Python dependencies
```

## Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/AnantIngale22/PAS-TFT-Model.git
cd PAS-TFT-Model

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env  # Edit with your database credentials
```

### 2. Database Configuration
Update `.env` file with your PostgreSQL credentials:
```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=forecast_model
DB_USER=your_username
DB_PASSWORD=your_password
```

### 3. Run the Application

**Training Mode:**
```bash
python main.py --mode train --company-id 1
```

**API Server:**
```bash
python main.py --mode api
```

**List Companies:**
```bash
python main.py --mode list
```

## Features

### 🤖 Machine Learning
- **Temporal Fusion Transformer (TFT)** - State-of-the-art time series forecasting
- **Adaptive Configuration** - Optimized for small datasets
- **Feature Engineering** - Automated feature processing pipeline
- **Model Evaluation** - Comprehensive performance metrics

### 🌐 API & Visualization
- **FastAPI Server** - RESTful API for model serving
- **Interactive Forecasts** - Real-time prediction endpoints
- **Data Visualization** - Charts and forecast plots
- **PAS Contract Templates** - Predefined contract configurations

### 💾 Data Management
- **PostgreSQL Integration** - Robust database connectivity
- **Multi-company Support** - Handle multiple client datasets
- **Data Validation** - Automated data quality checks
- **Logging System** - Comprehensive application logging

## API Endpoints

- `GET /` - Health check and API information
- `POST /forecast` - Generate forecasts for company
- `GET /pas-contracts/templates` - Available contract templates
- `GET /companies` - List available companies

## Requirements

- Python 3.8+
- PostgreSQL 12+
- See `requirements.txt` for Python dependencies

## License

This project is proprietary software developed for PAS forecasting applications.