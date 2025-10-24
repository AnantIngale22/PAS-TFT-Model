#!/usr/bin/env python3
"""
Train model with whatever data is available
"""

import logging
from data.database_connector import DatabaseConnector
from models.tft_model import PASForecaster

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_with_available_data(company_id, forecast_type="PHYSICIAN"):
    """Train model with whatever data is available"""
    db = DatabaseConnector()
    
    print(f"🚀 Training model for company {company_id} with available data")
    
    # Get available data
    if forecast_type == "PHYSICIAN":
        data = db.get_purchase_data(company_id, include_unlocked=True)  # Use all data
    elif forecast_type == "MANUFACTURER":
        data = db.get_manufacturer_data(company_id, include_unlocked=True)
    else:
        data = db.get_product_data(company_id, include_unlocked=True)
    
    if data.empty:
        print(f"❌ No data found for company {company_id}")
        return None
    
    print(f"📊 Using {len(data)} records for training")
    
    # Train model
    forecaster = PASForecaster(company_id)
    
    try:
        model = forecaster.train(data, use_feature_engineering=False)  # Simple features for small data
        print("✅ Model trained successfully!")
        
        # Evaluate
        evaluation = forecaster.evaluate(data)
        print("📊 Model Evaluation:")
        for key, value in evaluation.items():
            print(f"   {key}: {value}")
        
        return forecaster
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python train_with_limited_data.py <company_id> [forecast_type]")
        print("Example: python train_with_limited_data.py 54 PHYSICIAN")
        return
    
    company_id = int(sys.argv[1])
    forecast_type = sys.argv[2] if len(sys.argv) > 2 else "PHYSICIAN"
    
    forecaster = train_with_available_data(company_id, forecast_type)
    
    if forecaster:
        print(f"\n🎯 Now you can generate forecasts for company {company_id}")
        print("   Use the API or run: python main.py --mode api")

if __name__ == "__main__":
    main()