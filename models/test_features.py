from data.database_connector import DatabaseConnector
from models.feature_engineering import PASFeatureEngineer
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_feature_engineering():
    """Test the complete feature engineering pipeline"""
    logger.info("🧪 Testing Feature Engineering Pipeline...")
    
    # 1. Load sample data
    db = DatabaseConnector()
    data = db.get_purchase_data(company_id=1, years=1)
    
    if data.empty:
        logger.error("❌ No data to test feature engineering")
        return
    
    logger.info(f"📊 Input data: {data.shape}")
    print("Sample of original data:")
    print(data[['entity_id', 'timestamp', 'spend_amount', 'entity_name']].head())
    
    # 2. Initialize feature engineer
    engineer = PASFeatureEngineer()
    
    # 3. Run complete pipeline
    logger.info("🏗️ Running feature engineering pipeline...")
    features_df = engineer.prepare_features(data)
    
    # 4. Display results
    logger.info(f"✅ Output data: {features_df.shape}")
    print("\nFeature Summary:")
    feature_summary = engineer.get_feature_summary(features_df)
    print(feature_summary.head(10))
    
    # 5. Show some engineered features
    print("\nSample of Engineered Features:")
    engineered_cols = [col for col in features_df.columns if any(x in col for x in ['lag', 'rolling', 'trend', 'growth', 'interaction'])]
    sample_features = features_df[['entity_id', 'timestamp', 'spend_amount'] + engineered_cols[:5]].head()
    print(sample_features)
    
    # 6. Show feature types breakdown
    time_features = [col for col in features_df.columns if 'month' in col or 'quarter' in col or 'year' in col]
    lag_features = [col for col in features_df.columns if 'lag' in col or 'rolling' in col]
    trend_features = [col for col in features_df.columns if 'trend' in col or 'growth' in col]
    entity_features = [col for col in features_df.columns if 'entity_' in col and 'entity_id' not in col]
    interaction_features = [col for col in features_df.columns if 'interaction' in col]
    
    print(f"\n📈 Feature Type Breakdown:")
    print(f"   Time Features: {len(time_features)}")
    print(f"   Lag Features: {len(lag_features)}")
    print(f"   Trend Features: {len(trend_features)}")
    print(f"   Entity Features: {len(entity_features)}")
    print(f"   Interaction Features: {len(interaction_features)}")
    print(f"   Total Features: {len(features_df.columns)}")
    
    return features_df, engineer

if __name__ == "__main__":
    features_df, engineer = test_feature_engineering()