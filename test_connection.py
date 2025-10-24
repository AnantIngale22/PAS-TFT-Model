from data.database_connector import DatabaseConnector
from models.tft_model import PASForecaster
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_database_connection():
    """Test database connection and basic queries"""
    logger.info("🧪 Testing database connection...")
    logger.info("Using credentials: anantingale@localhost:5432/forecast_model")
    
    db = DatabaseConnector()
    
    # Test connection
    if not db.test_connection():
        logger.error("❌ Database connection test failed")
        return False
    
    # Test available companies
    companies = db.get_available_companies()
    if not companies.empty:
        logger.info("✅ Available companies:")
        for _, row in companies.iterrows():
            logger.info(f"   Company ID: {row['company_id']}, Name: {row['company_name']}")
    else:
        logger.warning("⚠️  No companies found in database")
    
    # Test data loading for first company
    if not companies.empty:
        company_id = companies.iloc[0]['company_id']
        data = db.get_purchase_data(company_id)
        
        if data.empty:
            logger.warning("⚠️  No purchase data found, but connection successful")
        else:
            logger.info(f"✅ Data test passed - loaded {len(data)} records")
            logger.info(f"   Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    
    return True

def test_model_initialization():
    """Test that models can be initialized"""
    logger.info("🧪 Testing model initialization...")
    
    try:
        forecaster = PASForecaster(company_id=1)
        logger.info("✅ Model initialization test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering"""
    logger.info("🧪 Testing feature engineering...")
    
    try:
        from models.feature_engineering import FeatureEngineer
        
        # Create sample data
        import pandas as pd
        sample_data = pd.DataFrame({
            'entity_id': [1, 1, 2, 2],
            'timestamp': pd.date_range('2023-01-01', periods=4, freq='M'),
            'spend_amount': [1000, 1200, 800, 900],
            'entity_type': ['PHYSICIAN', 'PHYSICIAN', 'PHYSICIAN', 'PHYSICIAN'],
            'entity_name': ['Dr. A', 'Dr. A', 'Dr. B', 'Dr. B']
        })
        
        engineer = FeatureEngineer()
        features = engineer.prepare_features(sample_data)
        
        logger.info(f"✅ Feature engineering test passed - output shape: {features.shape}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Feature engineering test failed: {e}")
        return False

def test_all():
    """Run all tests"""
    logger.info("🏃 Starting all tests with your database credentials...")
    
    tests = [
        test_database_connection,
        test_model_initialization,
        test_feature_engineering
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            logger.error(f"Test {test.__name__} crashed: {e}")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    logger.info(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! You're ready to proceed.")
    else:
        logger.warning("⚠️  Some tests failed. Check the logs above.")

if __name__ == "__main__":
    test_all()