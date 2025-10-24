import logging
import argparse
from data.database_connector import DatabaseConnector
from models.tft_model import PASForecaster
from api.fastapi_server import app, PASContractTerms, create_readable_forecasts
import uvicorn


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/forecasting.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def list_companies():
    """List available companies in the database"""
    db = DatabaseConnector()
    companies = db.get_available_companies()
    
    if companies.empty:
        logger.error("❌ No companies found in the database")
        return None
    
    logger.info("🏢 Available Companies:")
    for _, row in companies.iterrows():
        logger.info(f"   ID: {row['company_id']} - {row['company_name']}")
    
    return companies

def train_model(company_id=None):
    """Train forecasting model for a company"""
    logger.info(f"🚀 Starting PAS Forecasting Module")
    logger.info("Using database: anantingale@localhost:5432/forecast_model")
    
    db = DatabaseConnector()
    if not db.test_connection():
        logger.error("❌ Cannot proceed without database connection")
        return
    
    if company_id is None:
        companies = list_companies()
        if companies is None or companies.empty:
            return
        company_id = companies.iloc[0]['company_id']
        logger.info(f"Using company ID: {company_id}")
    
    logger.info("📊 Loading purchase data...")
    data = db.get_purchase_data(company_id)
    
    if data.empty:
        logger.error("❌ No data found for training")
        return
    
    logger.info(f"✅ Loaded {len(data)} records")
    logger.info(f"📈 Time range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    logger.info(f"👥 Unique entities: {data['entity_id'].nunique()}")
    
    logger.info("🤖 Training TFT model...")
    forecaster = PASForecaster(company_id)
    model = forecaster.train(data)
    
    logger.info("✅ Training completed successfully!")
    
    logger.info("🔮 Generating sample predictions...")
    predictions = forecaster.predict(data)
    
    logger.info("🎯 Forecast generation completed!")
    
    evaluation = forecaster.evaluate(data)
    logger.info("📊 Model Evaluation:")
    for key, value in evaluation.items():
        logger.info(f"   {key}: {value}")
    
    return forecaster

def start_api_server():
    """Start the FastAPI server"""
    logger.info("🌐 Starting FastAPI server on http://localhost:8000")
    logger.info("Using database: anantingale@localhost:5432/forecast_model")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

def main():
    parser = argparse.ArgumentParser(description='PAS Forecasting Module')
    parser.add_argument('--mode', choices=['train', 'api', 'test', 'list'], 
                       default='train', help='Operation mode')
    parser.add_argument('--company-id', type=int, 
                       help='Company ID for training (optional)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_model(args.company_id)
    elif args.mode == 'api':
        start_api_server()
    elif args.mode == 'test':
        from tests.test_connection import test_all
        test_all()
    elif args.mode == 'list':
        list_companies()
class PASContractTemplates:
    """Predefined PAS contract templates"""
    
    @staticmethod
    def standard():
        return PASContractTerms(discount_rate=0.15, rebate_rate=0.08, fee_rate=0.03)
    
    @staticmethod
    def premium():
        return PASContractTerms(discount_rate=0.20, rebate_rate=0.12, fee_rate=0.04)
    
    @staticmethod
    def basic():
        return PASContractTerms(discount_rate=0.10, rebate_rate=0.05, fee_rate=0.02)

@app.get("/pas-contracts/templates")
async def get_pas_contract_templates():
    """Get available PAS contract templates"""
    return {
        "standard": PASContractTemplates.standard().dict(),
        "premium": PASContractTemplates.premium().dict(),
        "basic": PASContractTemplates.basic().dict()
    }

if __name__ == "__main__":
    main()