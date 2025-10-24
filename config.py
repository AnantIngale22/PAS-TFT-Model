import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Database Configuration
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_NAME = os.getenv('DB_NAME', 'forecast_model')
    DB_USER = os.getenv('DB_USER', 'anantingale')
    DB_PASSWORD = os.getenv('DB_PASSWORD', 'admin')
    
    # Model Configuration - Adaptive for small datasets
    MAX_ENCODER_LENGTH = 6   # Reduced for small datasets
    MAX_PREDICTION_LENGTH = 2  # Reduced for small datasets
    BATCH_SIZE = 8           # Smaller batches
    LEARNING_RATE = 0.01     # Slower learning
    HIDDEN_SIZE = 12         # Smaller model
    
    # Training Configuration
    MAX_EPOCHS = 20          # More epochs for small data
    EARLY_STOPPING_PATIENCE = 5