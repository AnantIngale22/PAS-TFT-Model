import pandas as pd
from sqlalchemy import create_engine, text
import logging
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseConnector:
    def __init__(self):
        # Using your exact connection string
        self.connection_string = f"postgresql+psycopg2://{Config.DB_USER}:{Config.DB_PASSWORD}@{Config.DB_HOST}:{Config.DB_PORT}/{Config.DB_NAME}"
        self.engine = create_engine(self.connection_string)
        logger.info("Database connector initialized with your credentials")
    
    def test_connection(self):
        """Test database connection"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                logger.info("✅ Database connection successful")
                return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    def get_purchase_data(self, company_id, years=2, include_unlocked=True):
        """Get historical purchase data for training"""
        # Build query conditionally
        where_conditions = ["pt.company_id = :company_id", "pt.created_at >= NOW() - INTERVAL ':years years'"]
        
        if not include_unlocked:
            where_conditions.append("pt.is_locked = true")
        
        where_clause = " AND ".join(where_conditions)
        
        query = text(f"""
        SELECT 
            pt.company_id,
            pt.customer_id as entity_id,
            'PHYSICIAN' as entity_type,
            DATE_TRUNC('month', pt.created_at) as timestamp,
            SUM(pt.amount) as spend_amount,
            COUNT(*) as transaction_count,
            AVG(pt.amount) as avg_transaction_value,
            c.customer_name as entity_name,
            EXTRACT(YEAR FROM DATE_TRUNC('month', pt.created_at)) as year,
            EXTRACT(MONTH FROM DATE_TRUNC('month', pt.created_at)) as month
        FROM purchase_transactions pt
        JOIN customers c ON pt.customer_id = c.id
        WHERE {where_clause}
        GROUP BY pt.company_id, pt.customer_id, DATE_TRUNC('month', pt.created_at), c.customer_name
        ORDER BY timestamp, entity_id
        """)
        
        try:
            df = pd.read_sql(query, self.engine, params={"company_id": company_id, "years": years})
            logger.info(f"📊 Loaded {len(df)} records for company_id {company_id} (include_unlocked: {include_unlocked})")
            
            if df.empty:
                logger.warning(f"⚠️  No purchase data found for company_id {company_id}")
                self._debug_company_data(company_id)
                
            return df
        except Exception as e:
            logger.error(f"Error loading purchase data: {e}")
            return pd.DataFrame()
    
    def _debug_company_data(self, company_id):
        """Debug method to check what data is available for a company_id"""
        try:
            # Check if this company_id exists in any table
            logger.info(f"🔍 Debugging company_id: {company_id}")
            
            # Check purchase_transactions
            tx_query = text("""
            SELECT COUNT(*) as count, 
                   MIN(created_at) as earliest, 
                   MAX(created_at) as latest
            FROM purchase_transactions 
            WHERE company_id = :company_id
            """)
            tx_stats = pd.read_sql(tx_query, self.engine, params={"company_id": company_id})
            logger.info(f"   Purchase transactions: {tx_stats.iloc[0]['count']} records")
            logger.info(f"   Date range: {tx_stats.iloc[0]['earliest']} to {tx_stats.iloc[0]['latest']}")
            
            # Check customers
            customer_query = text("SELECT COUNT(*) as count FROM customers WHERE company_id = :company_id")
            customer_count = pd.read_sql(customer_query, self.engine, params={"company_id": company_id})
            logger.info(f"   Customers: {customer_count.iloc[0]['count']}")
            
            # Check manufacturers
            manufacturer_query = text("SELECT COUNT(*) as count FROM manufacturers WHERE company_id = :company_id")
            manufacturer_count = pd.read_sql(manufacturer_query, self.engine, params={"company_id": company_id})
            logger.info(f"   Manufacturers: {manufacturer_count.iloc[0]['count']}")
            
        except Exception as e:
            logger.error(f"Error in debug: {e}")
    
    def get_manufacturer_data(self, company_id, include_unlocked=False):
        """Get manufacturer-level data"""
        where_conditions = ["pt.company_id = :company_id"]
        if not include_unlocked:
            where_conditions.append("pt.is_locked = true")
        
        where_clause = " AND ".join(where_conditions)
        
        query = text(f"""
        SELECT 
            pt.manufacturer_id as entity_id,
            'MANUFACTURER' as entity_type,
            DATE_TRUNC('month', pt.created_at) as timestamp,
            SUM(pt.amount) as spend_amount,
            COUNT(*) as transaction_count,
            m.manufacturer_name as entity_name,
            EXTRACT(YEAR FROM DATE_TRUNC('month', pt.created_at)) as year,
            EXTRACT(MONTH FROM DATE_TRUNC('month', pt.created_at)) as month
        FROM purchase_transactions pt
        JOIN manufacturers m ON pt.manufacturer_id = m.id
        WHERE {where_clause}
        GROUP BY pt.manufacturer_id, DATE_TRUNC('month', pt.created_at), m.manufacturer_name
        ORDER BY timestamp
        """)
        
        try:
            df = pd.read_sql(query, self.engine, params={"company_id": company_id})
            logger.info(f"📊 Loaded {len(df)} manufacturer records for company_id {company_id}")
            return df
        except Exception as e:
            logger.error(f"Error loading manufacturer data: {e}")
            return pd.DataFrame()
    
    def get_product_data(self, company_id, include_unlocked=False):
        """Get product-level data"""
        where_conditions = ["pt.company_id = :company_id"]
        if not include_unlocked:
            where_conditions.append("pt.is_locked = true")
        
        where_clause = " AND ".join(where_conditions)
        
        query = text(f"""
        SELECT 
            pt.product_id as entity_id,
            'PRODUCT' as entity_type,
            DATE_TRUNC('month', pt.created_at) as timestamp,
            SUM(pt.amount) as spend_amount,
            SUM(pt.quantity) as total_quantity,
            COUNT(*) as transaction_count,
            p.product_name as entity_name,
            p.product_type,
            p.market_segment,
            EXTRACT(YEAR FROM DATE_TRUNC('month', pt.created_at)) as year,
            EXTRACT(MONTH FROM DATE_TRUNC('month', pt.created_at)) as month
        FROM purchase_transactions pt
        JOIN products p ON pt.product_id = p.id
        WHERE {where_clause}
        GROUP BY pt.product_id, DATE_TRUNC('month', pt.created_at), 
                 p.product_name, p.product_type, p.market_segment
        ORDER BY timestamp
        """)
        
        try:
            df = pd.read_sql(query, self.engine, params={"company_id": company_id})
            logger.info(f"📊 Loaded {len(df)} product records for company_id {company_id}")
            return df
        except Exception as e:
            logger.error(f"Error loading product data: {e}")
            return pd.DataFrame()
    
    def get_available_companies(self):
        """Get list of available company_ids that actually have data"""
        query = text("""
        SELECT DISTINCT pt.company_id, c.company_name
        FROM purchase_transactions pt
        LEFT JOIN companies c ON pt.company_id = c.id
        WHERE pt.is_locked = true
        ORDER BY pt.company_id
        """)
        
        try:
            companies = pd.read_sql(query, self.engine)
            logger.info(f"Found {len(companies)} company_ids with purchase data")
            return companies
        except Exception as e:
            logger.error(f"Error fetching companies with data: {e}")
            
            # Fallback: just get company_ids from purchase_transactions
            fallback_query = text("""
            SELECT DISTINCT company_id 
            FROM purchase_transactions 
            WHERE is_locked = true
            ORDER BY company_id
            """)
            companies = pd.read_sql(fallback_query, self.engine)
            companies['company_name'] = f"Company {companies['company_id']}"
            return companies
    
    def get_company_name(self, company_id):
        """Get company name by company_id"""
        query = text("""
        SELECT company_name 
        FROM companies 
        WHERE id = :company_id AND is_deleted = false
        """)
        
        try:
            result = pd.read_sql(query, self.engine, params={"company_id": company_id})
            if not result.empty:
                return result['company_name'].iloc[0]
            else:
                # If company not found in companies table, return generic name
                return f"Company {company_id}"
        except Exception as e:
            logger.error(f"Error fetching company name: {e}")
            return f"Company {company_id}"
    
    def get_all_company_ids(self):
        """Get all company_ids that exist in the database (from any table)"""
        try:
            # Get company_ids from purchase_transactions
            pt_companies = pd.read_sql("SELECT DISTINCT company_id FROM purchase_transactions ORDER BY company_id", self.engine)
            
            # Get company ids from companies table
            companies_table = pd.read_sql("SELECT id as company_id FROM companies WHERE is_deleted = false ORDER BY id", self.engine)
            
            # Combine and deduplicate
            all_companies = pd.concat([pt_companies, companies_table]).drop_duplicates().sort_values('company_id')
            
            logger.info(f"All company_ids in database: {list(all_companies['company_id'])}")
            return all_companies
        except Exception as e:
            logger.error(f"Error getting all company_ids: {e}")
            return pd.DataFrame()

    def get_sample_data(self, limit=5):
        """Get sample data from all tables for debugging"""
        try:
            logger.info("🔍 SAMPLE DATA FROM ALL TABLES:")
            
            # Sample from companies table (using id)
            companies_sample = pd.read_sql("SELECT id, company_name FROM companies WHERE is_deleted = false LIMIT 3", self.engine)
            logger.info("🏢 Companies table (id column):")
            for _, row in companies_sample.iterrows():
                logger.info(f"   ID: {row['id']}, Name: {row['company_name']}")
            
            # Sample from purchase_transactions (using company_id)
            tx_sample = pd.read_sql(f"SELECT DISTINCT company_id FROM purchase_transactions LIMIT {limit}", self.engine)
            logger.info("💰 Purchase transactions (company_id column):")
            logger.info(f"   Unique company_ids: {list(tx_sample['company_id'])}")
            
            # Show sample transactions
            sample_tx = pd.read_sql(f"""
            SELECT company_id, customer_id, amount, created_at, is_locked 
            FROM purchase_transactions 
            ORDER BY created_at DESC 
            LIMIT {limit}
            """, self.engine)
            logger.info("   Recent transactions:")
            for _, tx in sample_tx.iterrows():
                logger.info(f"      Company: {tx['company_id']}, Customer: {tx['customer_id']}, Amount: ${tx['amount']}")
            
            # Count records by company_id
            counts = pd.read_sql("""
            SELECT 
                company_id,
                COUNT(*) as transaction_count,
                COUNT(CASE WHEN is_locked = true THEN 1 END) as locked_count
            FROM purchase_transactions 
            GROUP BY company_id
            ORDER BY company_id
            """, self.engine)
            
            logger.info("📊 Transactions by company_id:")
            for _, row in counts.iterrows():
                logger.info(f"   Company {row['company_id']}: {row['transaction_count']} total, {row['locked_count']} locked")
            
        except Exception as e:
            logger.error(f"Error getting sample data: {e}")