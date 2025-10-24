#!/usr/bin/env python3
"""
Debug script to understand the relationship between companies.id and company_id in other tables
"""

import logging
import pandas as pd
from sqlalchemy import create_engine, text
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("🔍 UNDERSTANDING COMPANY ID RELATIONSHIPS")
    
    connection_string = f"postgresql+psycopg2://{Config.DB_USER}:{Config.DB_PASSWORD}@{Config.DB_HOST}:{Config.DB_PORT}/{Config.DB_NAME}"
    engine = create_engine(connection_string)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            logger.info("✅ Database connection successful")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return
    
    # 1. Understand the ID relationships
    logger.info("\n🆔 ID RELATIONSHIP ANALYSIS:")
    
    # Get all companies from companies table
    companies = pd.read_sql("SELECT id, company_name FROM companies WHERE is_deleted = false ORDER BY id", engine)
    logger.info("🏢 Companies table (id column):")
    for _, row in companies.iterrows():
        logger.info(f"   ID: {row['id']} - '{row['company_name']}'")
    
    # Get all company_ids from purchase_transactions
    pt_companies = pd.read_sql("SELECT DISTINCT company_id FROM purchase_transactions ORDER BY company_id", engine)
    logger.info("💰 Purchase transactions table (company_id column):")
    logger.info(f"   Company IDs: {list(pt_companies['company_id'])}")
    
    # Find matches between companies.id and purchase_transactions.company_id
    logger.info("\n🔗 MATCHING ANALYSIS:")
    
    matching_companies = []
    non_matching_pt = []
    
    for pt_company_id in pt_companies['company_id']:
        # Check if this company_id exists in companies table
        check_query = text("SELECT id, company_name FROM companies WHERE id = :company_id AND is_deleted = false")
        result = pd.read_sql(check_query, engine, params={"company_id": pt_company_id})
        
        if not result.empty:
            matching_companies.append({
                'company_id': pt_company_id,
                'company_name': result.iloc[0]['company_name'],
                'id_in_companies_table': result.iloc[0]['id']
            })
        else:
            non_matching_pt.append(pt_company_id)
    
    logger.info("✅ Company IDs that match between tables:")
    for match in matching_companies:
        logger.info(f"   Company ID: {match['company_id']} - '{match['company_name']}'")
    
    if non_matching_pt:
        logger.info("❌ Company IDs in purchase_transactions but NOT in companies table:")
        for company_id in non_matching_pt:
            logger.info(f"   Company ID: {company_id}")
    
    # 2. Check data availability for each company_id
    logger.info("\n📊 DATA AVAILABILITY BY COMPANY_ID:")
    
    all_company_ids = list(pt_companies['company_id'])
    
    for company_id in all_company_ids:
        # Count transactions
        count_query = text("""
        SELECT 
            COUNT(*) as total_count,
            COUNT(CASE WHEN is_locked = true THEN 1 END) as locked_count,
            MIN(created_at) as earliest_date,
            MAX(created_at) as latest_date
        FROM purchase_transactions 
        WHERE company_id = :company_id
        """)
        
        stats = pd.read_sql(count_query, engine, params={"company_id": company_id})
        total = stats.iloc[0]['total_count']
        locked = stats.iloc[0]['locked_count']
        earliest = stats.iloc[0]['earliest_date']
        latest = stats.iloc[0]['latest_date']
        
        # Get company name if available
        name_query = text("SELECT company_name FROM companies WHERE id = :company_id")
        name_result = pd.read_sql(name_query, engine, params={"company_id": company_id})
        company_name = name_result.iloc[0]['company_name'] if not name_result.empty else "Unknown Company"
        
        logger.info(f"Company ID {company_id} ('{company_name}'):")
        logger.info(f"   Total transactions: {total}")
        logger.info(f"   Locked transactions: {locked}")
        logger.info(f"   Date range: {earliest} to {latest}")
        
        if locked >= 10:
            logger.info(f"   ✅ SUITABLE for training (has {locked} locked transactions)")
        else:
            logger.info(f"   ⚠️  May need more data (only {locked} locked transactions)")
    
    # 3. Recommendations
    logger.info("\n🎯 RECOMMENDATIONS:")
    
    suitable_companies = []
    for company_id in all_company_ids:
        # Check if suitable for training
        count_query = text("SELECT COUNT(*) as locked_count FROM purchase_transactions WHERE company_id = :company_id AND is_locked = true")
        locked_count = pd.read_sql(count_query, engine, params={"company_id": company_id}).iloc[0]['locked_count']
        
        if locked_count >= 10:
            suitable_companies.append(company_id)
    
    if suitable_companies:
        logger.info("✅ Use these company_ids for training:")
        for company_id in suitable_companies:
            name_query = text("SELECT company_name FROM companies WHERE id = :company_id")
            name_result = pd.read_sql(name_query, engine, params={"company_id": company_id})
            company_name = name_result.iloc[0]['company_name'] if not name_result.empty else "Unknown"
            logger.info(f"   Company ID: {company_id} - '{company_name}'")
    else:
        logger.info("❌ No company_ids have sufficient locked data (need at least 10 locked transactions)")
        logger.info("   Try using include_unlocked=True or create more sample data")

if __name__ == "__main__":
    main()