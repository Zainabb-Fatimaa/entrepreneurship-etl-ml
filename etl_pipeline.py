import pandas as pd
import numpy as np
import sqlite3
import json
import os
from datetime import datetime, timedelta
import random
import psycopg2
from sqlalchemy import create_engine, text
import logging 
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DB_HOST = os.getenv('DB_HOST')  
DB_NAME = os.getenv('DB_NAME')  
DB_USER = os.getenv('DB_USER')  
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_SSLMODE = os.getenv('DB_SSLMODE', 'require')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OLTP1_DIR = os.path.join(DATA_DIR, 'oltp1_startups')
OLTP2_DIR = os.path.join(DATA_DIR, 'oltp2_investors')
CSV_DIR = os.path.join(DATA_DIR, 'csv_source')
API_CACHE_DIR = os.path.join(DATA_DIR, 'api_cache')


def get_sqlalchemy_engine():
    """Creates SQLAlchemy engine for database operations."""
    try:
        from urllib.parse import quote_plus
        
        if not DB_PASSWORD:
            raise ValueError("DB_PASSWORD is not set in environment variables")
        
        encoded_password = quote_plus(str(DB_PASSWORD))
        
        connection_string = f"postgresql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode={DB_SSLMODE}"
        engine = create_engine(connection_string, pool_pre_ping=True)
        
        # Test the connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        logging.info(f" Successfully connected to PostgreSQL at {DB_HOST}:{DB_PORT}")
        return engine
        
    except Exception as e:
        logging.error("="*70)
        logging.error("DATABASE CONNECTION FAILED")
        logging.error("="*70)
        logging.error(f"Error: {e}")
        logging.error("\nTroubleshooting steps:")
        logging.error("1. Verify your .env file has the correct Session Pooler settings:")
        logging.error("   DB_HOST=aws-1-ap-southeast-1.pooler.supabase.com")
        logging.error("   DB_PORT=5432")
        logging.error("   DB_USER=postgres.bfbsqzaygxvdyfvzfmgd")
        logging.error("   DB_PASSWORD=[your-password]")
        logging.error("   DB_SSLMODE=require")
        logging.error("2. Ensure your Supabase project is not paused (free tier)")
        logging.error("3. Check if your password contains special characters that need escaping")
        logging.error("="*70)
        return None


def run_etl():
    for directory in [OLTP1_DIR, OLTP2_DIR, CSV_DIR, API_CACHE_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    logging.info("="*70)
    logging.info("  ENTREPRENEURSHIP DATA WAREHOUSE - ETL PIPELINE")
    logging.info("="*70)

    logging.info("\n" + "="*70)
    logging.info("  EXTRACTION PHASE")
    logging.info("="*70)

    try:
        conn_oltp1 = sqlite3.connect(os.path.join(OLTP1_DIR, 'startup_oltp.db'))
        query_startups = "SELECT * FROM startups"
        startups_df = pd.read_sql(query_startups, conn_oltp1)
        startups_df['source_system'] = 'OLTP1_Startups'
        logging.info(f" Extracted {len(startups_df)} startups from OLTP1")

        query_funding = "SELECT * FROM funding_rounds"
        funding_df = pd.read_sql(query_funding, conn_oltp1)
        logging.info(f" Extracted {len(funding_df)} funding rounds from OLTP1")

        query_revenue = "SELECT * FROM revenue_metrics"
        revenue_df = pd.read_sql(query_revenue, conn_oltp1)
        conn_oltp1.close()
        logging.info(f" Extracted {len(revenue_df)} revenue records from OLTP1")
    except Exception as e:
        logging.error(f"Error during OLTP1 extraction: {e}")
        return 


    try:
        conn_oltp2 = sqlite3.connect(os.path.join(OLTP2_DIR, 'investor_oltp.db'))
        query_investors = "SELECT * FROM investors"
        investors_df = pd.read_sql(query_investors, conn_oltp2)
        investors_df['source_system'] = 'OLTP2_Investors'
        logging.info(f" Extracted {len(investors_df)} investors from OLTP2")

        query_portfolio = "SELECT * FROM portfolio_companies"
        portfolio_df = pd.read_sql(query_portfolio, conn_oltp2)
        logging.info(f" Extracted {len(portfolio_df)} portfolio investments from OLTP2")

        query_trends = "SELECT * FROM market_trends"
        trends_df = pd.read_sql(query_trends, conn_oltp2)
        conn_oltp2.close()
        logging.info(f" Extracted {len(trends_df)} market trend records from OLTP2")
    except Exception as e:
        logging.error(f"Error during OLTP2 extraction: {e}")
        return 

    try:
        accelerators_df = pd.read_csv(os.path.join(CSV_DIR, 'accelerators.csv'))
        accelerators_df['source_system'] = 'CSV_Accelerators'
        logging.info(f" Extracted {len(accelerators_df)} accelerator programs from CSV")

        economic_df = pd.read_csv(os.path.join(CSV_DIR, 'economic_indicators.csv'))
        economic_df['source_system'] = 'CSV_EconomicData'
        logging.info(f" Extracted {len(economic_df)} economic indicators from CSV (API-Enhanced)")

        competitors_df = pd.read_csv(os.path.join(CSV_DIR, 'competitors.csv'))
        competitors_df['source_system'] = 'CSV_Competitors'
        logging.info(f" Extracted {len(competitors_df)} competitors from CSV")
    except Exception as e:
        logging.error(f"Error during CSV extraction: {e}")
        return 

    try:
        market_api_df = pd.read_csv(os.path.join(API_CACHE_DIR, 'live_market_data.csv'))
        market_api_df['source_system'] = 'API_YahooFinance'
        api_metadata = {}
        try:
            with open(os.path.join(API_CACHE_DIR, 'market_metadata.json'), 'r') as f:
                api_metadata = json.load(f)
        except FileNotFoundError:
            logging.warning("Warning: market_metadata.json not found in API cache.")

        logging.info(f" Extracted {len(market_api_df)} live securities from Yahoo Finance API")
        if api_metadata:
            logging.info(f"  API Timestamp: {api_metadata.get('fetch_timestamp', 'N/A')}")
            logging.info(f"  API Status: {api_metadata.get('api_status', 'N/A')}")
        else:
            logging.info("  API metadata not available.")

    except Exception as e:
        logging.error(f"Error during API data extraction: {e}")
        return 

    logging.info("\n EXTRACTION COMPLETED")

    logging.info("\n" + "="*70)
    logging.info("  TRANSFORMATION PHASE")
    logging.info("="*70)

    logging.info("\n[3/5] Transforming data...")

    try:
        logging.info("\nCleaning startup data...")
        startups_df = startups_df.drop_duplicates(subset=['startup_id'])
        startups_df['company_name'] = startups_df['company_name'].str.strip()
        startups_df['founded_date'] = pd.to_datetime(startups_df['founded_date'], errors='coerce')
        startups_df['headquarters_city'] = startups_df['headquarters_city'].fillna('Unknown')
        logging.info(f" Cleaned {len(startups_df)} startup records")

        logging.info("Cleaning investor data...")
        investors_df = investors_df.drop_duplicates(subset=['investor_id'])
        investors_df['investor_name'] = investors_df['investor_name'].str.strip()
        investors_df['contact_email'] = investors_df['contact_email'].fillna('unknown@email.com')
        logging.info(f"Cleaned {len(investors_df)} investor records")

        logging.info("Cleaning funding data...")
        funding_df = funding_df.drop_duplicates(subset=['funding_id'])
        funding_df['funding_date'] = pd.to_datetime(funding_df['funding_date'], errors='coerce')
        funding_df = funding_df[funding_df['amount_raised'] > 0]
        funding_df['valuation_multiple'] = (funding_df['valuation'] / funding_df['amount_raised']).replace([np.inf, -np.inf], np.nan).round(2)
        logging.info(f" Cleaned {len(funding_df)} funding records")

        logging.info("Cleaning economic indicators (API-enhanced)...")
        economic_df['date'] = pd.to_datetime(economic_df['date'], errors='coerce')
        economic_df = economic_df.sort_values('date')
        logging.info(f" Cleaned {len(economic_df)} economic indicator records")
        logging.info("  Includes real market volatility from Yahoo Finance")

        logging.info("Transforming live market data...")
        market_api_df['fetch_date'] = pd.to_datetime(market_api_df['fetch_date'], errors='coerce')
        market_api_df['market_cap_billions'] = (market_api_df['market_cap'] / 1e9).round(2)
        logging.info(f" Transformed {len(market_api_df)} live market records")

        logging.info("Generating date dimension...")
        all_dates = []
        if not funding_df.empty and 'funding_date' in funding_df.columns:
            all_dates.extend(funding_df['funding_date'].dropna())
        if not revenue_df.empty and 'reporting_date' in revenue_df.columns:
             all_dates.extend(revenue_df['reporting_date'].apply(pd.to_datetime, errors='coerce').dropna())
        if not economic_df.empty and 'date' in economic_df.columns:
            all_dates.extend(economic_df['date'].dropna())

        if all_dates:
            dates = pd.date_range(
                start=min(all_dates) - pd.Timedelta(days=365),
                end=max(all_dates) + pd.Timedelta(days=365),
                freq='D'
            )
        else:
            dates = pd.date_range(
                start=datetime.now() - timedelta(days=365),
                end=datetime.now() + timedelta(days=365),
                freq='D'
            )

        date_dim = pd.DataFrame({
            'date': dates,
            'day': dates.day,
            'month': dates.month,
            'quarter': dates.quarter,
            'year': dates.year,
            'day_of_week': dates.day_name(),
            'week_of_year': dates.isocalendar().week,
            'is_weekend': dates.dayofweek >= 5,
            'is_quarter_end': dates.is_quarter_end,
            'fiscal_year': dates.year,
            'fiscal_quarter': dates.quarter
        })

        logging.info(f" Generated {len(date_dim)} date records")
        logging.info("\n TRANSFORMATION COMPLETED")
    except Exception as e:
        logging.error(f"Error during transformation: {e}")
        return 
    
    logging.info("\n" + "="*70)
    logging.info("  CREATING ENTREPRENEURSHIP DATA WAREHOUSE SCHEMA (PostgreSQL)")
    logging.info("="*70)

    engine = get_sqlalchemy_engine()
    if engine is None:
        logging.error("\n Cannot proceed without database connection")
        return

    try:
        with engine.connect() as conn:
            conn.execute(text('''
                CREATE TABLE IF NOT EXISTS dim_startup (
                    startup_key SERIAL PRIMARY KEY,
                    startup_id INTEGER UNIQUE NOT NULL,
                    company_name TEXT,
                    founder_name TEXT,
                    industry TEXT,
                    founded_date DATE,
                    business_model TEXT,
                    employee_count INTEGER,
                    headquarters_city TEXT,
                    headquarters_country TEXT,
                    website TEXT,
                    stage TEXT,
                    source_system TEXT
                )
            '''))

            conn.execute(text('''
                CREATE TABLE IF NOT EXISTS dim_investor (
                    investor_key SERIAL PRIMARY KEY,
                    investor_id INTEGER UNIQUE NOT NULL,
                    investor_name TEXT,
                    investor_type TEXT,
                    total_aum REAL,
                    founded_year INTEGER,
                    headquarters TEXT,
                    investment_focus TEXT,
                    stage_preference TEXT,
                    contact_email TEXT,
                    source_system TEXT
                )
            '''))

            conn.execute(text('''
                CREATE TABLE IF NOT EXISTS dim_accelerator (
                    accelerator_key SERIAL PRIMARY KEY,
                    accelerator_id TEXT UNIQUE NOT NULL,
                    accelerator_name TEXT,
                    location TEXT,
                    cohort TEXT,
                    year INTEGER,
                    equity_taken REAL,
                    startups_accepted INTEGER,
                    funding_provided REAL,
                    success_rate REAL,
                    source_system TEXT
                )
            '''))

            conn.execute(text('''
                CREATE TABLE IF NOT EXISTS dim_date (
                    date_key SERIAL PRIMARY KEY,
                    date DATE UNIQUE NOT NULL,
                    day INTEGER,
                    month INTEGER,
                    quarter INTEGER,
                    year INTEGER,
                    day_of_week TEXT,
                    week_of_year INTEGER,
                    is_weekend BOOLEAN,
                    is_quarter_end BOOLEAN,
                    fiscal_year INTEGER,
                    fiscal_quarter INTEGER
                )
            '''))

            conn.execute(text('''
                CREATE TABLE IF NOT EXISTS fact_funding (
                    funding_key SERIAL PRIMARY KEY,
                    funding_id INTEGER UNIQUE NOT NULL,
                    startup_key INTEGER,
                    investor_key INTEGER,
                    date_key INTEGER,
                    round_type TEXT,
                    amount_raised REAL,
                    valuation REAL,
                    valuation_multiple REAL,
                    number_of_investors INTEGER,
                    source_system TEXT,
                    FOREIGN KEY (startup_key) REFERENCES dim_startup(startup_key),
                    FOREIGN KEY (investor_key) REFERENCES dim_investor(investor_key),
                    FOREIGN KEY (date_key) REFERENCES dim_date(date_key)
                )
            '''))

            conn.execute(text('''
                CREATE TABLE IF NOT EXISTS fact_revenue (
                    revenue_key SERIAL PRIMARY KEY,
                    metric_id INTEGER UNIQUE NOT NULL,
                    startup_key INTEGER,
                    date_key INTEGER,
                    monthly_revenue REAL,
                    customer_count INTEGER,
                    churn_rate REAL,
                    customer_acquisition_cost REAL,
                    lifetime_value REAL,
                    burn_rate REAL,
                    ltv_cac_ratio REAL,
                    runway_months REAL,
                    source_system TEXT,
                    FOREIGN KEY (startup_key) REFERENCES dim_startup(startup_key),
                    FOREIGN KEY (date_key) REFERENCES dim_date(date_key)
                )
            '''))

            conn.execute(text('''
                CREATE TABLE IF NOT EXISTS fact_market_data (
                    market_key SERIAL PRIMARY KEY,
                    ticker TEXT,
                    company_name TEXT,
                    current_price REAL,
                    market_cap REAL,
                    market_cap_billions REAL,
                    volume BIGINT,
                    price_change_30d REAL,
                    sector TEXT,
                    fetch_date TIMESTAMP,
                    source_system TEXT
                )
            '''))

            conn.commit()
            logging.info(" Data warehouse schema created (PostgreSQL)!")

            logging.info("  Clearing existing data...")
            conn.execute(text("TRUNCATE TABLE fact_funding RESTART IDENTITY CASCADE"))
            conn.execute(text("TRUNCATE TABLE fact_revenue RESTART IDENTITY CASCADE"))
            conn.execute(text("TRUNCATE TABLE fact_market_data RESTART IDENTITY CASCADE"))
            conn.execute(text("TRUNCATE TABLE dim_startup RESTART IDENTITY CASCADE"))
            conn.execute(text("TRUNCATE TABLE dim_investor RESTART IDENTITY CASCADE"))
            conn.execute(text("TRUNCATE TABLE dim_accelerator RESTART IDENTITY CASCADE"))
            conn.execute(text("TRUNCATE TABLE dim_date RESTART IDENTITY CASCADE"))
            conn.commit()
            logging.info(" Existing data cleared")

    except Exception as e:
        logging.error(f"Error during DWH schema creation or data clearing: {e}")
        if engine:
            engine.dispose()
        return


    logging.info("\n" + "="*70)
    logging.info("  LOADING PHASE (PostgreSQL)")
    logging.info("="*70)

    logging.info("\n[5/5] Loading data into data warehouse...")

    try:
        logging.info("\nLoading dim_startup...")
        startups_df.to_sql('dim_startup', engine, if_exists='append', index=False)
        startup_map = pd.read_sql('SELECT startup_id, startup_key FROM dim_startup', engine)
        logging.info(f" Loaded {len(startups_df)} startups")

        logging.info("Loading dim_investor...")
        investors_df.to_sql('dim_investor', engine, if_exists='append', index=False)
        investor_map = pd.read_sql('SELECT investor_id, investor_key FROM dim_investor', engine)
        logging.info(f" Loaded {len(investors_df)} investors")

        logging.info("Loading dim_accelerator...")
        accelerators_df = accelerators_df.drop_duplicates(subset=['accelerator_id'])
        accelerators_df.to_sql('dim_accelerator', engine, if_exists='append', index=False)
        accelerator_map = pd.read_sql('SELECT accelerator_id, accelerator_key FROM dim_accelerator', engine)
        logging.info(f" Loaded {len(accelerators_df)} accelerator programs")

        logging.info("Loading dim_date...")
        date_dim.to_sql('dim_date', engine, if_exists='append', index=False)
        date_map = pd.read_sql('SELECT date, date_key FROM dim_date', engine)
        logging.info(f" Loaded {len(date_dim)} dates")

        logging.info("\nLoading fact_funding...")
        funding_enriched = funding_df.copy()
        funding_enriched = funding_enriched.merge(startup_map, on='startup_id', how='left')

        if not investor_map.empty:
            investor_keys = investor_map['investor_key'].sample(len(funding_enriched), replace=True, random_state=42).values
            funding_enriched['investor_key'] = investor_keys
        else:
             funding_enriched['investor_key'] = None

        date_map['date'] = pd.to_datetime(date_map['date'])
        funding_enriched['funding_date'] = pd.to_datetime(funding_enriched['funding_date'], errors='coerce')
        funding_enriched = funding_enriched.merge(
            date_map[['date', 'date_key']],
            left_on='funding_date',
            right_on='date',
            how='left'
        )

        fact_funding = funding_enriched[[
            'funding_id', 'startup_key', 'investor_key', 'date_key',
            'round_type', 'amount_raised', 'valuation', 'valuation_multiple',
            'number_of_investors'
        ]].copy()
        fact_funding['source_system'] = 'OLTP1_Startups'
        fact_funding = fact_funding.dropna(subset=['startup_key', 'date_key'])
        fact_funding = fact_funding.where(pd.notna(fact_funding), None)
        fact_funding.to_sql('fact_funding', engine, if_exists='append', index=False)
        logging.info(f" Loaded {len(fact_funding)} funding transactions")

        logging.info("Loading fact_revenue...")
        revenue_enriched = revenue_df.copy()
        revenue_enriched = revenue_enriched.merge(startup_map, on='startup_id', how='left')

        revenue_enriched['reporting_date'] = pd.to_datetime(revenue_enriched['reporting_date'], errors='coerce')
        revenue_enriched = revenue_enriched.merge(
            date_map[['date', 'date_key']],
            left_on='reporting_date',
            right_on='date',
            how='left'
        )

        revenue_enriched['ltv_cac_ratio'] = (
            revenue_enriched['lifetime_value'] / revenue_enriched['customer_acquisition_cost']
        ).replace([np.inf, -np.inf], np.nan).round(2)

        revenue_enriched['runway_months'] = (
            revenue_enriched['monthly_revenue'].fillna(0) / revenue_enriched['burn_rate'].replace(0, np.nan) * 12
        ).replace([np.inf, -np.inf], np.nan).round(1)

        fact_revenue = revenue_enriched[[
            'metric_id', 'startup_key', 'date_key', 'monthly_revenue',
            'customer_count', 'churn_rate', 'customer_acquisition_cost',
            'lifetime_value', 'burn_rate', 'ltv_cac_ratio', 'runway_months'
        ]].copy()
        fact_revenue['source_system'] = 'OLTP1_Startups'
        fact_revenue = fact_revenue.dropna(subset=['startup_key', 'date_key'])
        fact_revenue = fact_revenue.where(pd.notna(fact_revenue), None)
        fact_revenue.to_sql('fact_revenue', engine, if_exists='append', index=False)
        logging.info(f" Loaded {len(fact_revenue)} revenue metrics")

        logging.info("Loading fact_market_data (REAL-TIME API DATA)...")
        market_api_df['fetch_date'] = pd.to_datetime(market_api_df['fetch_date'], errors='coerce')
        market_api_df.to_sql('fact_market_data', engine, if_exists='append', index=False)
        logging.info(f" Loaded {len(market_api_df)} live market records from Yahoo Finance API")

        engine.dispose()

    except Exception as e:
        logging.error(f"Error during loading: {e}")
        if engine:
            engine.dispose()
        return


    logging.info("\n" + "="*70)
    logging.info("  ETL PIPELINE COMPLETED SUCCESSFULLY!")
    logging.info("="*70)
    logging.info("\n Data Integration Summary:")
    logging.info(f"   Source 1 (OLTP Startups): {len(startups_df)} companies, {len(funding_df)} rounds")
    logging.info(f"   Source 2 (OLTP Investors): {len(investors_df)} investors")
    logging.info(f"   Source 3 (CSV Accelerators): {len(accelerators_df)} programs")
    logging.info(f"   Source 4 (CSV Economic Data): {len(economic_df)} indicators")
    logging.info(f"   Source 5 (LIVE API - Yahoo Finance): {len(market_api_df)} securities")
    logging.info("\n Data Warehouse:")
    logging.info(f"   Fact Tables: {len(fact_funding)} funding + {len(fact_revenue)} revenue + {len(market_api_df)} market records")
    logging.info(f"   Dimension Tables: 4 tables")
    logging.info(f"   Database Type: PostgreSQL (via Supabase)")
    if 'api_metadata' in locals() and api_metadata:
        logging.info("\n Real-Time Integration:")
        logging.info(f"   API Source: Yahoo Finance")
        logging.info(f"   API Timestamp: {api_metadata.get('fetch_timestamp', 'N/A')}")
        logging.info(f"   Securities Tracked: {len(market_api_df)}")
        logging.info(f"   Market Volatility: Integrated into economic indicators")
    logging.info("="*70)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    run_etl()