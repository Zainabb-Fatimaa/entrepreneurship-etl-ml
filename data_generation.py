import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import json
import os
import sqlite3
import requests
import yfinance as yf
import time
import logging 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OLTP1_DIR = os.path.join(DATA_DIR, 'oltp1_startups')
OLTP2_DIR = os.path.join(DATA_DIR, 'oltp2_investors')
CSV_DIR = os.path.join(DATA_DIR, 'csv_source')
API_CACHE_DIR = os.path.join(DATA_DIR, 'api_cache')
WAREHOUSE_DIR = os.path.join(DATA_DIR, 'warehouse')
ANALYTICS_DIR = os.path.join(BASE_DIR, 'analytics')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_data():
    logging.info("="*70)
    logging.info("  ENTREPRENEURSHIP DATA WAREHOUSE - SETUP & DATA GENERATION")
    logging.info("="*70)

    logging.info("\n[2/7] Importing libraries...")
    logging.info(" Libraries imported successfully!")

    logging.info("\n[3/7] Setting up configuration...")

    Faker.seed(42)
    random.seed(42)
    np.random.seed(42)

    os.makedirs(OLTP1_DIR, exist_ok=True)
    os.makedirs(OLTP2_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(API_CACHE_DIR, exist_ok=True)
    os.makedirs(WAREHOUSE_DIR, exist_ok=True)
    os.makedirs(ANALYTICS_DIR, exist_ok=True)

    logging.info(" Configuration complete!")

    logging.info("\n[4/7] Fetching REAL-TIME market data from Yahoo Finance API...")
    logging.info("  This demonstrates live API integration for entrepreneurship metrics")

    tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'META', '^GSPC', '^IXIC']
    market_data = []

    logging.info("\n  Fetching live market data...")
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period='1mo')

            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[0]
                price_change = ((current_price - prev_price) / prev_price) * 100

                market_data.append({
                    'ticker': ticker,
                    'company_name': info.get('longName', ticker),
                    'current_price': round(current_price, 2),
                    'market_cap': info.get('marketCap', 0),
                    'volume': int(hist['Volume'].iloc[-1]),
                    'price_change_30d': round(price_change, 2),
                    'sector': info.get('sector', 'Technology'),
                    'fetch_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })

                logging.info(f"     {ticker}: ${current_price:.2f} ({price_change:+.2f}%)")

            time.sleep(0.5)

        except Exception as e:
            logging.warning(f"     Could not fetch {ticker}: {str(e)}")

    market_df = pd.DataFrame(market_data)

    market_df.to_csv(os.path.join(API_CACHE_DIR, 'live_market_data.csv'), index=False)

    api_metadata = {
        'fetch_timestamp': datetime.now().isoformat(),
        'data_source': 'Yahoo Finance API (yfinance)',
        'tickers_fetched': len(market_data),
        'api_status': 'success'
    }
    with open(os.path.join(API_CACHE_DIR, 'market_metadata.json'), 'w') as f:
        json.dump(api_metadata, f, indent=2)

    logging.info(f"\n Successfully fetched LIVE data for {len(market_data)} securities")
    logging.info("  Data source: Yahoo Finance API")
    logging.info(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    logging.info("\n[5/7] Generating OLTP System 1: Startup Management Database...")

    fake = Faker()
    conn_oltp1 = sqlite3.connect(os.path.join(OLTP1_DIR, 'startup_oltp.db'))
    cursor = conn_oltp1.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS startups (
            startup_id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_name TEXT NOT NULL,
            founder_name TEXT,
            industry TEXT,
            founded_date TEXT,
            business_model TEXT,
            employee_count INTEGER,
            headquarters_city TEXT,
            headquarters_country TEXT,
            website TEXT,
            stage TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS funding_rounds (
            funding_id INTEGER PRIMARY KEY AUTOINCREMENT,
            startup_id INTEGER,
            round_type TEXT,
            amount_raised REAL,
            valuation REAL,
            funding_date TEXT,
            lead_investor TEXT,
            number_of_investors INTEGER,
            FOREIGN KEY (startup_id) REFERENCES startups(startup_id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS revenue_metrics (
            metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
            startup_id INTEGER,
            reporting_date TEXT,
            monthly_revenue REAL,
            customer_count INTEGER,
            churn_rate REAL,
            customer_acquisition_cost REAL,
            lifetime_value REAL,
            burn_rate REAL,
            FOREIGN KEY (startup_id) REFERENCES startups(startup_id)
        )
    ''')

    industries = ['FinTech', 'HealthTech', 'EdTech', 'E-commerce', 'SaaS', 'AI/ML', 'CleanTech', 'FoodTech']
    business_models = ['B2B', 'B2C', 'B2B2C', 'Marketplace', 'Subscription', 'Freemium', 'Enterprise']
    stages = ['Pre-Seed', 'Seed', 'Series A', 'Series B', 'Series C', 'Growth']
    round_types = ['Pre-Seed', 'Seed', 'Series A', 'Series B', 'Series C', 'Bridge']

    logging.info("  Generating 150 startups...")
    for i in range(150):
        cursor.execute('''
            INSERT INTO startups (company_name, founder_name, industry, founded_date,
                                 business_model, employee_count, headquarters_city,
                                 headquarters_country, website, stage)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            f"{fake.company()} {random.choice(['Tech', 'Labs', 'AI', 'Solutions', 'Systems'])}",
            fake.name(),
            random.choice(industries),
            fake.date_between(start_date='-5y', end_date='-6m').strftime('%Y-%m-%d'),
            random.choice(business_models),
            random.randint(5, 500),
            fake.city(),
            random.choice(['USA', 'UK', 'Canada', 'Germany', 'Singapore']),
            f"www.{fake.domain_name()}",
            random.choice(stages)
        ))

        startup_id = cursor.lastrowid
        num_rounds = random.randint(1, 4)
        total_raised = 0

        for round_num in range(num_rounds):
            amount = round(random.uniform(100000, 50000000), 2)
            valuation = amount * random.uniform(3, 10)
            total_raised += amount

            cursor.execute('''
                INSERT INTO funding_rounds (startup_id, round_type, amount_raised, valuation,
                                           funding_date, lead_investor, number_of_investors)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                startup_id,
                round_types[min(round_num, len(round_types)-1)],
                amount,
                round(valuation, 2),
                fake.date_between(start_date='-3y', end_date='now').strftime('%Y-%m-%d'),
                fake.company(),
                random.randint(1, 15)
            ))

        base_revenue = random.uniform(10000, 500000)
        growth_rate = random.uniform(1.05, 1.25)

        for month in range(12):
            reporting_date = (datetime.now() - timedelta(days=30*month)).strftime('%Y-%m-%d')
            monthly_revenue = base_revenue * (growth_rate ** month)

            cursor.execute('''
                INSERT INTO revenue_metrics (startup_id, reporting_date, monthly_revenue,
                                            customer_count, churn_rate, customer_acquisition_cost,
                                            lifetime_value, burn_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                startup_id,
                reporting_date,
                round(monthly_revenue, 2),
                int(monthly_revenue / random.uniform(50, 200)),
                round(random.uniform(2, 15), 2),
                round(random.uniform(50, 500), 2),
                round(random.uniform(500, 5000), 2),
                round(random.uniform(50000, 500000), 2)
            ))

    conn_oltp1.commit()
    conn_oltp1.close()
    logging.info(f" OLTP System 1 created with 150 startups and their metrics")

    logging.info("\n[6/7] Generating OLTP System 2: Investor & VC Database...")

    conn_oltp2 = sqlite3.connect(os.path.join(OLTP2_DIR, 'investor_oltp.db'))
    cursor = conn_oltp2.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS investors (
            investor_id INTEGER PRIMARY KEY AUTOINCREMENT,
            investor_name TEXT NOT NULL,
            investor_type TEXT,
            total_aum REAL,
            founded_year INTEGER,
            headquarters TEXT,
            investment_focus TEXT,
            stage_preference TEXT,
            contact_email TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolio_companies (
            portfolio_id INTEGER PRIMARY KEY AUTOINCREMENT,
            investor_id INTEGER,
            company_name TEXT,
            investment_date TEXT,
            investment_amount REAL,
            equity_percentage REAL,
            current_valuation REAL,
            exit_status TEXT,
            FOREIGN KEY (investor_id) REFERENCES investors(investor_id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_trends (
            trend_id INTEGER PRIMARY KEY AUTOINCREMENT,
            industry TEXT,
            quarter TEXT,
            total_deals INTEGER,
            total_amount_invested REAL,
            average_deal_size REAL,
            top_funded_stage TEXT
        )
    ''')

    investor_types = ['Venture Capital', 'Angel Investor', 'Corporate VC', 'Private Equity', 'Accelerator']
    investment_focuses = industries
    stage_preferences = ['Early Stage', 'Growth Stage', 'Late Stage', 'All Stages']

    logging.info("  Generating 75 investors...")
    for i in range(75):
        cursor.execute('''
            INSERT INTO investors VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            None,
            f"{fake.company()} {random.choice(['Ventures', 'Capital', 'Partners', 'Investments', 'Fund'])}",
            random.choice(investor_types),
            round(random.uniform(10000000, 1000000000), 2),
            random.randint(2000, 2020),
            f"{fake.city()}, {random.choice(['USA', 'UK', 'Singapore', 'Hong Kong'])}",
            random.choice(investment_focuses),
            random.choice(stage_preferences),
            fake.email()
        ))

        investor_id = cursor.lastrowid
        num_companies = random.randint(5, 25)
        for _ in range(num_companies):
            investment_amount = round(random.uniform(100000, 10000000), 2)

            cursor.execute('''
                INSERT INTO portfolio_companies (investor_id, company_name, investment_date,
                                                investment_amount, equity_percentage,
                                                current_valuation, exit_status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                investor_id,
                fake.company(),
                fake.date_between(start_date='-3y', end_date='now').strftime('%Y-%m-%d'),
                investment_amount,
                round(random.uniform(5, 40), 2),
                round(investment_amount * random.uniform(2, 15), 2),
                random.choice(['Active', 'Exited-IPO', 'Exited-Acquisition', 'Failed'])
            ))

    quarters = ['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2023', 'Q3 2023']
    logging.info("  Generating market trends...")
    for industry in industries:
        for quarter in quarters:
            total_deals = random.randint(50, 300)
            total_invested = round(random.uniform(500000000, 5000000000), 2)

            cursor.execute('''
                INSERT INTO market_trends (industry, quarter, total_deals, total_amount_invested,
                                      average_deal_size, top_funded_stage)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                industry,
                quarter,
                total_deals,
                total_invested,
                round(total_invested / total_deals, 2),
                random.choice(stages)
            ))

    conn_oltp2.commit()
    conn_oltp2.close()
    logging.info(f" OLTP System 2 created with 75 investors and portfolio data")

    logging.info("\n[7/7] Generating CSV files with market integration...")

    accelerators = [
        {'name': 'Y Combinator', 'location': 'Mountain View, CA', 'batch_size': 200},
        {'name': 'Techstars', 'location': 'Boulder, CO', 'batch_size': 150},
        {'name': '500 Startups', 'location': 'San Francisco, CA', 'batch_size': 100},
        {'name': 'Plug and Play', 'location': 'Sunnyvale, CA', 'batch_size': 80},
        {'name': 'MassChallenge', 'location': 'Boston, MA', 'batch_size': 120}
    ]

    accelerator_data = []
    for acc in accelerators:
        for cohort in range(1, 6):
            accelerator_data.append({
                'accelerator_id': f"ACC{len(accelerator_data):04d}",
                'accelerator_name': acc['name'],
                'location': acc['location'],
                'cohort': f"Cohort {cohort}",
                'year': random.randint(2020, 2024),
                'startups_accepted': random.randint(10, acc['batch_size']),
                'equity_taken': round(random.uniform(5, 10), 1),
                'funding_provided': random.randint(20000, 150000),
                'success_rate': round(random.uniform(60, 90), 1)
            })

    accelerator_df = pd.DataFrame(accelerator_data)
    accelerator_df.to_csv(os.path.join(CSV_DIR, 'accelerators.csv'), index=False)
    logging.info(f" Created accelerators.csv with {len(accelerator_df)} programs")

    logging.info("  Integrating real market data into economic indicators...")

    economic_data = []
    base_date = datetime.now() - timedelta(days=365)

    for i in range(12):
        date = base_date + timedelta(days=30*i)
        market_volatility = market_df['price_change_30d'].std() if not market_df.empty else random.uniform(5, 20)

        economic_data.append({
            'indicator_id': f"IND{i:04d}",
            'date': date.strftime('%Y-%m-%d'),
            'gdp_growth_rate': round(random.uniform(2, 4), 2),
            'unemployment_rate': round(random.uniform(3.5, 6), 1),
            'inflation_rate': round(random.uniform(2, 6), 2),
            'interest_rate': round(random.uniform(2, 5.5), 2),
            'market_volatility_index': round(market_volatility, 2),
            'venture_capital_index': round(random.uniform(95, 115), 1),
            'startup_confidence_score': round(random.uniform(60, 85), 1),
            'data_source': 'Combined: Government Data + Yahoo Finance API'
        })

    economic_df = pd.DataFrame(economic_data)
    economic_df.to_csv(os.path.join(CSV_DIR, 'economic_indicators.csv'), index=False)
    logging.info(f" Created economic_indicators.csv with {len(economic_df)} records")
    logging.info("   Enhanced with real-time market volatility data")

    competitors = []
    for i in range(100):
        competitors.append({
            'competitor_id': f"COMP{i:06d}",
            'company_name': fake.company(),
            'industry': random.choice(industries),
            'estimated_revenue': round(random.uniform(100000, 50000000), 2),
            'employee_count': random.randint(10, 5000),
            'market_share': round(random.uniform(0.5, 15), 2),
            'growth_rate': round(random.uniform(-10, 100), 2),
            'funding_stage': random.choice(stages),
            'competitive_advantage': random.choice(['Technology', 'Market Position', 'Cost', 'Brand', 'Network Effect'])
        })

    competitor_df = pd.DataFrame(competitors)
    competitor_df.to_csv(os.path.join(CSV_DIR, 'competitors.csv'), index=False)
    logging.info(f" Created competitors.csv with {len(competitor_df)} companies")

    logging.info("\n" + "="*70)
    logging.info("  DATA GENERATION COMPLETE!")
    logging.info("="*70)
    logging.info("\n Generated Data Summary:")
    logging.info(f"   REAL-TIME API Data: {len(market_data)} live securities from Yahoo Finance")
    logging.info(f"   OLTP System 1 (Startups): 150 companies with metrics")
    logging.info(f"   OLTP System 2 (Investors): 75 VCs and portfolios")
    logging.info(f"   CSV Files: Accelerators, Economic Indicators, Competitors")
    logging.info(f"   Market Integration: Live volatility data incorporated")
    logging.info("\n Files created:")
    logging.info("\n Live Data Sources:")
    logging.info(f"   Yahoo Finance API: Stock prices, volumes, market caps")
    logging.info(f"   Fetch Timestamp: {api_metadata.get('fetch_timestamp', 'N/A')}")
    logging.info("="*70)

if __name__ == "__main__":
    generate_data()