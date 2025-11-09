import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import sqlite3
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OLTP1_DIR = os.path.join(DATA_DIR, 'oltp1_startups')
OLTP2_DIR = os.path.join(DATA_DIR, 'oltp2_investors')
CSV_DIR = os.path.join(DATA_DIR, 'csv_source')

fake = Faker()
random.seed(int(datetime.now().timestamp()) % 10000)
np.random.seed(int(datetime.now().timestamp()) % 10000)


def check_data_exists():
    """Check if initial data files exist"""
    startup_db = os.path.join(OLTP1_DIR, 'startup_oltp.db')
    investor_db = os.path.join(OLTP2_DIR, 'investor_oltp.db')
    
    if not os.path.exists(startup_db) or not os.path.exists(investor_db):
        logging.error("="*70)
        logging.error("  ERROR: Initial data not found!")
        logging.error("="*70)
        logging.error("Please run 'python data_generation.py' first to create initial data.")
        logging.error("="*70)
        return False
    return True


def add_new_startups(num_new=5):
    """Add new startups to simulate company formations"""
    conn = sqlite3.connect(os.path.join(OLTP1_DIR, 'startup_oltp.db'))
    cursor = conn.cursor()
    
    industries = ['FinTech', 'HealthTech', 'EdTech', 'E-commerce', 'SaaS', 'AI/ML', 'CleanTech', 'FoodTech']
    business_models = ['B2B', 'B2C', 'B2B2C', 'Marketplace', 'Subscription', 'Freemium', 'Enterprise']
    stages = ['Pre-Seed', 'Seed']
    
    logging.info(f"\n Adding {num_new} new startups...")
    
    for i in range(num_new):
        cursor.execute('''
            INSERT INTO startups (company_name, founder_name, industry, founded_date,
                                 business_model, employee_count, headquarters_city,
                                 headquarters_country, website, stage)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            f"{fake.company()} {random.choice(['Tech', 'Labs', 'AI', 'Solutions', 'Systems'])}",
            fake.name(),
            random.choice(industries),
            (datetime.now() - timedelta(days=random.randint(30, 180))).strftime('%Y-%m-%d'),
            random.choice(business_models),
            random.randint(3, 25),
            fake.city(),
            random.choice(['USA', 'UK', 'Canada', 'Germany', 'Singapore']),
            f"www.{fake.domain_name()}",
            random.choice(stages)
        ))
    
    conn.commit()
    
    new_startups = pd.read_sql(
        "SELECT startup_id FROM startups ORDER BY startup_id DESC LIMIT ?", 
        conn, 
        params=(num_new,)
    )
    
    conn.close()
    logging.info(f"  Added {num_new} new startups (IDs: {new_startups['startup_id'].tolist()})")
    return new_startups['startup_id'].tolist()


def add_new_funding_rounds(num_rounds=None):
    """Add new funding rounds to simulate daily investment activity"""
    conn = sqlite3.connect(os.path.join(OLTP1_DIR, 'startup_oltp.db'))
    cursor = conn.cursor()
    
    startups = pd.read_sql("SELECT startup_id, stage FROM startups", conn)
    
    if startups.empty:
        logging.warning(" No startups found. Skipping funding rounds.")
        conn.close()
        return
    
    if num_rounds is None:
        num_rounds = random.randint(5, 15)
    
    round_types = ['Pre-Seed', 'Seed', 'Series A', 'Series B', 'Series C', 'Bridge']
    
    logging.info(f"\n Adding {num_rounds} new funding rounds...")
    
    added_count = 0
    for _ in range(num_rounds):
        startup = startups.sample(1).iloc[0]
        startup_id = startup['startup_id']
        current_stage = startup['stage']
        
        if current_stage in ['Pre-Seed', 'Seed']:
            round_type = random.choice(['Pre-Seed', 'Seed'])
            amount = round(random.uniform(100000, 2000000), 2)
        elif current_stage == 'Series A':
            round_type = random.choice(['Seed', 'Series A', 'Bridge'])
            amount = round(random.uniform(2000000, 10000000), 2)
        elif current_stage == 'Series B':
            round_type = random.choice(['Series A', 'Series B', 'Bridge'])
            amount = round(random.uniform(10000000, 30000000), 2)
        else:
            round_type = random.choice(['Series B', 'Series C', 'Bridge'])
            amount = round(random.uniform(20000000, 100000000), 2)
        
        valuation = amount * random.uniform(3, 10)
        
        try:
            cursor.execute('''
                INSERT INTO funding_rounds 
                (startup_id, round_type, amount_raised, valuation, funding_date, 
                 lead_investor, number_of_investors)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                startup_id,
                round_type,
                amount,
                round(valuation, 2),
                datetime.now().strftime('%Y-%m-%d'),
                fake.company() + " Ventures",
                random.randint(1, 12)
            ))
            added_count += 1
        except sqlite3.IntegrityError:
            continue
    
    conn.commit()
    conn.close()
    logging.info(f"  Added {added_count} new funding rounds")


def update_revenue_metrics():
    """Update revenue metrics for all startups (monthly growth simulation)"""
    conn = sqlite3.connect(os.path.join(OLTP1_DIR, 'startup_oltp.db'))
    cursor = conn.cursor()
    
    startups = pd.read_sql("SELECT startup_id FROM startups", conn)
    
    if startups.empty:
        logging.warning(" No startups found. Skipping revenue updates.")
        conn.close()
        return
    
    logging.info(f"\n Updating revenue metrics for {len(startups)} startups...")
    
    updated_count = 0
    for startup_id in startups['startup_id']:
        latest = pd.read_sql(f"""
            SELECT monthly_revenue, customer_count, burn_rate 
            FROM revenue_metrics 
            WHERE startup_id = {startup_id} 
            ORDER BY reporting_date DESC LIMIT 1
        """, conn)
        
        if not latest.empty:
            base_revenue = latest['monthly_revenue'].iloc[0]
            base_customers = latest['customer_count'].iloc[0]
            base_burn = latest['burn_rate'].iloc[0]
            
            growth = random.uniform(0.95, 1.20)  
            new_revenue = base_revenue * growth
            new_customers = int(base_customers * random.uniform(0.98, 1.15))
            new_burn = base_burn * random.uniform(0.95, 1.10)
            
            cac = round(random.uniform(50, 500), 2)
            ltv = round(random.uniform(500, 5000), 2)
            
            cursor.execute('''
                INSERT INTO revenue_metrics 
                (startup_id, reporting_date, monthly_revenue, customer_count, 
                 churn_rate, customer_acquisition_cost, lifetime_value, burn_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                startup_id,
                datetime.now().strftime('%Y-%m-%d'),
                round(new_revenue, 2),
                new_customers,
                round(random.uniform(2, 15), 2),
                cac,
                ltv,
                round(new_burn, 2)
            ))
            updated_count += 1
        else:
            cursor.execute('''
                INSERT INTO revenue_metrics 
                (startup_id, reporting_date, monthly_revenue, customer_count, 
                 churn_rate, customer_acquisition_cost, lifetime_value, burn_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                startup_id,
                datetime.now().strftime('%Y-%m-%d'),
                round(random.uniform(10000, 100000), 2),
                random.randint(10, 500),
                round(random.uniform(2, 15), 2),
                round(random.uniform(50, 500), 2),
                round(random.uniform(500, 5000), 2),
                round(random.uniform(50000, 200000), 2)
            ))
            updated_count += 1
    
    conn.commit()
    conn.close()
    logging.info(f"  Updated revenue metrics for {updated_count} startups")


def add_new_investors(num_new=3):
    """Add new investors/VCs to the market"""
    conn = sqlite3.connect(os.path.join(OLTP2_DIR, 'investor_oltp.db'))
    cursor = conn.cursor()
    
    investor_types = ['Venture Capital', 'Angel Investor', 'Corporate VC', 'Private Equity']
    industries = ['FinTech', 'HealthTech', 'EdTech', 'E-commerce', 'SaaS', 'AI/ML', 'CleanTech', 'FoodTech']
    stage_preferences = ['Early Stage', 'Growth Stage', 'Late Stage', 'All Stages']
    
    logging.info(f"\n Adding {num_new} new investors...")
    
    for _ in range(num_new):
        cursor.execute('''
            INSERT INTO investors 
            (investor_name, investor_type, total_aum, founded_year, headquarters, 
             investment_focus, stage_preference, contact_email)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            f"{fake.company()} {random.choice(['Ventures', 'Capital', 'Partners', 'Investments'])}",
            random.choice(investor_types),
            round(random.uniform(10000000, 500000000), 2),
            random.randint(2015, 2024),
            f"{fake.city()}, {random.choice(['USA', 'UK', 'Singapore'])}",
            random.choice(industries),
            random.choice(stage_preferences),
            fake.email()
        ))
    
    conn.commit()
    conn.close()
    logging.info(f"  Added {num_new} new investors")


def update_market_trends():
    """Update quarterly market trends"""
    conn = sqlite3.connect(os.path.join(OLTP2_DIR, 'investor_oltp.db'))
    cursor = conn.cursor()
    
    industries = ['FinTech', 'HealthTech', 'EdTech', 'E-commerce', 'SaaS', 'AI/ML', 'CleanTech', 'FoodTech']
    stages = ['Pre-Seed', 'Seed', 'Series A', 'Series B', 'Series C', 'Growth']
    
    now = datetime.now()
    quarter = f"Q{(now.month-1)//3 + 1} {now.year}"
    
    logging.info(f"\n Updating market trends for {quarter}...")
    
    added_count = 0
    for industry in industries:
        existing = pd.read_sql(
            f"SELECT * FROM market_trends WHERE industry = '{industry}' AND quarter = '{quarter}'",
            conn
        )
        
        if existing.empty:
            total_deals = random.randint(50, 300)
            total_invested = round(random.uniform(500000000, 5000000000), 2)
            
            cursor.execute('''
                INSERT INTO market_trends 
                (industry, quarter, total_deals, total_amount_invested,
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
            added_count += 1
    
    conn.commit()
    conn.close()
    
    if added_count > 0:
        logging.info(f"  Added market trends for {added_count} industries in {quarter}")
    else:
        logging.info(f"  Market trends for {quarter} already exist")


def update_csv_files():
    """Update CSV files with new data"""
    logging.info("\n Updating CSV files...")
    
    econ_file = os.path.join(CSV_DIR, 'economic_indicators.csv')
    
    if os.path.exists(econ_file):
        econ_df = pd.read_csv(econ_file)
        econ_df['date'] = pd.to_datetime(econ_df['date'])
        current_month = datetime.now().replace(day=1)
        
        if current_month not in econ_df['date'].values:
            new_record = {
                'indicator_id': f"IND{len(econ_df):04d}",
                'date': current_month.strftime('%Y-%m-%d'),
                'gdp_growth_rate': round(random.uniform(2, 4), 2),
                'unemployment_rate': round(random.uniform(3.5, 6), 1),
                'inflation_rate': round(random.uniform(2, 6), 2),
                'interest_rate': round(random.uniform(2, 5.5), 2),
                'market_volatility_index': round(random.uniform(10, 30), 2),
                'venture_capital_index': round(random.uniform(95, 115), 1),
                'startup_confidence_score': round(random.uniform(60, 85), 1),
                'data_source': 'Combined: Government Data + Yahoo Finance API'
            }
            
            econ_df = pd.concat([econ_df, pd.DataFrame([new_record])], ignore_index=True)
            econ_df.to_csv(econ_file, index=False)
            logging.info(f"  Updated economic indicators (added {current_month.strftime('%Y-%m')})")
        else:
            logging.info(" Economic indicators already up to date")
    
    comp_file = os.path.join(CSV_DIR, 'competitors.csv')
    
    if os.path.exists(comp_file):
        comp_df = pd.read_csv(comp_file)
        
        num_new_competitors = random.randint(2, 5)
        industries = ['FinTech', 'HealthTech', 'EdTech', 'E-commerce', 'SaaS', 'AI/ML']
        stages = ['Pre-Seed', 'Seed', 'Series A', 'Series B', 'Series C', 'Growth']
        
        new_competitors = []
        for i in range(num_new_competitors):
            new_competitors.append({
                'competitor_id': f"COMP{len(comp_df) + i:06d}",
                'company_name': fake.company(),
                'industry': random.choice(industries),
                'estimated_revenue': round(random.uniform(100000, 50000000), 2),
                'employee_count': random.randint(10, 5000),
                'market_share': round(random.uniform(0.5, 15), 2),
                'growth_rate': round(random.uniform(-10, 100), 2),
                'funding_stage': random.choice(stages),
                'competitive_advantage': random.choice(['Technology', 'Market Position', 'Cost', 'Brand', 'Network Effect'])
            })
        
        comp_df = pd.concat([comp_df, pd.DataFrame(new_competitors)], ignore_index=True)
        comp_df.to_csv(comp_file, index=False)
        logging.info(f"  Updated competitors (added {num_new_competitors} companies)")


def generate_update_summary():
    """Generate summary of data updates"""
    logging.info("\n" + "="*70)
    logging.info("  UPDATE SUMMARY")
    logging.info("="*70)
    conn1 = sqlite3.connect(os.path.join(OLTP1_DIR, 'startup_oltp.db'))
    startups_count = pd.read_sql("SELECT COUNT(*) as count FROM startups", conn1)['count'].iloc[0]
    funding_count = pd.read_sql("SELECT COUNT(*) as count FROM funding_rounds", conn1)['count'].iloc[0]
    revenue_count = pd.read_sql("SELECT COUNT(*) as count FROM revenue_metrics", conn1)['count'].iloc[0]
    conn1.close()
    
    conn2 = sqlite3.connect(os.path.join(OLTP2_DIR, 'investor_oltp.db'))
    investors_count = pd.read_sql("SELECT COUNT(*) as count FROM investors", conn2)['count'].iloc[0]
    portfolio_count = pd.read_sql("SELECT COUNT(*) as count FROM portfolio_companies", conn2)['count'].iloc[0]
    trends_count = pd.read_sql("SELECT COUNT(*) as count FROM market_trends", conn2)['count'].iloc[0]
    conn2.close()
    
    logging.info(f"\n Current Data Counts:")
    logging.info(f"   Startups: {startups_count}")
    logging.info(f"   Funding Rounds: {funding_count}")
    logging.info(f"   Revenue Metrics: {revenue_count}")
    logging.info(f"   Investors: {investors_count}")
    logging.info(f"   Portfolio Companies: {portfolio_count}")
    logging.info(f"   Market Trends: {trends_count}")
    logging.info(f"\n Update Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("="*70)


def main():
    """Main function to run incremental updates"""
    logging.info("="*70)
    logging.info("  INCREMENTAL SOURCE DATA UPDATE")
    logging.info("="*70)
    logging.info(f" Date: {datetime.now().strftime('%Y-%m-%d')}")
    logging.info(f" Time: {datetime.now().strftime('%H:%M:%S')}")
    logging.info("="*70)
    if not check_data_exists():
        return
    
    try:
        add_new_startups(num_new=random.randint(3, 7))
        add_new_funding_rounds()
        update_revenue_metrics()
        add_new_investors(num_new=random.randint(2, 4))
        update_market_trends()
        update_csv_files()
        
        generate_update_summary()
        
        logging.info("\n" + "="*70)
        logging.info("  INCREMENTAL UPDATE COMPLETED SUCCESSFULLY!")
        logging.info("="*70)
        
    except Exception as e:
        logging.error(f"\n Error during update: {e}")
        logging.error("="*70)
        raise


if __name__ == "__main__":
    main()