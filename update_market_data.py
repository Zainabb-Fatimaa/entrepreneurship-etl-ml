import pandas as pd
import yfinance as yf
from datetime import datetime
import time
import os
import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# PostgreSQL connection details
DB_HOST = os.getenv('DB_HOST')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_PORT = os.getenv('DB_PORT', '6543')
DB_SSLMODE = os.getenv('DB_SSLMODE', 'require')

# Define base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
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
        logging.error("1. Verify your Session Pooler settings in .env or GitHub Secrets")
        logging.error("2. Ensure DB_PORT=6543 (Session Pooler port)")
        logging.error("3. Check DB_SSLMODE=require")
        logging.error("="*70)
        return None


def fetch_live_market_data():
    """Fetch live market data from Yahoo Finance API"""
    logging.info("="*70)
    logging.info("  REAL-TIME MARKET DATA UPDATE")
    logging.info("="*70)
    
    # Create cache directory
    os.makedirs(API_CACHE_DIR, exist_ok=True)
    
    # Tickers to track
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'META', 'AMZN', 'NFLX', 
               '^GSPC', '^IXIC', '^DJI']
    
    market_data = []
    
    logging.info(f"\n Fetching live data for {len(tickers)} securities...")
    logging.info(f" Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
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
                    'sector': info.get('sector', 'Index' if ticker.startswith('^') else 'Technology'),
                    'fetch_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source_system': 'API_YahooFinance'
                })
                
                logging.info(f"    {ticker:>8} | ${current_price:>10.2f} | {price_change:>+7.2f}%")
            
            time.sleep(0.5)  
            
        except Exception as e:
            logging.warning(f"    {ticker:>8} | Error: {str(e)[:50]}")
    
    if market_data:
        market_df = pd.DataFrame(market_data)
        
        csv_path = os.path.join(API_CACHE_DIR, 'live_market_data.csv')
        market_df.to_csv(csv_path, index=False)
        logging.info(f"\n  Cached to: {csv_path}")
        
        return market_df
    else:
        logging.error("\n  No market data fetched")
        return None


def update_database(market_df):
    """Update PostgreSQL database with fresh market data"""
    if market_df is None or market_df.empty:
        logging.warning(" Skipping database update - no data")
        return False
    
    engine = get_sqlalchemy_engine()
    if engine is None:
        return False
    
    try:
        with engine.connect() as conn:
            # Clear old market data
            result = conn.execute(text("DELETE FROM fact_market_data"))
            conn.commit()
            deleted_count = result.rowcount
            logging.info(f"\n Cleared {deleted_count} old market records")
            
            market_df['market_cap_billions'] = (market_df['market_cap'] / 1e9).round(2)
            
            market_df.to_sql('fact_market_data', engine, if_exists='append', index=False)
            
            logging.info(f"  Inserted {len(market_df)} new market records")
            
            logging.info("\n Market Data Summary:")
            logging.info(f"   Average Price Change: {market_df['price_change_30d'].mean():+.2f}%")
            logging.info(f"   Best Performer: {market_df.loc[market_df['price_change_30d'].idxmax(), 'ticker']} ({market_df['price_change_30d'].max():+.2f}%)")
            logging.info(f"   Worst Performer: {market_df.loc[market_df['price_change_30d'].idxmin(), 'ticker']} ({market_df['price_change_30d'].min():+.2f}%)")
        
        engine.dispose()
        return True
        
    except Exception as e:
        logging.error(f"  Database update failed: {e}")
        if engine:
            engine.dispose()
        return False


def main():
    """Main execution function"""
    start_time = datetime.now()
    
    # Fetch live data
    market_df = fetch_live_market_data()
    
    # Update database
    if market_df is not None:
        success = update_database(market_df)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logging.info("\n" + "="*70)
        if success:
            logging.info("  MARKET DATA UPDATE COMPLETED SUCCESSFULLY!")
            logging.info(f"  Duration: {duration:.2f} seconds")
            logging.info(f"  Records Updated: {len(market_df)}")
            logging.info(f"  Next Update: Check your workflow schedule")
        else:
            logging.info("  MARKET DATA UPDATE FAILED")
            logging.info(f"  Duration: {duration:.2f} seconds")
        logging.info("="*70)
    else:
        logging.error("\n" + "="*70)
        logging.error("  MARKET DATA UPDATE FAILED - NO DATA FETCHED")
        logging.error("="*70)


if __name__ == "__main__":
    main()