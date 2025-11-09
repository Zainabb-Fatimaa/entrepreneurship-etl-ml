import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
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
ANALYTICS_DIR = os.path.join(BASE_DIR, 'analytics')


def get_sqlalchemy_engine():
    """Creates SQLAlchemy engine for database operations."""
    try:
        from urllib.parse import quote_plus
        
        if not DB_PASSWORD:
            raise ValueError("DB_PASSWORD is not set in environment variables")
        
        encoded_password = quote_plus(str(DB_PASSWORD))
        
        connection_string = f"postgresql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode={DB_SSLMODE}"
        engine = create_engine(connection_string, pool_pre_ping=True)
        
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
        logging.error("="*70)
        return None


def generate_analytics_dashboard():
    logging.info("\n" + "="*70)
    logging.info("  ENTREPRENEURSHIP DATA WAREHOUSE - ANALYTICS & ML")
    logging.info("="*70)

    logging.info("\n[1/7] Importing libraries...")
    logging.info(" Libraries imported!")

    logging.info("\n[2/7] Loading data from entrepreneurship data warehouse (PostgreSQL)...")

    engine = get_sqlalchemy_engine()
    if engine is None:
        return

    try:
        query = """
        SELECT
            ff.funding_id,
            ff.round_type,
            ff.amount_raised,
            ff.valuation,
            ff.valuation_multiple,
            ff.number_of_investors,
            d.date,
            d.month,
            d.year,
            d.quarter,
            s.startup_id,
            s.company_name,
            s.industry,
            s.business_model,
            s.employee_count,
            s.headquarters_country,
            s.stage,
            i.investor_name,
            i.investor_type,
            i.investment_focus
        FROM fact_funding ff
        JOIN dim_date d ON ff.date_key = d.date_key
        JOIN dim_startup s ON ff.startup_key = s.startup_key
        JOIN dim_investor i ON ff.investor_key = i.investor_key
        """

        funding_data = pd.read_sql(query, engine)
        funding_data['date'] = pd.to_datetime(funding_data['date'])
        logging.info(f" Loaded {len(funding_data)} funding records")

        query_revenue = """
        SELECT
            fr.monthly_revenue,
            fr.customer_count,
            fr.churn_rate,
            fr.customer_acquisition_cost,
            fr.lifetime_value,
            fr.burn_rate,
            fr.ltv_cac_ratio,
            fr.runway_months,
            d.date,
            d.month,
            d.year,
            s.company_name,
            s.industry,
            s.stage
        FROM fact_revenue fr
        JOIN dim_date d ON fr.date_key = d.date_key
        JOIN dim_startup s ON fr.startup_key = s.startup_key
        """

        revenue_data = pd.read_sql(query_revenue, engine)
        revenue_data['date'] = pd.to_datetime(revenue_data['date'])
        logging.info(f" Loaded {len(revenue_data)} revenue metrics")

        query_market = """
        SELECT
            ticker,
            company_name,
            current_price,
            market_cap_billions,
            volume,
            price_change_30d,
            sector,
            fetch_date,
            source_system
        FROM fact_market_data
        """

        market_data = pd.read_sql(query_market, engine)
        logging.info(f" Loaded {len(market_data)} LIVE market records from API")

        engine.dispose()
        
    except Exception as e:
        logging.error(f" Error loading data from data warehouse: {e}")
        if engine:
            engine.dispose()
        return


    logging.info("\n" + "="*70)
    logging.info("  KEY PERFORMANCE INDICATORS (KPIs)")
    logging.info("="*70)

    if not funding_data.empty:
        total_funding = funding_data['amount_raised'].sum()
        total_deals = len(funding_data)
        avg_deal_size = funding_data['amount_raised'].mean()
        avg_valuation = funding_data['valuation'].mean()
        unique_startups = funding_data['startup_id'].nunique()
        unique_investors = funding_data['investor_name'].nunique()

        logging.info(f"\n{'Total Funding Raised':.<50} ${total_funding:>15,.0f}")
        logging.info(f"{'Total Funding Deals':.<50} {total_deals:>15,}")
        logging.info(f"{'Average Deal Size':.<50} ${avg_deal_size:>15,.0f}")
        logging.info(f"{'Average Valuation':.<50} ${avg_valuation:>15,.0f}")
        logging.info(f"{'Unique Startups Funded':.<50} {unique_startups:>15,}")
        logging.info(f"{'Unique Investors':.<50} {unique_investors:>15,}")
    else:
        logging.info("\n No funding data available for KPIs.")

    if not revenue_data.empty:
        logging.info(f"{'Average Monthly Revenue':.<50} ${revenue_data['monthly_revenue'].mean():>15,.0f}")
        logging.info(f"{'Average LTV/CAC Ratio':.<50} {revenue_data['ltv_cac_ratio'].mean():>15,.2f}")
        avg_runway = revenue_data['runway_months'].replace([np.inf, -np.inf], np.nan).mean()
        logging.info(f"{'Average Runway (months)':.<50} {avg_runway:>15,.1f}")
    else:
        logging.info(" No revenue data available for KPIs.")

    logging.info("\n" + "-"*70)
    logging.info("  LIVE MARKET DATA (Yahoo Finance API)")
    logging.info("-"*70)

    if not market_data.empty:
        for _, row in market_data.iterrows():
            logging.info(f"  {row['ticker']:>6} | {row['company_name']:<25} | ${row['current_price']:>8.2f} | {row['price_change_30d']:>+6.2f}%")
    else:
        logging.info("   No live market data available.")

    logging.info("\n" + "-"*70)
    logging.info("  TOP INDUSTRIES BY FUNDING")
    logging.info("-"*70)

    if not funding_data.empty:
        industry_funding = funding_data.groupby('industry').agg({
            'amount_raised': 'sum',
            'funding_id': 'count'
        }).round(0)
        industry_funding.columns = ['Total Funding', 'Deal Count']
        industry_funding = industry_funding.sort_values('Total Funding', ascending=False).head(5)
        logging.info(industry_funding)
    else:
        logging.info(" No funding data available to determine top industries.")

    logging.info("\n[3/7] Creating entrepreneurship analytics dashboard...")

    if not funding_data.empty or not market_data.empty or not revenue_data.empty:
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=(
                'Quarterly Funding Trends',
                'Funding by Industry',
                'Round Type Distribution',
                'Investor Type Analysis',
                'Valuation Multiple by Stage',
                'Geographic Distribution',
                'LIVE: Market Performance (API)',
                'Business Model Analysis'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "pie"}, {"type": "bar"}],
                [{"type": "box"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "pie"}]
            ]
        )

        if not funding_data.empty:
            quarterly = funding_data.groupby(['year', 'quarter'])['amount_raised'].sum().reset_index()
            quarterly['period'] = quarterly['year'].astype(str) + '-Q' + quarterly['quarter'].astype(str)
            fig.add_trace(
                go.Scatter(x=quarterly['period'], y=quarterly['amount_raised'],
                           mode='lines+markers', name='Quarterly Funding',
                           line=dict(color='#1f77b4', width=3)),
                row=1, col=1
            )
        else:
             fig.add_annotation(text="No funding data for quarterly trends", xref="x", yref="y", x=0.5, y=0.5, showarrow=False, row=1, col=1)

        if not funding_data.empty:
            industry = funding_data.groupby('industry')['amount_raised'].sum().sort_values(ascending=True).tail(8)
            fig.add_trace(
                go.Bar(x=industry.values, y=industry.index,
                       orientation='h', marker_color='lightblue', name='Industry'),
                row=1, col=2
            )
        else:
            fig.add_annotation(text="No funding data for industry analysis", xref="x", yref="y", x=0.5, y=0.5, showarrow=False, row=1, col=2)

        if not funding_data.empty:
            round_dist = funding_data['round_type'].value_counts()
            fig.add_trace(
                go.Pie(labels=round_dist.index, values=round_dist.values,
                       marker=dict(colors=px.colors.qualitative.Set3)),
                row=2, col=1
            )
        else:
            fig.add_annotation(text="No funding data for round type distribution", xref="x", yref="y", x=0.5, y=0.5, showarrow=False, row=2, col=1)

        if not funding_data.empty:
            investor_type = funding_data.groupby('investor_type')['amount_raised'].sum().reset_index()
            fig.add_trace(
                go.Bar(x=investor_type['investor_type'], y=investor_type['amount_raised'],
                       marker_color='lightcoral', name='Investor Type'),
                row=2, col=2
            )
        else:
             fig.add_annotation(text="No funding data for investor type analysis", xref="x", yref="y", x=0.5, y=0.5, showarrow=False, row=2, col=2)

        if not funding_data.empty and 'valuation_multiple' in funding_data.columns:
             fig.add_trace(
                go.Box(x=funding_data['stage'], y=funding_data['valuation_multiple'],
                       marker_color='lightgreen', name='Valuation Multiple'),
                row=3, col=1
            )
        else:
             fig.add_annotation(text="No funding data for valuation multiple analysis", xref="x", yref="y", x=0.5, y=0.5, showarrow=False, row=3, col=1)

        if not funding_data.empty:
            geo = funding_data['headquarters_country'].value_counts().head(5)
            fig.add_trace(
                go.Bar(x=geo.index, y=geo.values,
                       marker_color='lightsalmon', name='Geography'),
                row=3, col=2
            )
        else:
             fig.add_annotation(text="No funding data for geographic distribution", xref="x", yref="y", x=0.5, y=0.5, showarrow=False, row=3, col=2)

        if not market_data.empty:
            fig.add_trace(
                go.Bar(x=market_data['ticker'], y=market_data['price_change_30d'],
                       marker_color=market_data['price_change_30d'].apply(lambda x: 'green' if x > 0 else 'red'),
                       name='30D Change %', showlegend=False),
                row=4, col=1
            )
        else:
             fig.add_annotation(text="No live market data available", xref="x", yref="y", x=0.5, y=0.5, showarrow=False, row=4, col=1)

        if not funding_data.empty:
            biz_model = funding_data.groupby('business_model')['amount_raised'].sum()
            fig.add_trace(
                go.Pie(labels=biz_model.index, values=biz_model.values),
                row=4, col=2
            )
        else:
             fig.add_annotation(text="No funding data for business model analysis", xref="x", yref="y", x=0.5, y=0.5, showarrow=False, row=4, col=2)


        fig.update_layout(
            height=1600,
            showlegend=False,
            title_text="<b>Entrepreneurship Data Warehouse - Comprehensive Dashboard</b>",
            title_font_size=22
        )

        # Create analytics directory if it doesn't exist
        os.makedirs(ANALYTICS_DIR, exist_ok=True)
        
        output_path = os.path.join(ANALYTICS_DIR, 'entrepreneurship_dashboard.html')
        fig.write_html(output_path)
        fig.show() 

        logging.info(f" Dashboard created: {output_path}")
    else:
        logging.info(" No data available to create the dashboard.")

    logging.info("\n" + "="*70)
    logging.info("  ANALYTICS GENERATION COMPLETED")
    logging.info("="*70)


if __name__ == "__main__":
    generate_analytics_dashboard()