import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings('ignore')
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# PostgreSQL connection details (Session Pooler for IPv4 compatibility)
DB_HOST = os.getenv('DB_HOST')  # aws-1-ap-southeast-1.pooler.supabase.com
DB_NAME = os.getenv('DB_NAME')  # postgres
DB_USER = os.getenv('DB_USER')  # postgres.bfbsqzaygxvdyfvzfmgd
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_SSLMODE = os.getenv('DB_SSLMODE', 'require')

# Define base paths
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
        logging.error("="*70)
        return None


def run_ml_models():
    # Create analytics directory if it doesn't exist
    os.makedirs(ANALYTICS_DIR, exist_ok=True)
    
    print("\n" + "="*70)
    print("  ENTREPRENEURSHIP DATA WAREHOUSE - ML MODELS")
    print("="*70)

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

        engine.dispose()
        
    except Exception as e:
        logging.error(f" Error loading data from data warehouse: {e}")
        if engine:
            engine.dispose()
        return


    logging.info("\n[4/7] Running ML Model: Startup Success Prediction...")

    if not funding_data.empty:
        startup_features = funding_data.groupby('startup_id').agg({
            'amount_raised': 'sum',
            'funding_id': 'count',
            'valuation': 'max',
            'number_of_investors': 'sum',
            'industry': 'first',
            'stage': 'first',
            'business_model': 'first'
        }).reset_index()

        startup_features.columns = [
            'startup_id', 'total_raised', 'funding_rounds', 'max_valuation',
            'total_investors', 'industry', 'stage', 'business_model'
        ]

        success_stages = ['Series B', 'Series C', 'Growth']
        startup_features['is_successful'] = (
            (startup_features['stage'].isin(success_stages)) &
            (startup_features['total_raised'] > 5000000)
        ).astype(int)

        le_industry = LabelEncoder()
        le_model = LabelEncoder()

        startup_features['industry_encoded'] = le_industry.fit_transform(startup_features['industry'])
        startup_features['model_encoded'] = le_model.fit_transform(startup_features['business_model'])

        X = startup_features[[
            'total_raised', 'funding_rounds', 'max_valuation',
            'total_investors', 'industry_encoded', 'model_encoded'
        ]]
        y = startup_features['is_successful']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        clf_success = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        clf_success.fit(X_train, y_train)

        train_score = clf_success.score(X_train, y_train)
        test_score = clf_success.score(X_test, y_test)

        print(f"\n Startup Success Prediction Model:")
        print(f"  Training Accuracy: {train_score:.2%}")
        print(f"  Testing Accuracy: {test_score:.2%}")
        print(f"  Successful Startups: {y.sum()} / {len(y)} ({y.mean():.1%})")

        feature_importance = pd.DataFrame({
            'Feature': ['Total Raised', 'Funding Rounds', 'Max Valuation',
                        'Total Investors', 'Industry', 'Business Model'],
            'Importance': clf_success.feature_importances_
        }).sort_values('Importance', ascending=False)

        print("\n  Feature Importance:")
        print(feature_importance.to_string(index=False))

        fig_success = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Startup Success Prediction - Feature Importance',
            color='Importance',
            color_continuous_scale='Viridis'
        )

        success_path = os.path.join(ANALYTICS_DIR, 'success_prediction.html')
        fig_success.write_html(success_path)
        fig_success.show() 

        print(f" Success prediction model completed: {success_path}")
    else:
        print(" No funding data available for Startup Success Prediction.")
        train_score = None
        test_score = None


    print("\n[5/7] Running ML Model: Funding Amount Prediction...")

    if not funding_data.empty:
        reg_features = funding_data[funding_data['amount_raised'] > 0].copy()

        if not reg_features.empty:
            le_industry_reg = LabelEncoder()
            le_round_reg = LabelEncoder()
            le_investor_type_reg = LabelEncoder()

            reg_features['industry_encoded'] = le_industry_reg.fit_transform(reg_features['industry'])
            reg_features['round_encoded'] = le_round_reg.fit_transform(reg_features['round_type'])
            reg_features['investor_type_encoded'] = le_investor_type_reg.fit_transform(reg_features['investor_type'])

            X_reg = reg_features[[
                'employee_count', 'number_of_investors',
                'industry_encoded', 'round_encoded', 'investor_type_encoded'
            ]].fillna(0)

            y_reg = reg_features['amount_raised']

            if len(X_reg) > 1:
                X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
                    X_reg, y_reg, test_size=0.3, random_state=42
                )

                reg_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
                reg_model.fit(X_train_reg, y_train_reg)

                y_pred = reg_model.predict(X_test_reg)

                r2 = r2_score(y_test_reg, y_pred)
                mae = mean_absolute_error(y_test_reg, y_pred)

                print(f"\n Funding Amount Prediction Model:")
                print(f"  R² Score: {r2:.3f}")
                print(f"  Mean Absolute Error: ${mae:,.0f}")
                print(f"  Average Actual Funding: ${y_test_reg.mean():,.0f}")
                print(f"  Average Predicted Funding: ${y_pred.mean():,.0f}")

                comparison_df = pd.DataFrame({
                    'Actual': y_test_reg.values,
                    'Predicted': y_pred
                })

                fig_reg = px.scatter(
                    comparison_df,
                    x='Actual',
                    y='Predicted',
                    title='Funding Amount: Actual vs Predicted',
                    labels={'Actual': 'Actual Funding ($)', 'Predicted': 'Predicted Funding ($)'},
                    trendline='ols'
                )

                fig_reg.add_trace(
                    go.Scatter(x=[0, comparison_df['Actual'].max()],
                               y=[0, comparison_df['Actual'].max()],
                               mode='lines', name='Perfect Prediction',
                               line=dict(dash='dash', color='red'))
                )

                funding_pred_path = os.path.join(ANALYTICS_DIR, 'funding_prediction.html')
                fig_reg.write_html(funding_pred_path)
                fig_reg.show() 

                print(f" Funding prediction model completed: {funding_pred_path}")
            else:
                print(" Not enough data after filtering for funding amount prediction.")
                r2 = None
                mae = None
        else:
            print(" No data with amount_raised > 0 for Funding Amount Prediction.")
            r2 = None
            mae = None
    else:
        print(" No funding data available for Funding Amount Prediction.")
        r2 = None
        mae = None


    print("\n[6/7] Running K-Means: Investor Segmentation...")

    if not funding_data.empty:
        investor_metrics = funding_data.groupby('investor_name').agg({
            'funding_id': 'count',
            'amount_raised': 'sum',
            'valuation': 'mean',
            'investor_type': 'first',
            'investment_focus': 'first'
        }).reset_index()

        investor_metrics.columns = [
            'investor_name', 'deal_count', 'total_invested',
            'avg_valuation', 'investor_type', 'investment_focus'
        ]

        investor_metrics['avg_investment'] = (
            investor_metrics['total_invested'] / investor_metrics['deal_count'].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)

        features_investor = investor_metrics[[
            'deal_count', 'total_invested', 'avg_investment', 'avg_valuation'
        ]].fillna(0).values

        if len(features_investor) > 4:
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_investor)

            kmeans_inv = KMeans(n_clusters=4, random_state=42, n_init=10)
            investor_metrics['cluster'] = kmeans_inv.fit_predict(features_scaled)

            cluster_summary_temp = investor_metrics.groupby('cluster')['total_invested'].mean().sort_values()
            ordered_clusters = cluster_summary_temp.index.tolist()

            cluster_names_map = {
                ordered_clusters[0]: 'Smaller Investors',
                ordered_clusters[1]: 'Mid-Tier Investors',
                ordered_clusters[2]: 'Large Investors',
                ordered_clusters[3]: 'Mega-Fund Investors'
            }
            investor_metrics['cluster_name'] = investor_metrics['cluster'].map(cluster_names_map)


            print("\n Investor Cluster Analysis:")
            cluster_summary = investor_metrics.groupby('cluster_name').agg({
                'deal_count': 'mean',
                'total_invested': 'mean',
                'avg_investment': 'mean',
                'avg_valuation': 'mean'
            }).round(0)

            print(cluster_summary)

            fig_investor_cluster = px.scatter_3d(
                investor_metrics,
                x='deal_count',
                y='total_invested',
                z='avg_valuation',
                color='cluster_name',
                title='Investor Segmentation (K-Means Clustering)',
                labels={'cluster_name': 'Investor Segment'},
                hover_data=['investor_name'],
                color_discrete_sequence=px.colors.qualitative.T10
            )

            investor_cluster_path = os.path.join(ANALYTICS_DIR, 'investor_clusters.html')
            fig_investor_cluster.write_html(investor_cluster_path)
            fig_investor_cluster.show() 

            print(f" Investor segmentation completed: {investor_cluster_path}")
            num_investor_segments = investor_metrics['cluster'].nunique()
        else:
            print(" Not enough investor data for clustering.")
            num_investor_segments = 0

    else:
        print(" No funding data available for Investor Clustering.")
        num_investor_segments = 0


    print("\n" + "="*80)
    print("  ANALYTICS & ML PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\n ANALYTICS OUTPUTS:")
    print(f"   Comprehensive Dashboard: {os.path.join(ANALYTICS_DIR, 'entrepreneurship_dashboard.html')}")
    print(f"   Success Prediction Model: {os.path.join(ANALYTICS_DIR, 'success_prediction.html')} (Accuracy: {test_score:.2%})" if test_score is not None else "   Success Prediction Model: Not enough data.")
    print(f"   Funding Prediction Model: {os.path.join(ANALYTICS_DIR, 'funding_prediction.html')} (R²: {r2:.3f})" if r2 is not None else "   Funding Prediction Model: Not enough data.")
    print(f"   Investor Clustering: {os.path.join(ANALYTICS_DIR, 'investor_clusters.html')} ({num_investor_segments} segments)" if num_investor_segments > 0 else "   Investor Clustering: Not enough data.")
    print("\n FILES CREATED:")
    print(f"  All visualization HTML files are in {ANALYTICS_DIR}/")
    print("\n" + "="*80)
    print("  ANALYTICS & ML DONE!")
    print("="*80)


if __name__ == "__main__":
    run_ml_models()