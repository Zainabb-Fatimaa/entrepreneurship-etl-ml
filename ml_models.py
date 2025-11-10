import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings('ignore')
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DB_HOST = os.getenv('DB_HOST')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_PORT = os.getenv('DB_PORT', '6543')
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
        logging.error("1. Verify your .env file has the correct Session Pooler settings")
        logging.error("2. Ensure your Supabase project is not paused (free tier)")
        logging.error("="*70)
        return None


def run_ml_models():
    os.makedirs(ANALYTICS_DIR, exist_ok=True)
    
    print("\n" + "="*70)
    print("  ENTREPRENEURSHIP DATA WAREHOUSE - ML MODELS")
    print("="*70)

    logging.info("\n[1/7] Importing libraries...")
    logging.info("Libraries imported!")

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
        
        feature_importance['Percentage'] = (feature_importance['Importance'] * 100).round(1)

        print("\n  Feature Importance:")
        for _, row in feature_importance.iterrows():
            print(f"    {row['Feature']:<20} {row['Importance']:.4f} ({row['Percentage']:.1f}%)")

        fig_success = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Startup Success Prediction - Feature Importance',
            color='Importance',
            color_continuous_scale='Viridis',
            text='Percentage'
        )
        
        fig_success.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside'
        )
        
        fig_success.update_layout(
            height=500,
            width=900,
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            showlegend=False
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

        if not reg_features.empty and len(reg_features) > 10:
            le_industry_reg = LabelEncoder()
            le_round_reg = LabelEncoder()
            le_investor_type_reg = LabelEncoder()
            le_stage_reg = LabelEncoder()

            reg_features['industry_encoded'] = le_industry_reg.fit_transform(reg_features['industry'])
            reg_features['round_encoded'] = le_round_reg.fit_transform(reg_features['round_type'])
            reg_features['investor_type_encoded'] = le_investor_type_reg.fit_transform(reg_features['investor_type'])
            reg_features['stage_encoded'] = le_stage_reg.fit_transform(reg_features['stage'])
            
            reg_features['valuation_filled'] = reg_features['valuation'].fillna(reg_features['valuation'].median())
            
            if 'valuation_multiple' in reg_features.columns:
                reg_features['valuation_multiple_filled'] = reg_features['valuation_multiple'].fillna(
                    reg_features['valuation_multiple'].median()
                )
            else:
                reg_features['valuation_multiple_filled'] = 0
            
            reg_features['investors_x_employees'] = reg_features['number_of_investors'] * reg_features['employee_count']
            reg_features['valuation_per_investor'] = reg_features['valuation_filled'] / (reg_features['number_of_investors'] + 1)
            reg_features['valuation_per_employee'] = reg_features['valuation_filled'] / (reg_features['employee_count'] + 1)
            
            reg_features['year_normalized'] = (reg_features['year'] - reg_features['year'].min()) / (reg_features['year'].max() - reg_features['year'].min() + 1)
            reg_features['quarter_encoded'] = reg_features['quarter']
            
            reg_features['valuation_squared'] = reg_features['valuation_filled'] ** 2
            reg_features['valuation_log'] = np.log1p(reg_features['valuation_filled'])
            
            feature_cols = [
                'employee_count', 'number_of_investors',
                'valuation_filled', 'valuation_squared', 'valuation_log',
                'valuation_multiple_filled',
                'investors_x_employees', 'valuation_per_investor', 'valuation_per_employee',
                'industry_encoded', 'round_encoded', 'investor_type_encoded',
                'stage_encoded', 'year_normalized', 'quarter_encoded'
            ]
            
            X_reg = reg_features[feature_cols].fillna(0)
            y_reg = reg_features['amount_raised']
            
            y_reg_log = np.log1p(y_reg)

            if len(X_reg) > 20:
                X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
                    X_reg, y_reg_log, test_size=0.3, random_state=42
                )
                
                scaler_reg = StandardScaler()
                X_train_scaled = scaler_reg.fit_transform(X_train_reg)
                X_test_scaled = scaler_reg.transform(X_test_reg)

                rf_model = RandomForestRegressor(
                    n_estimators=500, 
                    random_state=42, 
                    max_depth=25,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features='sqrt',
                    n_jobs=-1
                )
                
                gb_model = GradientBoostingRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=8,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    subsample=0.8,
                    random_state=42
                )
                
                print("  Training Random Forest model...")
                rf_model.fit(X_train_scaled, y_train_reg)
                print("  Training Gradient Boosting model...")
                gb_model.fit(X_train_scaled, y_train_reg)
                
                y_pred_rf_log = rf_model.predict(X_test_scaled)
                y_pred_gb_log = gb_model.predict(X_test_scaled)
                y_pred_log = 0.5 * y_pred_rf_log + 0.5 * y_pred_gb_log
                
                y_pred = np.expm1(y_pred_log)
                y_test_actual = np.expm1(y_test_reg)

                r2 = r2_score(y_test_actual, y_pred)
                mae = mean_absolute_error(y_test_actual, y_pred)
                mape = np.mean(np.abs((y_test_actual - y_pred) / (y_test_actual + 1))) * 100
                
                y_train_pred_rf_log = rf_model.predict(X_train_scaled)
                y_train_pred_gb_log = gb_model.predict(X_train_scaled)
                y_train_pred_log = 0.5 * y_train_pred_rf_log + 0.5 * y_train_pred_gb_log
                y_train_pred = np.expm1(y_train_pred_log)
                y_train_actual = np.expm1(y_train_reg)
                r2_train = r2_score(y_train_actual, y_train_pred)

                print(f"\n Funding Amount Prediction Model (Ensemble: RF + GradientBoosting):")
                print(f"  R² Score (Train): {r2_train:.3f}")
                print(f"  R² Score (Test): {r2:.3f}")
                print(f"  Mean Absolute Error: ${mae:,.0f}")
                print(f"  Mean Absolute Percentage Error: {mape:.2f}%")
                print(f"  Average Actual Funding: ${y_test_actual.mean():,.0f}")
                print(f"  Average Predicted Funding: ${y_pred.mean():,.0f}")
                
                feature_imp = pd.DataFrame({
                    'Feature': feature_cols,
                    'RF_Importance': rf_model.feature_importances_,
                    'GB_Importance': gb_model.feature_importances_
                })
                feature_imp['Avg_Importance'] = (feature_imp['RF_Importance'] + feature_imp['GB_Importance']) / 2
                feature_imp = feature_imp.sort_values('Avg_Importance', ascending=False)
                
                print("\n  Top 8 Features:")
                for _, row in feature_imp.head(8).iterrows():
                    print(f"    {row['Feature']:<30} {row['Avg_Importance']:.4f}")

                comparison_df = pd.DataFrame({
                    'Actual': y_test_actual,
                    'Predicted': y_pred
                })

                fig_reg = px.scatter(
                    comparison_df,
                    x='Actual',
                    y='Predicted',
                    title=f'Funding Amount: Actual vs Predicted (Ensemble Model)<br>R² = {r2:.3f}, MAE = ${mae:,.0f}',
                    labels={'Actual': 'Actual Funding ($)', 'Predicted': 'Predicted Funding ($)'},
                    opacity=0.6,
                    trendline='ols'
                )

                max_val = max(comparison_df['Actual'].max(), comparison_df['Predicted'].max())
                fig_reg.add_trace(
                    go.Scatter(
                        x=[0, max_val],
                        y=[0, max_val],
                        mode='lines', 
                        name='Perfect Prediction',
                        line=dict(dash='dash', color='red', width=2)
                    )
                )
                
                fig_reg.update_layout(
                    showlegend=True,
                    height=600,
                    width=900
                )

                funding_pred_path = os.path.join(ANALYTICS_DIR, 'funding_prediction.html')
                fig_reg.write_html(funding_pred_path)
                fig_reg.show() 

                print(f" Funding prediction model completed: {funding_pred_path}")
            else:
                print(" Not enough data points (need at least 20).")
                r2 = None
                mae = None
        else:
            print(" Not enough data with amount_raised > 0.")
            r2 = None
            mae = None
    else:
        print(" No funding data available.")
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
    print(f"   Success Prediction: {os.path.join(ANALYTICS_DIR, 'success_prediction.html')} (Accuracy: {test_score:.2%})" if test_score is not None else "   Success Prediction: Not enough data.")
    print(f"   Funding Prediction: {os.path.join(ANALYTICS_DIR, 'funding_prediction.html')} (R²: {r2:.3f})" if r2 is not None else "   Funding Prediction: Not enough data.")
    print(f"   Investor Clustering: {os.path.join(ANALYTICS_DIR, 'investor_clusters.html')} ({num_investor_segments} segments)" if num_investor_segments > 0 else "   Investor Clustering: Not enough data.")
    print("\n FILES CREATED:")
    print(f"  All HTML files in {ANALYTICS_DIR}/")
    print("\n" + "="*80)

if __name__ == "__main__":
    run_ml_models()