# Entrepreneurship Data Warehouse (DWH)

A comprehensive data warehouse solution for entrepreneurship and startup ecosystem analytics, featuring ETL pipelines, real-time API integration, machine learning models, and interactive dashboards.

## 📋 Project Overview

This project implements a complete **Data Warehouse (DWH)** system for analyzing entrepreneurship data, including startups, investors, funding rounds, revenue metrics, and market trends. The system integrates multiple data sources, performs ETL operations, and provides advanced analytics and machine learning capabilities.

## 🎯 Key Features

### 1. **Multi-Source Data Integration**
   - **OLTP System 1**: Startup management database (SQLite) with startups, funding rounds, and revenue metrics
   - **OLTP System 2**: Investor & VC database (SQLite) with investors, portfolio companies, and market trends
   - **CSV Sources**: Accelerator programs, economic indicators, and competitor data
   - **Real-Time API**: Live market data from Yahoo Finance API (stock prices, market caps, volatility)

### 2. **ETL Pipeline**
   - Extracts data from multiple heterogeneous sources
   - Transforms and cleans data (deduplication, normalization, date handling)
   - Loads data into a PostgreSQL data warehouse (Supabase) using a star schema design
   - Implements dimension and fact tables following data warehouse best practices

### 3. **Data Warehouse Schema**
   - **Dimension Tables**:
     - `dim_startup`: Startup information (company details, industry, stage, location)
     - `dim_investor`: Investor profiles (type, AUM, investment focus, preferences)
     - `dim_accelerator`: Accelerator program details
     - `dim_date`: Date dimension with fiscal calendar attributes
   
   - **Fact Tables**:
     - `fact_funding`: Funding round transactions with valuations
     - `fact_revenue`: Revenue metrics and KPIs (LTV/CAC, burn rate, runway)
     - `fact_market_data`: Real-time market data from Yahoo Finance API

### 4. **Analytics & Visualization**
   - Interactive dashboards using Plotly
   - Key Performance Indicators (KPIs) tracking
   - Industry and geographic analysis
   - Funding trends and investor type analysis
   - Real-time market performance visualization

### 5. **Machine Learning Models**
   - **Startup Success Prediction**: Random Forest classifier to predict startup success based on funding history, industry, and business model
   - **Funding Amount Prediction**: Ensemble model (Random Forest + Gradient Boosting) to predict funding amounts
   - **Investor Segmentation**: K-Means clustering to segment investors into categories (Smaller, Mid-Tier, Large, Mega-Fund)

### 6. **Data Updates & Maintenance**
   - Incremental source data updates (new startups, funding rounds, revenue metrics)
   - Scheduled market data updates from Yahoo Finance API
   - Automated data refresh capabilities

## 🛠️ Technology Stack

- **Languages**: Python 3.10+
- **Data Processing**: Pandas, NumPy
- **Database**: 
  - SQLite (source systems)
  - PostgreSQL via Supabase (data warehouse)
- **ETL**: SQLAlchemy, psycopg2
- **Data Generation**: Faker
- **API Integration**: yfinance, requests
- **Visualization**: Plotly
- **Machine Learning**: scikit-learn (Random Forest, Gradient Boosting, K-Means)
- **Environment Management**: python-dotenv

## 📁 Project Structure

```
enterpreneurship_dwh/
├── data/
│   ├── oltp1_startups/          # OLTP System 1 (SQLite)
│   │   └── startup_oltp.db
│   ├── oltp2_investors/         # OLTP System 2 (SQLite)
│   │   └── investor_oltp.db
│   ├── csv_source/              # CSV data sources
│   │   ├── accelerators.csv
│   │   ├── economic_indicators.csv
│   │   └── competitors.csv
│   ├── api_cache/               # Cached API data
│   │   ├── live_market_data.csv
│   │   └── market_metadata.json
│   └── warehouse/               # Data warehouse exports
├── analytics/                    # Generated analytics outputs
│   ├── entrepreneurship_dashboard.html
│   ├── success_prediction.html
│   ├── funding_prediction.html
│   └── investor_clusters.html
├── data_generation.py           # Initial data generation script
├── etl_pipeline.py              # Main ETL pipeline
├── analytics_dashboard.py       # Analytics and visualization
├── ml_models.py                 # Machine learning models
├── update_source_data.py        # Incremental source data updates
├── update_market_data.py        # Market data API updates
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- PostgreSQL database (Supabase recommended)
- Internet connection (for API data fetching)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd enterpreneurship_dwh
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv myenv
   myenv\Scripts\activate  # On Windows
   # or
   source myenv/bin/activate  # On Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root with your database credentials:
   ```env
   DB_HOST=your-supabase-host
   DB_NAME=your-database-name
   DB_USER=your-database-user
   DB_PASSWORD=your-database-password
   DB_PORT=5432
   DB_SSLMODE=require
   ```

### Usage

#### Step 1: Generate Initial Data
```bash
python data_generation.py
```
This script:
- Fetches live market data from Yahoo Finance API
- Generates synthetic startup data (150 startups)
- Generates investor data (75 investors)
- Creates CSV files with accelerator and economic data

#### Step 2: Run ETL Pipeline
```bash
python etl_pipeline.py
```
This script:
- Extracts data from all source systems
- Transforms and cleans the data
- Creates the data warehouse schema in PostgreSQL
- Loads data into dimension and fact tables

#### Step 3: Generate Analytics Dashboard
```bash
python analytics_dashboard.py
```
This script:
- Queries the data warehouse
- Calculates KPIs
- Generates interactive HTML dashboards
- Displays real-time market data

#### Step 4: Run Machine Learning Models
```bash
python ml_models.py
```
This script:
- Trains startup success prediction model
- Trains funding amount prediction model
- Performs investor segmentation
- Generates visualization outputs

#### Step 5: Update Data (Optional)
```bash
# Update source data incrementally
python update_source_data.py

# Update market data from API
python update_market_data.py
```

## 📊 Data Sources

### Source Systems

1. **OLTP1 - Startups Database**
   - 150+ startup companies
   - Funding rounds and valuations
   - Revenue metrics and KPIs
   - Employee counts and business models

2. **OLTP2 - Investors Database**
   - 75+ investors and VCs
   - Portfolio company investments
   - Market trend analysis by industry

3. **CSV Files**
   - Accelerator programs (Y Combinator, Techstars, etc.)
   - Economic indicators (GDP, unemployment, inflation)
   - Competitor analysis data

4. **Yahoo Finance API**
   - Real-time stock prices
   - Market capitalization
   - 30-day price changes
   - Trading volume

## 🔍 Analytics Capabilities

### Key Performance Indicators
- Total funding raised
- Average deal size
- Average valuation
- Unique startups funded
- Unique investors
- LTV/CAC ratios
- Runway calculations

### Visualizations
- Quarterly funding trends
- Funding by industry
- Round type distribution
- Investor type analysis
- Valuation multiple by stage
- Geographic distribution
- Real-time market performance
- Business model analysis

### Machine Learning Insights
- Startup success probability
- Predicted funding amounts
- Investor segmentation clusters
- Feature importance analysis

## 🎓 Skills Demonstrated

This project showcases expertise in:

- **Data Engineering**: ETL pipeline design and implementation
- **Database Design**: Star schema data warehouse architecture
- **Data Integration**: Multi-source data consolidation
- **API Integration**: Real-time data fetching and caching
- **Data Analysis**: Statistical analysis and KPI calculation
- **Machine Learning**: Classification, regression, and clustering
- **Data Visualization**: Interactive dashboard creation
- **Python Programming**: Object-oriented and functional programming
- **Database Management**: SQL, PostgreSQL, SQLite
- **Version Control**: Git and repository management

## 📈 Use Cases

This data warehouse can be used for:

- **Investment Analysis**: Identify trends in startup funding
- **Market Research**: Analyze industry performance and competition
- **Investor Relations**: Track investor behavior and preferences
- **Startup Evaluation**: Predict startup success and funding potential
- **Economic Analysis**: Correlate market conditions with startup activity
- **Strategic Planning**: Make data-driven decisions for entrepreneurship initiatives

## 🔧 Configuration

### Database Connection
The project uses Supabase (PostgreSQL) as the data warehouse. Configure connection settings in the `.env` file.

### API Configuration
Market data is fetched from Yahoo Finance API using the `yfinance` library. No API key required.

### Data Generation
Synthetic data generation uses seeded random numbers for reproducibility. Modify seeds in scripts to generate different datasets.

## 📝 Notes

- The project uses synthetic data for demonstration purposes
- Real-time API data requires an active internet connection
- Database credentials should be kept secure and not committed to version control
- The data warehouse schema follows dimensional modeling best practices
- ML models are trained on the generated dataset and may need retraining for production use

## 🤝 Contributing

This is an academic/portfolio project demonstrating data warehouse and analytics capabilities. For questions or suggestions, please open an issue or contact the repository maintainer.

## 📄 License

This project is for educational and demonstration purposes.

---

**Built with ❤️ for Data Warehouse and Analytics**

