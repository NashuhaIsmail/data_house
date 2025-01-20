import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = 'C:\\Users\\nesa.nashuha\\OneDrive - Habib Jewels Sdn Bhd\\Sr DSC\\extract from xbrl\\extracted.csv'  
df = pd.read_csv(file_path)

# Helper function to calculate financial metrics
def calculate_revenue_per_employee(row):
    if row['average_number_employees_during_period'] > 0:
        return row['turnover_gross_operating_revenue'] / row['average_number_employees_during_period']
    else:
        return None

# 1. Financial Health Analysis
def financial_health_analysis(df):
    # Filter profitable companies
    profitable_companies = df[df['profit_loss_for_period'] > 0]
    # High net assets companies
    strong_financials = df[df['net_assets_liabilities_including_pension_asset_liability'] > 0]
    return profitable_companies, strong_financials

# 2. Workforce and Operational Efficiency
def workforce_efficiency(df):
    df['revenue_per_employee'] = df.apply(calculate_revenue_per_employee, axis=1)
    efficient_companies = df[df['revenue_per_employee'] > 100000]  # Customize threshold
    return efficient_companies

# 3. Asset Utilization and Debt Assessment
def asset_and_debt_analysis(df):
    manageable_debt = df[
        (df['creditors_due_within_one_year'] < df['current_assets']) & 
        (df['creditors_due_after_one_year'] < df['net_assets_liabilities_including_pension_asset_liability'])
    ]
    return manageable_debt

# 4. Growth Opportunities
def growth_opportunities(df):
    positive_working_capital = df[df['net_current_assets_liabilities'] > 0]
    low_admin_expense = df[df['administrative_expenses'] < 0.2 * df['turnover_gross_operating_revenue']]  # <20% admin cost
    return positive_working_capital, low_admin_expense

# 5. Market Insights
def market_insights(df):
    active_companies = df[df['company_dormant'] == 'false']  # Assuming `company_dormant` is a boolean or 'false/true'
    return active_companies

# 6. Trend Analysis
#def trend_analysis(df):
    # Ensure date fields are datetime
    df['balance_sheet_date'] = pd.to_datetime(df['balance_sheet_date'], errors='coerce')
    trend_df = df.groupby(df['balance_sheet_date'].dt.to_period('M')).mean()  # Monthly trends
    trend_df.plot(figsize=(10, 6), title="Trends Over Time in Financial Metrics")
    plt.ylabel("Financial Metrics (Averaged)")
    plt.show()
    return trend_df

# 7. Segmentation
def financial_segmentation(df):
    # Select relevant financial metrics
    clustering_data = df[['shareholder_funds', 'net_assets_liabilities_including_pension_asset_liability']].dropna()
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clustering_data['Cluster'] = kmeans.fit_predict(clustering_data)
    sns.scatterplot(
        data=clustering_data, 
        x='shareholder_funds', 
        y='net_assets_liabilities_including_pension_asset_liability', 
        hue='Cluster', 
        palette='viridis'
    )
    plt.title("Company Segmentation Based on Financial Metrics")
    plt.show()
    return clustering_data

# 8. Outlier Detection
def detect_outliers(df):
    outliers = df[(df['profit_loss_for_period'] < -1000000) |  # Customize thresholds
                  (df['profit_loss_for_period'] > 10000000)]
    return outliers

# Run all analyses
profitable, strong_financials = financial_health_analysis(df)
efficient_companies = workforce_efficiency(df)
manageable_debt = asset_and_debt_analysis(df)
growth, low_admin_cost = growth_opportunities(df)
active_companies = market_insights(df)
#trend_df = trend_analysis(df)
segmentation_results = financial_segmentation(df)
outliers = detect_outliers(df)

# Save results
output_folder = 'C:\\Users\\nesa.nashuha\\OneDrive - Habib Jewels Sdn Bhd\\Sr DSC\\extract from xbrl\\output_analysis\\'
profitable.to_csv(output_folder + 'profitable_companies.csv', index=False)
strong_financials.to_csv(output_folder + 'strong_financials.csv', index=False)
efficient_companies.to_csv(output_folder + 'efficient_companies.csv', index=False)
manageable_debt.to_csv(output_folder + 'manageable_debt_companies.csv', index=False)
growth.to_csv(output_folder + 'positive_working_capital.csv', index=False)
low_admin_cost.to_csv(output_folder + 'low_admin_expense_companies.csv', index=False)
active_companies.to_csv(output_folder + 'active_companies.csv', index=False)
segmentation_results.to_csv(output_folder + 'segmentation_results.csv', index=False)
outliers.to_csv(output_folder + 'outliers.csv', index=False)

print("Analysis completed. Results saved in:", output_folder)
