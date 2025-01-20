import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
# Load data
@st.cache_data
def load_data():
    # Update paths to the CSV files generated from the previous analysis script
    profitable = pd.read_csv('output_analysis/profitable_companies.csv')
    strong_financials = pd.read_csv('output_analysis/strong_financials.csv')
    efficient = pd.read_csv('output_analysis/efficient_companies.csv')
    manageable_debt = pd.read_csv('output_analysis/manageable_debt_companies.csv')
    growth = pd.read_csv('output_analysis/positive_working_capital.csv')
    low_admin = pd.read_csv('output_analysis/low_admin_expense_companies.csv')
    active = pd.read_csv('output_analysis/active_companies.csv')
    return {
        "Profitable Companies": profitable,
        "Strong Financials": strong_financials,
        "Efficient Companies": efficient,
        "Manageable Debt Companies": manageable_debt,
        "Positive Working Capital": growth,
        "Low Admin Expense Companies": low_admin,
        "Active Companies": active,
    }

data = load_data()

# Dashboard
st.title("Sales and Marketing Insights Dashboard")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = list(data.keys())
selected_analysis = st.sidebar.radio("Select Analysis", options)

# Display selected analysis
st.header(f"Analysis: {selected_analysis}")

# Show data and summary statistics
df = data[selected_analysis]

if st.checkbox("Show raw data"):
    st.dataframe(df)

st.markdown(f"### Summary of {selected_analysis}")
st.write(df.describe())

# Visualization options
if "turnover_gross_operating_revenue" in df.columns:
    st.markdown("#### Revenue Distribution")
    fig = px.histogram(df, x="turnover_gross_operating_revenue", title="Revenue Distribution")
    st.plotly_chart(fig)

if "average_number_employees_during_period" in df.columns:
    st.markdown("#### Revenue Per Employee")
    df['revenue_per_employee'] = df['turnover_gross_operating_revenue'] / df['average_number_employees_during_period']
    fig = px.box(df, y="revenue_per_employee", title="Revenue Per Employee Distribution")
    st.plotly_chart(fig)

if "net_current_assets_liabilities" in df.columns:
    st.markdown("#### Net Current Assets vs. Liabilities")
    fig = px.scatter(
        df,
        x="net_current_assets_liabilities",
        y="creditors_due_within_one_year",
        color="company_dormant",
        title="Net Current Assets vs. Creditors Due Within One Year",
        labels={"company_dormant": "Dormant Status"},
    )
    st.plotly_chart(fig)

if "profit_loss_for_period" in df.columns:
    st.markdown("#### Profit Distribution")
    fig = px.box(df, y="profit_loss_for_period", title="Profit Distribution")
    st.plotly_chart(fig)

# Analysis: Segmentation
st.markdown("### Segmentation")
if "turnover_gross_operating_revenue" in df.columns:
    st.markdown("#### Revenue Segmentation")
    bins = [0, 1e6, 5e6, 10e6, 50e6, 1e9]
    labels = ["<1M", "1M-5M", "5M-10M", "10M-50M", ">50M"]
    df['revenue_segment'] = pd.cut(df['turnover_gross_operating_revenue'], bins=bins, labels=labels)
    revenue_segment_counts = df['revenue_segment'].value_counts().sort_index()
    fig = px.bar(
        x=labels, 
        y=revenue_segment_counts,
        title="Revenue Segmentation",
        labels={"x": "Revenue Segment", "y": "Count"}
    )
    st.plotly_chart(fig)

# Analysis: Outlier Detection
st.markdown("### Outlier Detection")
if "turnover_gross_operating_revenue" in df.columns:
    st.markdown("#### Outliers in Revenue")
    fig = go.Figure()
    fig.add_trace(go.Box(y=df['turnover_gross_operating_revenue'], name='Revenue'))
    fig.update_layout(title="Outlier Detection for Revenue", yaxis_title="Revenue")
    st.plotly_chart(fig)

if "profit_loss_for_period" in df.columns:
    st.markdown("#### Outliers in Profit")
    fig = go.Figure()
    fig.add_trace(go.Box(y=df['profit_loss_for_period'], name='Profit'))
    fig.update_layout(title="Outlier Detection for Profit", yaxis_title="Profit")
    st.plotly_chart(fig)

# Highlight insights
st.markdown("### Key Insights")
if selected_analysis == "Profitable Companies":
    st.success("💡 These companies are profitable and could be prioritized for acquisition.")
elif selected_analysis == "Strong Financials":
    st.info("💡 Companies with strong financial health may have high acquisition potential.")
elif selected_analysis == "Efficient Companies":
    st.warning("💡 Highly efficient companies show strong revenue per employee metrics.")
elif selected_analysis == "Manageable Debt Companies":
    st.info("💡 These companies have manageable debt levels, indicating low financial risk.")
elif selected_analysis == "Positive Working Capital":
    st.success("💡 Positive working capital suggests strong liquidity and growth opportunities.")
elif selected_analysis == "Low Admin Expense Companies":
    st.warning("💡 Companies with low administrative expenses are operationally efficient.")
elif selected_analysis == "Active Companies":
    st.info("💡 Active companies (non-dormant) should be prioritized for business engagement.")

# Add filters to refine the results
st.markdown("### Filter Results")
columns_to_filter = st.multiselect("Select Columns to Filter", df.columns)
filters = {}
for col in columns_to_filter:
    unique_values = df[col].unique()
    selected_values = st.multiselect(f"Filter {col}", unique_values)
    if selected_values:
        filters[col] = selected_values

if filters:
    filtered_data = df
    for col, values in filters.items():
        filtered_data = filtered_data[filtered_data[col].isin(values)]
    st.markdown("#### Filtered Data")
    st.dataframe(filtered_data)

# Footer
st.markdown("Dashboard created by **Nur Nesa Nashuha** for Sales and Marketing Team.")
