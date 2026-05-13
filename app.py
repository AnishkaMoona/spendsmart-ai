import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="SpendSmart AI", page_icon="💰", layout="wide")
st.title("💰 SpendSmart AI")
st.markdown("**Clean Charts Version**")

uploaded_file = st.file_uploader("Upload your bank statement CSV", type=["csv"])

if uploaded_file is None:
    st.info("Upload sample_bank_statement (1).csv")
    st.stop()

# Load CSV
df = pd.read_csv(uploaded_file)

# Fix date parsing
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
df = df.dropna(subset=['Date']).copy()
df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
df = df[df['Amount'] != 0].copy()

# Categorization
def categorize(desc):
    desc = str(desc).lower()
    if any(k in desc for k in ['grab','taxi','mrt','bus']): return 'Transport'
    if any(k in desc for k in ['food','restaurant','hawker','starbucks','coffee']): return 'Food & Dining'
    if any(k in desc for k in ['shopee','lazada','amazon','shopping']): return 'Shopping'
    if any(k in desc for k in ['netflix','spotify']): return 'Subscriptions'
    if any(k in desc for k in ['salary','credit']): return 'Income'
    return 'Others'

df['Category'] = df['Description'].apply(categorize)

# Clean Monthly Grouping
df['Month'] = df['Date'].dt.to_period('M').dt.strftime('%b %Y')

# ====================== DASHBOARD ======================
st.metric("Total Spent", f"SGD {df['Amount'].sum():,.0f}")

col1, col2 = st.columns(2)

with col1:
    cat_df = df.groupby('Category')['Amount'].sum().reset_index()
    fig_pie = px.pie(cat_df, values='Amount', names='Category', title="Spending Breakdown")
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    monthly = df.groupby('Month')['Amount'].sum().reset_index()
    fig_bar = px.bar(monthly, x='Month', y='Amount', title="Monthly Spending Trend")
    fig_bar.update_layout(xaxis_title="Month", yaxis_title="Amount (SGD)")
    st.plotly_chart(fig_bar, use_container_width=True)

st.success("✅ Charts should look clean now!")
st.caption("SpendSmart AI • Anishka Moona")
