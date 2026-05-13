import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="SpendSmart AI", page_icon="💰", layout="wide")
st.title("💰 SpendSmart AI")
st.markdown("**Clean & Working Version**")

uploaded_file = st.file_uploader("Upload your bank statement CSV", type=["csv"])

if uploaded_file is None:
    st.info("👆 Please upload sample_bank_statement (1).csv")
    st.stop()

# Robust CSV reading
try:
    df = pd.read_csv(uploaded_file, encoding='utf-8')
except:
    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

# Flexible date parsing
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
if df['Date'].isna().all():
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

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

# ====================== DASHBOARD ======================
st.metric("Total Spent", f"SGD {df['Amount'].sum():,.0f}")

col1, col2 = st.columns(2)
with col1:
    cat_df = df.groupby('Category')['Amount'].sum().reset_index()
    st.plotly_chart(px.pie(cat_df, values='Amount', names='Category', title="Spending Breakdown"), use_container_width=True)

with col2:
    df['Month'] = df['Date'].dt.to_period('M').astype(str)
    monthly = df.groupby('Month')['Amount'].sum().reset_index()
    st.plotly_chart(px.bar(monthly, x='Month', y='Amount', title="Monthly Spending Trend"), use_container_width=True)

st.success("✅ App is working!")
st.caption("SpendSmart AI • Simple Working Version by Anishka Moona")
