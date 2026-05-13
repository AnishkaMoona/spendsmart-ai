import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="SpendSmart AI", page_icon="💰", layout="wide")
st.title("💰 SpendSmart AI")
st.markdown("**Clean Charts • Income vs Expenses**")

uploaded_file = st.file_uploader("Upload your bank statement CSV", type=["csv"])

if uploaded_file is None:
    st.info("Upload sample_bank_statement.csv")
    st.stop()

# Load and clean data
df = pd.read_csv(uploaded_file)
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

# Separate Income and Expenses
df['Type'] = df['Amount'].apply(lambda x: 'Income' if x > 0 else 'Expense')
df['Expense_Amount'] = df['Amount'].abs()  # For expense visualization

# ====================== DASHBOARD ======================
total_income = df[df['Type'] == 'Income']['Amount'].sum()
total_expense = df[df['Type'] == 'Expense']['Amount'].abs().sum()

st.metric("Total Income", f"SGD {total_income:,.0f}", delta=f"-SGD {total_expense:,.0f} spent")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Spending Breakdown (Expenses Only)")
    expense_df = df[df['Type'] == 'Expense'].groupby('Category')['Expense_Amount'].sum().reset_index()
    if not expense_df.empty:
        fig_pie = px.pie(expense_df, values='Expense_Amount', names='Category', title="Where Your Money Went")
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.write("No expenses found")

with col2:
    st.subheader("Monthly Trend")
    df['Month'] = df['Date'].dt.to_period('M').astype(str)
    monthly = df.groupby(['Month', 'Type'])['Amount'].sum().unstack(fill_value=0)
    monthly['Net'] = monthly.get('Income', 0) - monthly.get('Expense', 0).abs()
    fig_bar = px.bar(monthly.reset_index(), x='Month', y=['Income', 'Expense'], title="Income vs Expense Trend", barmode='group')
    st.plotly_chart(fig_bar, use_container_width=True)

st.caption("SpendSmart AI • Clean Version by Anishka Moona")
