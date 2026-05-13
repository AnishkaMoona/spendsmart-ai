import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import sqlite3
from groq import Groq
import os
from dotenv import load_dotenv
import pdfplumber
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

load_dotenv()

st.set_page_config(page_title="SpendSmart AI", page_icon="💰", layout="wide")
st.title("💰 SpendSmart AI")
st.markdown("**Fixed Date Parsing • Better Visuals • ML Anomaly**")

# Database
conn = sqlite3.connect('spendsmart.db', check_same_thread=False)
conn.execute('''CREATE TABLE IF NOT EXISTS transactions 
             (id INTEGER PRIMARY KEY, date TEXT, description TEXT, amount REAL, category TEXT, source TEXT)''')

def save_to_db(df, source):
    for _, row in df.iterrows():
        conn.execute("INSERT OR IGNORE INTO transactions (date, description, amount, category, source) VALUES (?, ?, ?, ?, ?)",
                    (row['Date'].strftime('%Y-%m-%d'), row['Description'], float(row['Amount']), row.get('Category', 'Others'), source))
    conn.commit()

# ====================== UPLOAD ======================
st.sidebar.header("Upload Bank Statement")
uploaded_file = st.sidebar.file_uploader("CSV or PDF", type=["csv", "pdf"])

df = pd.DataFrame()

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.pdf'):
            with pdfplumber.open(uploaded_file) as pdf:
                text = "\n".join([page.extract_text() or "" for page in pdf.pages])
            st.text_area("PDF Text", text[:1500], height=200)
        else:
            df = pd.read_csv(uploaded_file)
            
            # Robust date parsing
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
            if df['Date'].isna().all():
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            df = df.dropna(subset=['Date'])
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
            df = df[df['Amount'] != 0].copy()
            
            def categorize(desc):
                desc = str(desc).lower()
                if any(k in desc for k in ['grab','taxi','mrt','bus']): return 'Transport'
                if any(k in desc for k in ['food','restaurant','hawker','starbucks','coffee']): return 'Food & Dining'
                if any(k in desc for k in ['shopee','lazada','amazon','shopping']): return 'Shopping'
                if any(k in desc for k in ['netflix','spotify']): return 'Subscriptions'
                if any(k in desc for k in ['salary','credit']): return 'Income'
                return 'Others'
            
            df['Category'] = df['Description'].apply(categorize)
            save_to_db(df, uploaded_file.name)
            st.success(f"✅ Loaded {len(df)} transactions!")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.stop()

# Load history
df_all = pd.read_sql("SELECT * FROM transactions", conn)
if not df_all.empty:
    df_all['Date'] = pd.to_datetime(df_all['Date'], errors='coerce')
    if df.empty:
        df = df_all
    else:
        df = pd.concat([df, df_all], ignore_index=True).drop_duplicates(subset=['date', 'description', 'amount'])

if df.empty:
    st.info("Upload your sample CSV to see improved charts")
    st.stop()

# ====================== DASHBOARD ======================
st.metric("Total Spent", f"SGD {df['Amount'].sum():,.0f}")

col1, col2 = st.columns(2)

with col1:
    cat_df = df.groupby('Category')['Amount'].sum().reset_index()
    fig_pie = px.pie(cat_df, values='Amount', names='Category', title="Spending by Category")
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    df['Month'] = df['Date'].dt.to_period('M').astype(str)
    monthly = df.groupby('Month')['Amount'].sum().reset_index()
    fig_bar = px.bar(monthly, x='Month', y='Amount', title="Monthly Spending Trend")
    st.plotly_chart(fig_bar, use_container_width=True)

# ML Anomaly (optional)
if len(df) > 5:
    st.subheader("🔍 ML Detected Unusual Transactions")
    features = pd.get_dummies(df[['Category']])
    features['Amount'] = df['Amount']
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    iso = IsolationForest(contamination=0.15, random_state=42)
    df['Anomaly'] = iso.fit_predict(features_scaled)
    df['Anomaly'] = df['Anomaly'].map({1: 'Normal', -1: 'Unusual ⚠️'})
    anomalies = df[df['Anomaly'] == 'Unusual ⚠️']
    if not anomalies.empty:
        st.dataframe(anomalies[['Date','Description','Amount','Category']])

st.caption("SpendSmart AI • Anishka Moona")
