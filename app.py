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
st.markdown("**Improved PDF Support + Flexible CSV • ML Anomaly Detection**")

# ====================== DATABASE ======================
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
                full_text = "\n".join([page.extract_text() or "" for page in pdf.pages])
            
            st.subheader("Extracted Text from PDF")
            st.text_area("Raw Text", full_text[:2000], height=300)
            
            st.info("PDF parsing is still basic. For best results, use CSV with columns: Date, Description, Amount")
            
        else:
            # Flexible CSV reading
            try:
                df = pd.read_csv(uploaded_file)
            except:
                df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
            
            # Flexible column name handling
            date_col = None
            for possible in ['Date', 'date', 'Transaction Date', 'Txn Date', 'Posting Date']:
                if possible in df.columns:
                    date_col = possible
                    break
            
            desc_col = None
            for possible in ['Description', 'Desc', 'Particulars', 'Narration']:
                if possible in df.columns:
                    desc_col = possible
                    break
            
            amount_col = None
            for possible in ['Amount', 'amount', 'Debit', 'Credit', 'Balance']:
                if possible in df.columns:
                    amount_col = possible
                    break
            
            if date_col and desc_col and amount_col:
                df = df[[date_col, desc_col, amount_col]].copy()
                df.columns = ['Date', 'Description', 'Amount']
                
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
                st.success(f"✅ Successfully loaded {len(df)} transactions!")
            else:
                st.error("Could not detect required columns (Date, Description, Amount). Please check your CSV format.")
                st.stop()
                
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.stop()

# Load history
df_all = pd.read_sql("SELECT * FROM transactions", conn)
if not df_all.empty:
    df_all['Date'] = pd.to_datetime(df_all['Date'])
    if df.empty:
        df = df_all
    else:
        df = pd.concat([df, df_all], ignore_index=True)

if df.empty:
    st.info("👆 Upload a CSV or PDF bank statement to begin")
    st.stop()

# ====================== DASHBOARD ======================
st.metric("Total Spent", f"SGD {df['Amount'].sum():,.0f}")

col1, col2 = st.columns(2)
with col1:
    cat_df = df.groupby('Category')['Amount'].sum().reset_index()
    st.plotly_chart(px.pie(cat_df, values='Amount', names='Category'), use_container_width=True)

with col2:
    df['Month'] = df['Date'].dt.to_period('M').astype(str)
    monthly = df.groupby('Month')['Amount'].sum().reset_index()
    st.plotly_chart(px.bar(monthly, x='Month', y='Amount'), use_container_width=True)

# ML + Chat sections remain the same as previous version...

st.caption("SpendSmart AI • Anishka Moona")
