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
st.markdown("**Robust PDF + CSV • ML Anomaly Detection**")

# ====================== DATABASE ======================
conn = sqlite3.connect('spendsmart.db', check_same_thread=False)
conn.execute('''CREATE TABLE IF NOT EXISTS transactions 
             (id INTEGER PRIMARY KEY, date TEXT, description TEXT, amount REAL, category TEXT, source TEXT)''')

def save_to_db(df, source):
    for _, row in df.iterrows():
        conn.execute("""INSERT OR IGNORE INTO transactions 
                     (date, description, amount, category, source) 
                     VALUES (?, ?, ?, ?, ?)""",
                    (row['Date'].strftime('%Y-%m-%d'), 
                     row['Description'], 
                     float(row['Amount']), 
                     row.get('Category', 'Others'), 
                     source))
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
            st.subheader("PDF Extracted Text")
            st.text_area("Raw Text", text[:1500], height=250)
            st.info("PDF parsing is basic. Use CSV for full functionality.")
            
        else:
            # Robust CSV reading
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except:
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
                except:
                    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
            
            # Flexible column detection
            col_map = {}
            for col in df.columns:
                if 'date' in col.lower():
                    col_map['Date'] = col
                elif 'desc' in col.lower() or 'particular' in col.lower():
                    col_map['Description'] = col
                elif 'amount' in col.lower() or 'debit' in col.lower() or 'credit' in col.lower():
                    col_map['Amount'] = col
            
            if 'Date' in col_map and 'Description' in col_map and 'Amount' in col_map:
                df = df[[col_map['Date'], col_map['Description'], col_map['Amount']]].copy()
                df.columns = ['Date', 'Description', 'Amount']
                
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.dropna(subset=['Date'])
                df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
                df = df[df['Amount'] != 0].copy()
                
                def categorize(desc):
                    desc = str(desc).lower()
                    if any(k in desc for k in ['grab','taxi','mrt','bus']): return 'Transport'
                    if any(k in desc for k in ['food','restaurant','hawker','starbucks']): return 'Food & Dining'
                    if any(k in desc for k in ['shopee','lazada','amazon','shopping']): return 'Shopping'
                    if any(k in desc for k in ['netflix','spotify']): return 'Subscriptions'
                    if any(k in desc for k in ['salary','credit']): return 'Income'
                    return 'Others'
                
                df['Category'] = df['Description'].apply(categorize)
                save_to_db(df, uploaded_file.name)
                st.success(f"✅ Loaded {len(df)} transactions!")
            else:
                st.error("Could not auto-detect columns. Please ensure your CSV has Date, Description, and Amount columns.")
                st.stop()
                
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.stop()

# Load all history safely
try:
    df_all = pd.read_sql("SELECT * FROM transactions", conn)
    if not df_all.empty:
        df_all['Date'] = pd.to_datetime(df_all['Date'], errors='coerce')
        if df.empty:
            df = df_all
        else:
            df = pd.concat([df, df_all], ignore_index=True)
except:
    pass

if df.empty:
    st.info("👆 Please upload a CSV or PDF bank statement")
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

# ML + Chat (same as before)
st.caption("SpendSmart AI • Anishka Moona")
