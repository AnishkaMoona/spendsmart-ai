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
st.markdown("**PDF + CSV • ML Anomaly Detection • Persistent History**")

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
    if uploaded_file.name.endswith('.pdf'):
        with pdfplumber.open(uploaded_file) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        st.warning("PDF parsing is basic. Best results with CSV.")
        # For now, show raw text
        st.text_area("Extracted Text", text, height=200)
    else:
        df = pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        
        def categorize(desc):
            desc = str(desc).lower()
            if any(k in desc for k in ['grab','taxi','mrt','bus','comfort']): return 'Transport'
            if any(k in desc for k in ['food','restaurant','hawker','starbucks','coffee']): return 'Food & Dining'
            if any(k in desc for k in ['shopee','lazada','amazon','shopping']): return 'Shopping'
            if any(k in desc for k in ['netflix','spotify','disney','gym']): return 'Subscriptions'
            if any(k in desc for k in ['salary','credit','transfer in']): return 'Income'
            return 'Others'
        
        df['Category'] = df['Description'].apply(categorize)
        save_to_db(df, uploaded_file.name)
        st.success(f"✅ Saved {len(df)} transactions!")

# Load all history
df_all = pd.read_sql("SELECT * FROM transactions", conn)
if not df_all.empty:
    df_all['Date'] = pd.to_datetime(df_all['Date'])
    df = df_all

if df.empty:
    st.info("Upload a statement to begin analysis")
    st.stop()

# ====================== DASHBOARD ======================
st.metric("Total Spent", f"SGD {df['Amount'].sum():,.0f}")

col1, col2 = st.columns(2)
with col1:
    cat_df = df.groupby('Category')['Amount'].sum().reset_index()
    st.plotly_chart(px.pie(cat_df, values='Amount', names='Category', title="Spending Breakdown"), use_container_width=True)

with col2:
    df['Month'] = df['Date'].dt.to_period('M').astype(str)
    monthly = df.groupby('Month')['Amount'].sum().reset_index()
    st.plotly_chart(px.bar(monthly, x='Month', y='Amount', title="Monthly Trend"), use_container_width=True)

# ====================== ML ANOMALY DETECTION ======================
st.subheader("🔍 ML Anomaly Detection (Isolation Forest)")
features = pd.get_dummies(df[['Category']])
features['Amount'] = df['Amount']

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

iso = IsolationForest(contamination=0.1, random_state=42)
df['Anomaly'] = iso.fit_predict(features_scaled)
df['Anomaly'] = df['Anomaly'].map({1: 'Normal', -1: 'Unusual ⚠️'})

st.dataframe(df[df['Anomaly'] == 'Unusual ⚠️'][['Date','Description','Amount','Category']], use_container_width=True)

# ====================== AI CHAT ======================
st.subheader("💬 Ask AI")
query = st.text_input("E.g. What are my biggest spending leaks?")
if query and st.button("Get Insight"):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    context = f"Total spent: SGD {df['Amount'].sum():,.0f}\nCategories: {df.groupby('Category')['Amount'].sum().to_dict()}\nUnusual transactions: {len(df[df['Anomaly']=='Unusual ⚠️'])}"
    
    resp = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are an expert personal finance advisor."},
            {"role": "user", "content": f"{context}\n\nQuestion: {query}"}
        ],
        temperature=0.7
    )
    st.success(resp.choices[0].message.content)

st.caption("SpendSmart AI • Built by Anishka Moona • ML + PDF + SQLite + MCP Ready")
