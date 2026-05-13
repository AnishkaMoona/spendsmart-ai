import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from groq import Groq
import os
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np

load_dotenv()

st.set_page_config(page_title="SpendSmart AI", page_icon="💰", layout="wide")
st.title("💰 SpendSmart AI")
st.markdown("**Smart Finance Agent with ML Anomaly Detection + LLM Insights**")

# ====================== UPLOAD ======================
st.sidebar.header("Upload Bank Statement")
uploaded_file = st.sidebar.file_uploader("CSV (Date, Description, Amount)", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV to start analysis")
    st.stop()

df = pd.read_csv(uploaded_file)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])
df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
df = df[df['Amount'] != 0]  # remove zero amounts

# ====================== DYNAMIC CATEGORIZATION (LLM) ======================
st.sidebar.subheader("Categorizing with AI...")

@st.cache_data
def categorize_with_llm(df_sample):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    sample_text = df_sample[['Description', 'Amount']].head(30).to_string()
    
    prompt = f"""Categorize these transactions into: Food & Dining, Transport, Shopping, Subscriptions, Bills, Entertainment, Healthcare, Income, Others.
    Return only the category name for each.
    
    Transactions:
    {sample_text}
    """
    
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=800
    )
    # This is simplified — in real version we'd parse better
    return df  # For now we fall back to rule-based + LLM in chat

df['Category'] = df['Description'].apply(lambda x: "Others")  # placeholder

# Better rule-based + flag for LLM
def smart_categorize(desc):
    desc = str(desc).lower()
    if any(k in desc for k in ['grab','taxi','mrt','bus','comfort']): return 'Transport'
    if any(k in desc for k in ['food','restaurant','hawker','mcd','starbucks','coffee']): return 'Food & Dining'
    if any(k in desc for k in ['shopee','lazada','amazon','taobao']): return 'Shopping'
    if any(k in desc for k in ['netflix','spotify','disney','gym']): return 'Subscriptions'
    if any(k in desc for k in ['salary','credit','transfer in']): return 'Income'
    return 'Others'

df['Category'] = df['Description'].apply(smart_categorize)

# ====================== ML ANOMALY DETECTION ======================
st.subheader("🔍 ML-Powered Insights")

# Prepare data for Isolation Forest
df_ml = df.copy()
df_ml['DayOfWeek'] = df_ml['Date'].dt.dayofweek
df_ml['Month'] = df_ml['Date'].dt.month
features = pd.get_dummies(df_ml[['Category', 'DayOfWeek', 'Month']])
features['Amount'] = df_ml['Amount']

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

model = IsolationForest(contamination=0.1, random_state=42)
df['Anomaly'] = model.fit_predict(features_scaled)
df['Anomaly'] = df['Anomaly'].map({1: 'Normal', -1: 'Unusual'})

# ====================== DASHBOARD ======================
col1, col2, col3 = st.columns(3)
col1.metric("Total Spent", f"SGD {df['Amount'].sum():,.0f}")
col2.metric("Transactions", len(df))
col3.metric("Unusual Transactions", len(df[df['Anomaly'] == 'Unusual']))

# Charts
tab1, tab2, tab3 = st.tabs(["Overview", "ML Anomalies", "AI Chat"])

with tab1:
    cat_df = df.groupby('Category')['Amount'].sum().reset_index()
    st.plotly_chart(px.pie(cat_df, values='Amount', names='Category'), use_container_width=True)
    
    df['Month'] = df['Date'].dt.to_period('M').astype(str)
    monthly = df.groupby('Month')['Amount'].sum().reset_index()
    st.plotly_chart(px.bar(monthly, x='Month', y='Amount'), use_container_width=True)

with tab2:
    st.subheader("ML Detected Unusual Transactions")
    anomalies = df[df['Anomaly'] == 'Unusual'].sort_values('Amount', ascending=False)
    st.dataframe(anomalies[['Date', 'Description', 'Amount', 'Category']])
    st.caption("These were flagged as statistical outliers by Isolation Forest")

with tab3:
    st.subheader("💬 Ask AI Anything")
    query = st.text_input("E.g. What are my top spending categories? Any red flags?")
    
    if query and st.button("Ask AI"):
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        context = f"Total: SGD {df['Amount'].sum():,.0f}\nCategories: {df.groupby('Category')['Amount'].sum().to_dict()}\nUnusual: {len(anomalies)} transactions"
        
        resp = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "system", "content": "You are an expert personal finance advisor."},
                      {"role": "user", "content": f"{context}\n\nQuestion: {query}"}],
            temperature=0.7
        )
        st.success(resp.choices[0].message.content)

st.caption("SpendSmart AI • ML + LLM Powered • Built by Anishka Moona")
