import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Lead Conversion Prediction", layout="centered")

# Header
st.title("🎯 Lead-to-Customer Predictor")
st.write("Predict whether a lead will convert based on CRM attributes like source, score, and follow-ups.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("leads.csv")
    return df

df = load_data()


# Encode categorical columns
df_encoded = df.copy()
le = LabelEncoder()
for col in ['LeadSource', 'Industry', 'Region']:
    df_encoded[col] = le.fit_transform(df_encoded[col])

# Split data
X = df_encoded.drop(columns=['LeadID', 'Converted'])
y = df_encoded['Converted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Model performance
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)


# Sidebar – Model Performance
with st.sidebar:
    st.header("📈 Model Performance")
    st.metric("Accuracy", f"{acc*100:.2f}%")
    st.metric("Test Samples", len(y_test))
    st.caption("🧠 Model trained on 1,000 synthetic CRM records.")


st.markdown("### 🧾 Provide Input Details")
st.info("👉 Select or enter the details below, then click **Check Conversion** to see the prediction.")

# User Input details
col1, col2 = st.columns(2)
lead_source = col1.selectbox("Lead Source", df['LeadSource'].unique(),help="Where the lead originated from (e.g., Ad, Referral, Website).")
industry = col2.selectbox("Industry", df['Industry'].unique(),help="Industry or business domain of the lead.")
region = col1.selectbox("Region", df['Region'].unique(), help="Geographical region of the lead.")
contacted = col2.selectbox("Contacted", [0, 1],help="Whether the lead has been contacted by the sales team.")
followups = col1.slider("Follow Ups", 0, 10, 3,help="How many times the lead was followed up by a sales agent.")
lead_score = col2.slider("Lead Score", 20, 100, 60, help="A numeric score indicating lead quality (higher = better chance of conversion).")

if st.button("Check conversion "):
    input_data = pd.DataFrame({
        'LeadSource': [lead_source],
        'Industry': [industry],
        'Contacted': [contacted],
        'FollowUps': [followups],
        'LeadScore': [lead_score],
        'Region': [region]
    })
    for col in ['LeadSource', 'Industry', 'Region']:
        input_data[col] = le.fit_transform(input_data[col])

    prob = model.predict_proba(input_data)[0][1]
    result = "✅ Likely to Convert" if prob > 0.5 else "❌ Unlikely to Convert"

    st.success(f"**Prediction:** {result}")
    st.info(f"**Probability of Conversion:** {prob*100:.2f}%")

st.markdown("---")

#footer
st.caption("Built by Suvaranamaliya Jothibabu.")
