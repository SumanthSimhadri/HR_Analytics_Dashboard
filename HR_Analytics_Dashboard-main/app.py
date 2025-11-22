# import streamlit as st
# import pandas as pd
# from model import train_model
#
# st.set_page_config(page_title="HR Analytics Dashboard", layout="wide")
# st.title("🧠 AI-Driven HR Analytics Dashboard")
#
# model = train_model()
#
# st.sidebar.header("Employee Input")
# age = st.sidebar.slider("Age", 20, 60, 30)
# job_sat = st.sidebar.slider("Job Satisfaction (1-4)", 1, 4, 3)
# income = st.sidebar.number_input("Monthly Income", 1000, 20000, 5000)
# overtime = st.sidebar.radio("OverTime", ["Yes", "No"])
# overtime_val = 1 if overtime == "Yes" else 0
#
# input_df = pd.DataFrame([[age, job_sat, income, overtime_val]],
#                         columns=["Age", "JobSatisfaction", "MonthlyIncome", "OverTime"])
#
# if st.button("Predict Attrition Risk"):
#     prediction = model.predict(input_df)[0]
#     st.subheader("Prediction:")
#     if prediction == 1:
#         st.error("⚠️ High Risk of Attrition")
#     else:
#         st.success("✅ Low Risk of Attrition")
#
# # Optional: Visualize data
# if st.checkbox("Show Sample Data"):
#     df = pd.read_csv("data.csv")
#     st.dataframe(df.head())


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model import train_model

# Page config
st.set_page_config(
    page_title="AI-Driven HR Analytics",
    layout="wide",
    page_icon="🧠"
)

# Load model
model = train_model()

# Title and description
st.markdown("<h1 style='text-align: center;'>🧠 AI-Driven HR Analytics Dashboard</h1>", unsafe_allow_html=True)
st.markdown("### Get real-time insights into employee attrition risk and engagement levels")

# Sidebar input
st.sidebar.header("📋 Enter Employee Details")
age = st.sidebar.slider("Age", 18, 60, 30)
job_sat = st.sidebar.slider("Job Satisfaction (1-4)", 1, 4, 3)
income = st.sidebar.number_input("Monthly Income", 1000, 20000, 5000)
overtime = st.sidebar.radio("OverTime", ["Yes", "No"])
overtime_val = 1 if overtime == "Yes" else 0

# Input DataFrame
input_df = pd.DataFrame([[age, job_sat, income, overtime_val]],
                        columns=["Age", "JobSatisfaction", "MonthlyIncome", "OverTime"])

# Prediction card
col1, col2, col3 = st.columns(3)
with col2:
    if st.button("🔍 Predict Attrition Risk"):
        prediction = model.predict(input_df)[0]
        if prediction == 1:
            st.markdown("<div style='padding: 1rem; background-color: #ffe6e6; border-left: 5px solid red;'>"
                        "<h4 style='color: red;'>⚠️ High Risk of Attrition</h4>"
                        "</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='padding: 1rem; background-color: #e6ffec; border-left: 5px solid green;'>"
                        "<h4 style='color: green;'>✅ Low Risk of Attrition</h4>"
                        "</div>", unsafe_allow_html=True)

# Visualizations
st.markdown("---")
st.subheader("📊 HR Dataset Insights")

df = pd.read_csv("data.csv")

# KPIs
k1, k2, k3 = st.columns(3)
with k1:
    st.metric("Total Employees", len(df))
with k2:
    st.metric("Attrition Rate", f"{df['Attrition'].value_counts(normalize=True).get('Yes',0)*100:.1f}%")
with k3:
    st.metric("Avg. Monthly Income", f"${df['MonthlyIncome'].mean():,.0f}")

# Charts
chart1, chart2 = st.columns(2)

with chart1:
    st.markdown("#### Attrition by Department")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x="Department", hue="Attrition", ax=ax1)
    plt.xticks(rotation=15)
    st.pyplot(fig1)

with chart2:
    st.markdown("#### Income vs Attrition")
    fig2, ax2 = plt.subplots()
    sns.boxplot(data=df, x="Attrition", y="MonthlyIncome", ax=ax2)
    st.pyplot(fig2)

# Data preview
with st.expander("📄 Show Raw Dataset"):
    st.dataframe(df.head(10))
