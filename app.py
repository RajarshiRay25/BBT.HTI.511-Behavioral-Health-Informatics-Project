# Import libraries

from narwhals import col
import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import shap 

# Home Page

st.set_page_config(page_title="Student Stress Dashboard", layout="wide")

st.title("📊 Student Stress Analysis Dashboard")

# Upload dataset
uploaded_file = st.file_uploader("Upload Student Stress Dataset (CSV)", type=["csv"])

# Load dataset
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Please upload a dataset to proceed.")
    # st.stop()

# Sidebar navigation
menu = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Dataset Overview",
        "Correlation Analysis",
        "Heatmaps",
        "Boxplot Analysis",
        "SHAP Analysis",
        "About Me"
    ]
)

if menu =="Home":
    st.title("BBT.HTI.511 Behavioral Health Informatics Project")
    st.header("Welcome to the Student Stress Analysis Dashboard!")
    st.write("""
        This dashboard allows you to explore and analyze a dataset related to student stress levels. 
        Use the sidebar to navigate through different sections of the analysis, including dataset overview, correlation analysis, and various visualizations.
    """)

    st.markdown(
    """
    This dataset investigates the **root causes of stress among students**, derived from a **nationwide survey**.  
    It contains **~20 key variables** grouped into **five scientifically identified categories** that influence student stress.
    The numbers within each parameter are based on a **5-point Likert scale** (1-5), where higher values indicate greater intensity or frequency of the factor.

    For the target variable, **stress_level**, the values are categorized as follows:
    - **0**: Low Stress 
    - **1**: Moderate Stress
    - **2**: High Stress



    Use the sections below to explore each category.
    """
    )

    # Psychological
    with st.expander("🧠 Psychological Factors"):
        st.markdown("""
    - **anxiety_level** – intensity of anxiety experienced by students  
    - **self_esteem** – confidence and self-worth perception  
    - **mental_health_history** – prior mental health conditions  
    - **depression** – depressive symptoms affecting students
    """)

    # Physiological
    with st.expander("🏥 Physiological Factors"):
        st.markdown("""
    - **headache** – frequency of stress-related headaches  
    - **blood_pressure** – blood pressure variations  
    - **sleep_quality** – quality of sleep patterns  
    - **breathing_problem** – respiratory discomfort linked to stress
    """)

    # Environmental
    with st.expander("🌆 Environmental Factors"):
        st.markdown("""
    - **noise_level** – environmental noise around students  
    - **living_conditions** – housing and accommodation quality  
    - **safety** – perceived safety in surroundings  
    - **basic_needs** – access to food, water, and essential resources
    """)

    # Academic
    with st.expander("🎓 Academic Factors"):
        st.markdown("""
    - **academic_performance** – student academic achievement  
    - **study_load** – academic workload intensity  
    - **teacher_student_relationship** – quality of interactions with teachers  
    - **future_career_concerns** – stress related to career prospects
    """)

    # Social
    with st.expander("🤝 Social Factors"):
        st.markdown("""
    - **social_support** – support from friends and family  
    - **peer_pressure** – pressure from peers  
    - **extracurricular_activities** – involvement in non-academic activities  
    - **bullying** – experiences of bullying or harassment
    """)
        
    st.divider()

    st.markdown("""
    **📌 Data Source**

    The dataset used in this project is taken from the following source:

    🔗 https://www.kaggle.com/datasets/mdsultanulislamovi/student-stress-monitoring-datasets
    """)


# -----------------------------
# Dataset Overview
# -----------------------------
if menu == "Dataset Overview":

    st.header("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Summary Statistics")
    st.write(df.describe())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())


# -----------------------------
# Correlation Analysis
# -----------------------------
elif menu == "Correlation Analysis":
    st.header("Spearman Correlation Heatmap")
    if "Which type of stress do you primarily experience?" in df.columns:
        df = df.drop("Which type of stress do you primarily experience?", axis=1)
        corr = df.corr(method="spearman")
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(corr, cmap="Spectral", ax=ax)
        st.pyplot(fig)
    else:
        corr = df.corr(method="spearman")

        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(corr, cmap="Spectral", ax=ax)
        st.pyplot(fig)

# -----------------------------
# Boxplots
# -----------------------------
elif menu == "Boxplot Analysis":

    st.header("Feature vs Stress Level")

    feature = st.selectbox(
        "Select Feature",
        [col for col in df.columns if col != "stress_level" or col!="Which type of stress do you primarily experience?"]
    )

    
    if "stress_level" in df.columns:
        fig, ax = plt.subplots()
        sns.boxplot(x="stress_level", y=feature, data=df, ax=ax,palette="Spectral")
        # ax.set_xticklabels(["Academic", "Social", "Financial", "Health", "Other"])
        ax.set_xticklabels(["Low Stress", "Moderate Stress", "High Stress"])
        # Add axis labels and title
        ax.set_xlabel("Stress Level")
        ax.set_ylabel(feature.replace("_", " ").title())
        ax.set_title(f"{feature.replace('_', ' ').title()} vs Stress Level")
        st.pyplot(fig)

    elif "Which type of stress do you primarily experience?" in df.columns:
        stress_map = {
            "Distress (Negative Stress) - Stress that causes anxiety and impairs well-being." : "Distress",
            "Eustress (Positive Stress) - Stress that motivates and enhances performance."    : "Eustress",
            "No Stress - Currently experiencing minimal to no stress."                        : "No Stress"
        }
    
    # Make a temp column with short names
    df["stress_short"] = df["Which type of stress do you primarily experience?"].map(stress_map)
    
    fig, ax = plt.subplots()
    sns.boxplot(
        x="stress_short",
        y=feature,
        data=df,
        ax=ax,
        palette="Spectral",
        order=["Distress", "Eustress", "No Stress"]  # fixed order
    )
    ax.set_xlabel("Stress Type")
    ax.set_ylabel(feature.replace("_", " ").title())
    ax.set_title(f"{feature.replace('_', ' ').title()} vs Stress Type")
    st.pyplot(fig)
    # Set custom x-axis labels

    

# -----------------------------
# Crosstab Heatmaps
# -----------------------------
elif menu == "Heatmaps":

    st.header("Heatmaps")

    heatmap_type = st.radio(
        "Select Heatmap Type:",
        ["Feature Interaction (Crosstab)", "Correlation with Stress Level"]
    )

    if heatmap_type == "Feature Interaction (Crosstab)":
        col1 = st.selectbox("Select Feature 1", df.columns)
        col2 = st.selectbox("Select Feature 2", df.columns)

        if col1 != col2:
            table = pd.crosstab(df[col1], df[col2])

            fig, ax = plt.subplots()
            sns.heatmap(table, annot=True, cmap="YlGnBu", linewidths=0.5, ax=ax)
            ax.set_title(f"{col1.replace('_',' ').title()} vs {col2.replace('_',' ').title()} Crosstab Heatmap")
            ax.set_xlabel(col1.replace("_", " ").title())
            ax.set_ylabel(col2.replace("_", " ").title())
            st.pyplot(fig)

    elif heatmap_type == "Correlation with Stress Level":
        st.subheader("Spearman Correlation of Features with Stress Level")

        # Compute Spearman correlation
        stress_corr = df.corr(method="spearman")[["stress_level"]].sort_values(by="stress_level", ascending=False)

        fig, ax = plt.subplots(figsize=(5,8))
        sns.heatmap(stress_corr, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
        ax.set_title("Feature Correlation with Stress Level")
        st.pyplot(fig)

# -----------------------------
# SHAP Analysis
# -----------------------------
elif menu == "SHAP Analysis":

    st.header("SHAP Feature Importance")

    X = df.drop("stress_level", axis=1)
    y = df["stress_level"]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    st.subheader("SHAP Summary Plot")

    fig = plt.figure()
    shap.summary_plot(shap_values, X, show=False)

    st.pyplot(fig)

elif menu =="About Me":
    # Page Title
    st.header("👤 About Me")

    # Name
    st.subheader("Name")
    st.write("**Rajarshi Ray**")

    # Study
    st.subheader("Current Study")
    st.write("MSc(Tech) in Biomedical Sciences and Engineering")

    # Major / Track
    st.subheader("Major / Track")
    st.write("Biomedical Informatics, Bioinformatics Track")

    # University
    st.subheader("University")
    st.write("Tampere University, Finland")

    # Optional: add a small paragraph / intro
    st.markdown("""
    Hi! I am Rajarshi, currently pursuing my MSc in Biomedical Sciences and Engineering at Tampere University.  
    I am focusing on **Biomedical Informatics** with a track in **Bioinformatics**, exploring health data analysis and computational methods in biomedical research.
    """)