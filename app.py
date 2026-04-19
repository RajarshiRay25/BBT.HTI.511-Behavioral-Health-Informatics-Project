# Import libraries

from narwhals import col
import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
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
        "Machine Learning Analysis",
        "About Me"
    ]
)

# -----------------------------
# Home Page
# -----------------------------

if menu == "Home":

    st.title("BBT.HTI.511 Behavioral Health Informatics Project")
    st.header("Welcome to the Student Stress Analysis Dashboard!")

    st.write("""
    This dashboard allows you to explore and analyze datasets related to student stress levels and well-being.
    Use the sidebar to navigate through different sections including dataset overview, correlation analysis, and visualizations.
    """)

    st.markdown("""
    This project integrates **two complementary datasets** focused on understanding stress among students.

    The data uses **Likert scale (1–5)** responses where higher values indicate greater intensity of the factor.

    For classification-based analysis:
    - **0**: Low Stress  
    - **1**: Moderate Stress  
    - **2**: High Stress  
    """)

    # =========================
    # DATASET 1
    # =========================
    st.subheader("📊 Dataset 1: Student Stress Monitoring Dataset")

    st.markdown("""
    This dataset investigates the root causes of stress among students using a nationwide survey.
    It contains ~20 variables grouped into five scientifically identified categories.
    """)

    # Psychological
    with st.expander("🧠 Psychological Factors"):
        st.markdown("""
        - anxiety_level  
        - self_esteem  
        - mental_health_history  
        - depression  
        """)

    # Physiological
    with st.expander("🏥 Physiological Factors"):
        st.markdown("""
        - headache  
        - blood_pressure  
        - sleep_quality  
        - breathing_problem  
        """)

    # Environmental
    with st.expander("🌆 Environmental Factors"):
        st.markdown("""
        - noise_level  
        - living_conditions  
        - safety  
        - basic_needs  
        """)

    # Academic
    with st.expander("🎓 Academic Factors"):
        st.markdown("""
        - academic_performance  
        - study_load  
        - teacher_student_relationship  
        - future_career_concerns  
        """)

    # Social
    with st.expander("🤝 Social Factors"):
        st.markdown("""
        - social_support  
        - peer_pressure  
        - extracurricular_activities  
        - bullying  
        """)

    st.markdown("""
    **📌 Data Source:**  
    https://www.kaggle.com/datasets/mdsultanulislamovi/student-stress-monitoring-datasets
    """)

    st.divider()

    # =========================
    # DATASET 2
    # =========================
    st.subheader("📊 Dataset 2: Survey on Stress & Well-being (Ages 18–21)")

    st.markdown("""
    This dataset contains responses from **843 college students aged 18–21** collected via Google Forms.
    It focuses on emotional, physical, academic, and social well-being factors.
    """)

    with st.expander("👤 Demographic"):
        st.markdown("""
        - Gender (0 = Male, 1 = Female)  
        - Age (18–21)  
        """)

    with st.expander("🧠 Emotional & Stress Indicators"):
        st.markdown("""
        - Stress experience  
        - Anxiety / tension  
        - Sleep problems  
        - Sadness / low mood  
        - Loneliness  
        - Irritability  
        """)

    with st.expander("🩺 Physical & Health Indicators"):
        st.markdown("""
        - Headaches  
        - Illness  
        - Weight changes  
        """)

    with st.expander("📚 Academic & Environmental Factors"):
        st.markdown("""
        - Academic workload  
        - Peer competition  
        - Confidence issues  
        - Class attendance  
        - Environment stress  
        """)

    with st.expander("💬 Social Factors"):
        st.markdown("""
        - Relationship stress  
        - Lack of relaxation time  
        """)

    st.markdown("""
    **📌 Target Variable:**  
    - Which type of stress do you primarily experience?  
      (Eustress / Distress / No Stress)
    """)

    st.divider()

    st.markdown("""
    **📌 Data Sources**

    - Kaggle Dataset: Student Stress Monitoring Dataset  
      https://www.kaggle.com/datasets/mdsultanulislamovi/student-stress-monitoring-datasets  

    - Survey Dataset: 843 student well-being responses (Google Forms-based)
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

    st.subheader("Target Variable Distribution")
    if "stress_level" in df.columns:
        st.write(df["stress_level"].value_counts())
    elif "Which type of stress do you primarily experience?" in df.columns:
        st.write(df["Which type of stress do you primarily experience?"].value_counts())

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

    st.header("📊 Feature vs Stress")

    exclude_cols = ["stress_level", "Which type of stress do you primarily experience?"]

    feature = st.selectbox(
        "Select Feature",
        [col for col in df.columns if col not in exclude_cols]
    )

    # 🔹 Case 1: Numeric stress level
    if "stress_level" in df.columns:

        fig, ax = plt.subplots()

        sns.boxplot(
            x="stress_level",
            y=feature,
            data=df,
            palette="Spectral",
            ax=ax
        )

        ax.set_xticklabels(["Low Stress", "Moderate Stress", "High Stress"])
        ax.set_xlabel("Stress Level")
        ax.set_ylabel(feature.replace("_", " ").title())
        ax.set_title(f"{feature.replace('_', ' ').title()} vs Stress Level")

        st.pyplot(fig)

    # 🔹 Case 2: Stress type column
    elif "Which type of stress do you primarily experience?" in df.columns:

        stress_map = {
            "Distress (Negative Stress) - Stress that causes anxiety and impairs well-being.": "Distress",
            "Eustress (Positive Stress) - Stress that motivates and enhances performance.": "Eustress",
            "No Stress - Currently experiencing minimal to no stress.": "No Stress"
        }

        temp_df = df.copy()
        temp_df["stress_short"] = temp_df[
            "Which type of stress do you primarily experience?"
        ].map(stress_map)

        fig, ax = plt.subplots()

        sns.boxplot(
            x="stress_short",
            y=feature,
            data=temp_df,
            palette="Spectral",
            order=["Distress", "Eustress", "No Stress"],
            ax=ax
        )

        ax.set_xlabel("Stress Type")
        ax.set_ylabel(feature.replace("_", " ").title())
        ax.set_title(f"{feature.replace('_', ' ').title()} vs Stress Type")

        st.pyplot(fig)

    # 🔹 Case 3: Neither column exists → show alert
    else:
        st.warning("⚠️ This dataset does not contain a recognizable stress variable.\n\nYou can upload another dataset with 'stress_level' or stress type to perform analysis.")

    

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

            # ✅ FIXED LABELS
            ax.set_xlabel(col2.replace("_", " ").title())  # columns
            ax.set_ylabel(col1.replace("_", " ").title())  # rows

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

    st.header("🔍 SHAP Feature Importance")

    import shap
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor

    # Case 1: stress_level exists
    if "stress_level" in df.columns:

        target = "stress_level"
        X = df.drop(target, axis=1)
        y = df[target]

    # Case 2: stress type column exists
    elif "Which type of stress do you primarily experience?" in df.columns:

        stress_map = {
            "Distress (Negative Stress) - Stress that causes anxiety and impairs well-being.": 2,
            "Eustress (Positive Stress) - Stress that motivates and enhances performance.": 1,
            "No Stress - Currently experiencing minimal to no stress.": 0
        }

        temp_df = df.copy()
        temp_df["stress_encoded"] = temp_df[
            "Which type of stress do you primarily experience?"
        ].map(stress_map)

        target = "stress_encoded"

        X = temp_df.drop(
            ["Which type of stress do you primarily experience?", target],
            axis=1
        )
        y = temp_df[target]

    # Case 3: no valid target
    else:
        st.warning("⚠️ No valid stress variable found.\n\nUpload dataset with 'stress_level' or stress type to perform SHAP analysis.")
        st.stop()

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Loader
    with st.spinner("⏳ Training model and computing SHAP values..."):

        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

    st.success("✅ SHAP analysis complete!")

    # Case 1: SHAP Summary Plot
    st.subheader("📊 SHAP Summary Plot")

    fig = plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(fig)

    # Case 2: Interpretation
    st.subheader("🧠 Interpretation")

    st.markdown("""
    - Features at the top have the highest overall impact on stress prediction.  
    - Positive SHAP values indicate an increase in predicted stress level.  
    - Negative SHAP values indicate a decrease in predicted stress level.  
    - A spread on both sides suggests the feature has different effects depending on context.  

    ⚠️ SHAP explains model behavior and feature contribution, not causation.
    """)

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


# -----------------------------
# Machine Learning Analysis
# -----------------------------
elif menu == "Machine Learning Analysis":

    st.header("📊 Model Analysis & Interpretation")

    # Case 1: stress_level exists
    if "stress_level" in df.columns:

        target = "stress_level"
        labels = ["Low", "Moderate", "High"]

        X = df.drop(target, axis=1)
        y = df[target]

    # Case 2: stress type column exists
    elif "Which type of stress do you primarily experience?" in df.columns:

        stress_map = {
            "Distress (Negative Stress) - Stress that causes anxiety and impairs well-being.": 0,
            "Eustress (Positive Stress) - Stress that motivates and enhances performance.": 1,
            "No Stress - Currently experiencing minimal to no stress.": 2
        }

        temp_df = df.copy()
        temp_df["stress_encoded"] = temp_df[
            "Which type of stress do you primarily experience?"
        ].map(stress_map)

        target = "stress_encoded"
        labels = ["Distress", "Eustress", "No Stress"]

        X = temp_df.drop(
            ["Which type of stress do you primarily experience?", target],
            axis=1
        )
        y = temp_df[target]

    # Case 3: no valid target
    else:
        st.warning("⚠️ No valid stress variable found.\n\nUpload dataset with 'stress_level' or stress type.")
        st.stop()

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Loader
    with st.spinner("⏳ Training model... please wait..."):

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    st.success("✅ Model training complete!")

    # Accuracy
    st.subheader("📈 Model Performance")

    acc = accuracy_score(y_test, y_pred)
    st.write(f"**Accuracy:** {acc:.2f}")

    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    st.subheader("🔍 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="coolwarm",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)

    # Feature Importance
    st.subheader("⭐ Feature Importance")

    importances = model.feature_importances_

    feature_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    fig2, ax2 = plt.subplots(figsize=(8,5))

    sns.barplot(
        data=feature_df,
        x="Importance",
        y="Feature",
        palette="Spectral",
        ax=ax2
    )

    ax2.set_title("Feature Importance")

    st.pyplot(fig2)

    # Interpretation
    st.subheader("🧠 Interpretation")

    st.markdown(f"""
    - The model classifies stress into **{', '.join(labels)}** categories.  
    - Key features influencing stress prediction are shown above.  
    - Some overlap between categories may lead to misclassification.  
    - Results indicate **associations, not causation**.
    """)