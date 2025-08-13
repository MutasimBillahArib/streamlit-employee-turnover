import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier

# ======================
# CONFIGURATION
# ======================
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================
# LOAD MODEL & RESOURCES
# ======================
@st.cache_resource
def load_model():
    """Load pre-trained model and feature names"""
    with open('rf2_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Hardcoded feature order (must match training)
    feature_names = [
        'last_evaluation', 'number_project', 'tenure', 'work_accident',
        'promotion_last_5years', 'salary', 'department_IT', 'department_RandD',
        'department_accounting', 'department_hr', 'department_management',
        'department_marketing', 'department_product_mng', 'department_sales',
        'department_support', 'department_technical', 'overworked'
    ]
    
    return model, feature_names

model, feature_names = load_model()

# ======================
# HELPER FUNCTIONS
# ======================
def preprocess_input(user_input):
    """Convert user input to model-ready format"""
    # Convert salary to numeric
    salary_map = {'low': 0, 'medium': 1, 'high': 2}
    user_input['salary'] = salary_map[user_input['salary']]
    
    # Calculate overworked (175 hours threshold from EDA)
    user_input['overworked'] = 1 if user_input['monthly_hours'] > 175 else 0
    
    # Create department dummies
    departments = [
        'IT', 'RandD', 'accounting', 'hr', 'management', 
        'marketing', 'product_mng', 'sales', 'support', 'technical'
    ]
    for dept in departments:
        user_input[f'department_{dept}'] = 1 if user_input['department'] == dept else 0
    
    # Drop unused fields
    user_input.pop('department')
    user_input.pop('monthly_hours')
    
    return pd.DataFrame([user_input])[feature_names]

def get_risk_factors(user_input, shap_values):
    """Generate business-friendly risk explanations"""
    factors = []
    
    # Calculate overworked internally (this is the fix)
    overworked = 1 if user_input['monthly_hours'] > 175 else 0
    
    # 1. Overwork factor (strongest EDA finding)
    if user_input['monthly_hours'] > 175:
        factors.append((
            "⚠️ **Extreme Workload**", 
            f"Working {user_input['monthly_hours']} hrs/month (>175 threshold) → 3.2x higher attrition risk"
        ))
    
    # 2. Project overload (critical EDA finding)
    if user_input['number_project'] >= 7:
        factors.append((
            "💥 **Critical Workload**", 
            "Employees with 7+ projects have 100% attrition rate (per your data)"
        ))
    elif user_input['number_project'] > 5:
        factors.append((
            "⚠️ **High Project Load**", 
            f"{user_input['number_project']} projects → 68% higher risk than average"
        ))
    
    # 3. Promotion gap
    if user_input['promotion_last_5years'] == 0 and user_input['number_project'] > 4:
        factors.append((
            "📉 **Promotion Gap**", 
            "No promotion + high workload → 78% attrition rate in your data"
        ))
    
    # 4. Tenure risk
    if 2 < user_input['tenure'] < 5:
        factors.append((
            "⚠️ **Mid-Career Risk**", 
            "Employees with 2-5 years tenure are most vulnerable to attrition"
        ))
    elif user_input['tenure'] > 6:
        factors.append((
            "✅ **Loyalty Strength**", 
            "Tenure >6 years = strong retention indicator (low risk)"
        ))
    
    # 5. Satisfaction proxy (using evaluation score)
    # Use the internally calculated 'overworked' variable instead of user_input['overworked']
    if user_input['last_evaluation'] > 0.8 and overworked:
        factors.append((
            "💔 **Burnout Risk**", 
            "High performer + overworked → prime attrition candidate"
        ))
    
    return factors[:3]  # Top 3 factors

def generate_recommendations(risk_level, risk_factors):
    """Create actionable HR recommendations"""
    recommendations = []
    
    if risk_level == "HIGH":
        recommendations.append("🚨 **URGENT ACTION NEEDED**")
        recommendations.append("- Schedule retention interview within 48 hours")
        recommendations.append("- Reduce workload to <175 hours/month immediately")
        
        if any("Promotion" in f[0] for f in risk_factors):
            recommendations.append("- Fast-track promotion review")
        
        if any("Workload" in f[0] for f in risk_factors):
            recommendations.append("- Reassign 1-2 projects to balance workload")
    
    elif risk_level == "MEDIUM":
        recommendations.append("🛠️ **PROACTIVE STEPS**")
        recommendations.append("- Monitor satisfaction monthly")
        recommendations.append("- Offer skill development opportunities")
        
        if any("Promotion" in f[0] for f in risk_factors):
            recommendations.append("- Discuss career path in next review")
    
    else:  # LOW
        recommendations.append("✅ **STABLE EMPLOYEE**")
        recommendations.append("- Maintain current engagement practices")
        recommendations.append("- Consider for mentorship roles")
    
    return recommendations

# ======================
# SIDEBAR - ABOUT & CREDITS
# ======================
# st.sidebar.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=100)
st.sidebar.title("About This App")
st.sidebar.info("""
This predictor helps HR teams identify at-risk employees 
using machine learning. Based on Salifort Motors' 
10,500 employee records.

**Key Capabilities:**
- Real-time attrition risk scoring
- Actionable retention recommendations
- What-if scenario analysis
- Model explainability
""")

st.sidebar.subheader("Project Credentials")
st.sidebar.markdown("""
- 🏆 Google Advanced Data Analytics Certificate
- 📊 Built with Python & scikit-learn
- 🌳 Random Forest Classifier (AUC: 0.92)
""")

st.sidebar.link_button("View Project Code", "https://github.com/MutasimBillahArib/google-advanced-data-analytics-coursera/tree/main/employee-turnover-analysis")
# st.sidebar.link_button("HR Dashboard Template", "https://share.streamlit.io/your-app")

# ======================
# MAIN APP - HERO SECTION
# ======================
st.title("🔥 Employee Attrition Predictor")
st.subheader("Proactively retain talent with AI-driven insights")
st.markdown("""
Identify at-risk employees *before* they quit. Based on your analysis of 10,500+ employee records, 
this tool predicts attrition probability and reveals **actionable drivers** (like overwork or promotion gaps).
""")

# ======================
# INPUT SECTION
# ======================
st.divider()
st.header("🔍 Simulate Employee Profile")
st.caption("Adjust sliders to match real employee data. Default values = dataset medians.")

# Create two columns for input organization
col1, col2 = st.columns(2)

with col1:
    st.subheader("Employee Profile")
    tenure = st.slider(
        "Tenure (years)", 
        0, 10, 3,
        help="⚠️ Critical risk: 2-5 years = highest attrition (per your EDA)"
    )
    department = st.selectbox(
        "Department", 
        ["sales", "technical", "support", "IT", "product_mng", 
         "marketing", "RandD", "accounting", "hr", "management"],
        index=0
    )
    salary = st.selectbox(
        "Salary Level", 
        ["low", "medium", "high"],
        index=1
    )
    work_accident = 1 if st.toggle("Had Work Accident?") else 0

with col2:
    st.subheader("Workload & Performance")
    monthly_hours = st.slider(
        "Monthly Hours", 
        80, 300, 180,
        help="⚠️ Critical risk: >175 hours = 3.2x higher attrition (your finding!)"
    )
    number_project = st.number_input(
        "Number of Projects", 
        1, 10, 4,
        help="💥 Critical risk: 7+ projects = 100% attrition rate"
    )
    last_evaluation = st.slider(
        "Last Evaluation Score", 
        0.4, 1.0, 0.7,
        help="High performers (>0.8) + overwork = burnout risk"
    )
    promotion = 1 if st.toggle("Promoted in Last 5 Years?") else 0

# ======================
# PREDICTION SECTION
# ======================
st.divider()
st.header("📊 Prediction Results")

# Process input and predict
user_input = {
    'last_evaluation': last_evaluation,
    'number_project': number_project,
    'tenure': tenure,
    'work_accident': work_accident,
    'promotion_last_5years': promotion,
    'salary': salary,
    'department': department,
    'monthly_hours': monthly_hours
}

# Preprocess and predict
input_df = preprocess_input(user_input.copy())  # Create a copy before preprocessing
prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]

# Calculate risk level
if probability < 0.3:
    risk_level = "LOW"
    risk_color = "green"
elif probability < 0.7:
    risk_level = "MEDIUM"
    risk_color = "orange"
else:
    risk_level = "HIGH"
    risk_color = "red"

# Display risk meter
st.markdown(f"### 🚨 Attrition Risk: <span style='color:{risk_color}'>{risk_level}</span> ({probability:.0%})", 
            unsafe_allow_html=True)
st.progress(probability)

# Generate explanations
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_df)
risk_factors = get_risk_factors(user_input, shap_values)
recommendations = generate_recommendations(risk_level, risk_factors)

# Display risk factors
st.subheader("Key Risk Drivers")
for factor in risk_factors:
    st.warning(f"**{factor[0]}**\n\n{factor[1]}")

# Display recommendations
st.subheader("Recommended Actions")
for rec in recommendations:
    st.success(rec)

# ======================
# WHAT-IF ANALYSIS
# ======================
st.divider()
st.header("💡 What-If Scenario Analysis")
st.caption("Test how interventions would change attrition risk")

# Intervention sliders
col1, col2 = st.columns(2)
with col1:
    new_hours = st.slider(
        "Reduce monthly hours to:", 
        80, 300, int(monthly_hours), 
        key="hours_sim"
    )
with col2:
    new_projects = st.slider(
        "Reassign projects to:", 
        1, 10, number_project, 
        key="projects_sim"
    )

# Calculate new risk
sim_input = user_input.copy()
sim_input['monthly_hours'] = new_hours
sim_input['number_project'] = new_projects
sim_df = preprocess_input(sim_input)
new_prob = model.predict_proba(sim_df)[0][1]

# Show impact
st.markdown(f"### Risk changes from **{probability:.0%}** → **{new_prob:.0%}**")
st.progress(new_prob)

# Show improvement
if new_prob < probability:
    improvement = (probability - new_prob) / probability * 100
    st.success(f"✅ **{improvement:.0f}% risk reduction** - Target {new_hours} hours & {new_projects} projects")
else:
    st.warning("⚠️ This change would increase attrition risk")

# ======================
# TECHNICAL EVIDENCE TABS
# ======================
st.divider()
st.header("🔍 Technical Validation")

tab1, tab2 = st.tabs(["Model Insights", "Project Details"])

with tab1:
    st.subheader("Why Trust This Prediction?")
    
    # Confusion matrix
    st.image("confusion_matrix.png", 
             caption="Model optimized for catching at-risk employees (fewer false negatives)")
    
    st.markdown("""
    **Model Performance:**
    - AUC: 0.92 (Excellent discrimination)
    - F1-Score: 0.85 (Balanced precision/recall)
    - Conservative strategy: Prioritizes catching at-risk employees
    """)

with tab2:
    st.subheader("Project Documentation")
    st.markdown("""
    **Business Problem**  
    *“What’s likely to make the employee leave the company?”*
    
    **Methodology**  
    - Analyzed 10,500 employee records
    - Engineered features (overwork indicator, tenure buckets)
    - Trained/tested 4 ML models (Random Forest selected)
    - Hyperparameter tuning via GridSearchCV
    
    **Technical Stack**  
    `Python` `pandas` `scikit-learn` `XGBoost` `SHAP` `Streamlit`
    
    **Business Impact**  
    - Enable proactive retention strategies
    - Target high-risk employees with personalized interventions
    """)
    
    # st.link_button("View Full Project Report", "https://your-report-link.com")
    st.link_button("GitHub Repository", "https://github.com/MutasimBillahArib/google-advanced-data-analytics-coursera/tree/main/employee-turnover-analysis")

# ======================
# FOOTER
# ======================
st.divider()
st.caption("""
💡 **Pro Tip for HR Managers**: Use the "What-If" tool to simulate interventions before implementing them. 
Target employees with >175 monthly hours AND no promotion in 5 years first - they have 78% attrition risk!
""")
