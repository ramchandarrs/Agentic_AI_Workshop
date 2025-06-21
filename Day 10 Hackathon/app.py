# app.py
import streamlit as st
import os
from dotenv import load_dotenv
from agents.talent_acquisition import talent_acquisition_ui
from agents.roadmap_planning import roadmap_planning_ui
from agents.progress_monitoring import progress_monitoring_ui
from agents.gtm_strategy import gtm_strategy_ui
from agents.sales_feedback import sales_feedback_ui

# Load environment variables
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="PM AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Talent Acquisition Agent", "Roadmap Planning Agent", "Progress Monitoring Agent", "GTM Strategy Agent", "Sales & Feedback Agent"])

st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.info(
    "This is an AI-powered assistant designed to automate and assist "
    "Product Managers across the product lifecycle, from hiring to sales tracking. "
)


# --- Main Content Area ---
if page == "Home":
    st.title("Welcome to the Product Lifecycle Management AI Assistant 🚀")
    st.write(
        """
        This intelligent multi-agent system mimics or assists a Product Manager's core responsibilities
        in launching and managing a new SaaS product. 
        """
    )

    st.header("Key Responsibilities Covered:")
    st.markdown("""
    - **Talent Acquisition:** Automates hiring tasks like resume screening and interview scheduling. 
    - **Roadmap Planning:** Generates and updates product roadmaps based on market, user feedback, and internal goals. 
    - **Progress Monitoring:** Tracks execution, identifies blockers, and communicates progress to stakeholders. 
    - **Go-To-Market (GTM) Strategy:** Plans and executes launch campaigns. 
    - **Sales & Feedback:** Monitors product usage, sales, and collects feedback for future iterations. 
    """)

    st.image("https://i.imgur.com/your_agent_flow_diagram.png") # Replace with a real image URL if you have one
    st.caption("Conceptual Agent Flow Diagram (as provided in problem statement) ")


elif page == "Talent Acquisition Agent":
    st.title("🤖 Talent Acquisition Agent")
    st.write("This agent automates hiring tasks, including resume screening, filtering, and interview scheduling. ")
    st.info("Upload Job Description and Resumes to find the best candidates.")
    talent_acquisition_ui()

elif page == "Roadmap Planning Agent":
    st.title("🗓️ Roadmap Planning Agent")
    st.write("This agent generates and updates the product roadmap based on market, user feedback, and internal goals.")
    st.info("Provide market research, feature requests, and team capacity to generate a roadmap.")
    roadmap_planning_ui()

elif page == "Progress Monitoring Agent":
    st.title("📊 Progress Monitoring Agent")
    st.write("This agent tracks execution, identifies blockers, and communicates progress to stakeholders. ")
    st.info("Simulate sprint status and burndown metrics to get progress reports.")
    progress_monitoring_ui()

elif page == "GTM Strategy Agent":
    st.title("🚀 Go-To-Market (GTM) Strategy Agent")
    st.write("This agent plans and executes launch campaigns for the product. ")
    st.info("Input product features, user personas, and launch timeline to generate a campaign kit.")
    gtm_strategy_ui()

elif page == "Sales & Feedback Agent":
    st.title("📈 Sales & Feedback Agent")
    st.write("This agent monitors product usage, sales, and collects feedback for future iterations. ")
    st.info("Simulate CRM data, user reviews, and product analytics to get sales trends and feature suggestions.")
    sales_feedback_ui()