# agents/sales_feedback.py
import streamlit as st
import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    st.error("Google API Key not found. Please set it in your .env file.")
    st.stop()

# Initialize LLM
try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)
except Exception as e:
    st.error(f"Failed to initialize Google Generative AI model. Check your API key and network connection: {e}")
    st.stop()

MOCK_SALES_DATA_PATH = "./data/mock_data/sales_data.json"
MOCK_USER_FEEDBACK_PATH = "./data/mock_data/user_feedback.json"

def load_mock_sales_data():
    """Loads mock sales and product usage data from a JSON file."""
    if os.path.exists(MOCK_SALES_DATA_PATH):
        try:
            with open(MOCK_SALES_DATA_PATH, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            st.error(f"Error decoding mock sales data JSON: {e}")
            return None
    else:
        st.warning(f"Mock sales data file not found at: {MOCK_SALES_DATA_PATH}")
        return None

def load_mock_user_feedback():
    """Loads mock user feedback data from a JSON file."""
    if os.path.exists(MOCK_USER_FEEDBACK_PATH):
        try:
            with open(MOCK_USER_FEEDBACK_PATH, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            st.error(f"Error decoding mock user feedback JSON: {e}")
            return None
    else:
        st.warning(f"Mock user feedback file not found at: {MOCK_USER_FEEDBACK_PATH}")
        return None

def sales_feedback_ui():
    """
    Streamlit UI and logic for the Sales & Feedback Agent.
    Monitors product usage, sales, and collects feedback.
    """
    st.header("Monitor Product Performance & Collect Feedback")
    st.write("This agent monitors product usage, sales trends, and analyzes user feedback to provide actionable insights for future iterations.")
    st.info("Currently using simulated sales and feedback data. In a real scenario, this would integrate with CRM, analytics, and feedback APIs.")

    sales_data = load_mock_sales_data()
    user_feedback = load_mock_user_feedback()

    if not sales_data or not user_feedback:
        st.error("Cannot proceed without mock sales data and user feedback. Please ensure `data/mock_data/sales_data.json` and `data/mock_data/user_feedback.json` exist and are valid.")
        return

    st.subheader("📈 Product Usage & Sales Trends")
    usage_metrics = sales_data.get("product_usage_metrics", {})
    sales_metrics = sales_data.get("sales_data", {})

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Active Users", usage_metrics.get("total_active_users", "N/A"))
    col2.metric("Monthly Recurring Revenue (MRR)", f"${sales_metrics.get('mrr_current_month', 0):,.0f}")
    col3.metric("Churn Rate", sales_metrics.get("churn_rate_current_month", "N/A"))

    st.markdown("---")

    st.subheader("📊 Detailed Usage & Drop-off Points")
    st.json(usage_metrics, expanded=False) # Show detailed usage metrics
    
    st.subheader("📉 Sales Metrics Overview")
    st.json(sales_metrics, expanded=False) # Show detailed sales metrics


    st.markdown("---")
    st.subheader("🗣️ User Feedback Analysis")

    feedback_text = "\n---\n".join([f"Source: {f.get('source', 'N/A')}, Rating: {f.get('rating', 'N/A')}, Text: {f.get('text', '')}" for f in user_feedback])

    if st.button("Analyze Feedback & Generate Insights", use_container_width=True, type="primary"):
        with st.spinner("Analyzing user feedback and usage patterns for insights..."):
            analysis_prompt_template = PromptTemplate(
                input_variables=["usage_data", "feedback_data"],
                template="""You are a Product Manager analyzing user feedback and product usage data.
                Your goal is to identify key user pain points, common requests, and usage drop-off reasons.
                Based on this analysis, provide actionable insights and suggest specific feature improvements or new features for the product roadmap.
                Map insights to potential features.

                Product Usage Data:
                {usage_data}

                User Feedback Data:
                {feedback_data}

                Analysis & Feature Suggestions (Structured as: Insight -> Suggested Feature -> Why it matters):
                """
            )

            analysis_chain = LLMChain(llm=llm, prompt=analysis_prompt_template)

            try:
                usage_data_str = json.dumps(usage_metrics, indent=2) + "\n" + json.dumps(sales_metrics, indent=2)
                
                analysis_output = analysis_chain.run(
                    usage_data=usage_data_str,
                    feedback_data=feedback_text
                )

                st.subheader("💡 Actionable Insights & Feature Suggestions:")
                st.markdown(analysis_output)
                st.success("Analysis complete! Insights generated.")

                st.subheader("🔄 Closing the Feedback Loop")
                st.write("These insights are crucial for product evolution. In an integrated system, this agent would:")
                st.markdown("""
                -   **Automatically send these feature suggestions** to the **Roadmap Planning Agent**. 
                -   **Update a backlog** (e.g., in Jira/Trello) with new user stories derived from feedback.
                -   **Monitor A/B test results** for new iterations.
                """)
                st.info("For this demonstration, imagine these insights have been forwarded to the Roadmap Planning team!")


            except Exception as e:
                st.error(f"An error occurred during feedback analysis: {e}")
                st.warning("Ensure your GOOGLE_API_KEY is correct and you have an active internet connection.")