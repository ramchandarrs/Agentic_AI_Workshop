# agents/progress_monitoring.py
import streamlit as st
import os
import json
from datetime import datetime
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

MOCK_SPRINT_DATA_PATH = "./data/mock_data/sprint_data.json"

def load_mock_sprint_data():
    """Loads mock sprint data from a JSON file."""
    if os.path.exists(MOCK_SPRINT_DATA_PATH):
        try:
            with open(MOCK_SPRINT_DATA_PATH, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            st.error(f"Error decoding mock sprint data JSON: {e}")
            return None
    else:
        st.warning(f"Mock sprint data file not found at: {MOCK_SPRINT_DATA_PATH}")
        return None

def progress_monitoring_ui():
    """
    Streamlit UI and logic for the Progress Monitoring Agent.
    Tracks execution, identifies blockers, and communicates progress.
    """
    st.header("Monitor Product Development Progress")
    st.write("This agent tracks sprint execution, identifies blockers, and generates progress reports.")
    st.info("Currently using simulated sprint data. In a real scenario, this would integrate with Jira/Trello APIs.")

    sprint_data = load_mock_sprint_data()

    if not sprint_data:
        st.error("Cannot proceed without mock sprint data. Please ensure `data/mock_data/sprint_data.json` exists and is valid.")
        return

    current_sprint = sprint_data.get("current_sprint", {})
    if not current_sprint:
        st.error("No 'current_sprint' data found in the mock file.")
        return

    st.subheader(f"Current Sprint: {current_sprint.get('sprint_name', 'N/A')}")
    st.markdown(f"**Dates:** {current_sprint.get('start_date', 'N/A')} to {current_sprint.get('end_date', 'N/A')}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Story Points", current_sprint.get("total_story_points", 0))
    col2.metric("Completed Story Points", current_sprint.get("completed_story_points", 0))
    col3.metric("Remaining Story Points", current_sprint.get("remaining_story_points", 0))

    st.markdown("---")
    st.subheader("Tasks Status Overview")

    tasks = current_sprint.get("tasks", [])
    if tasks:
        # Create a simple DataFrame for display
        task_df = st.dataframe(tasks, use_container_width=True)

        overdue_tasks = [task for task in tasks if task.get("status", "").lower() == "overdue"]
        blocked_tasks = [task for task in tasks if task.get("status", "").lower() == "blocked"]

        if overdue_tasks:
            st.warning("⚠️ **Overdue Tasks:**")
            for task in overdue_tasks:
                st.markdown(f"- **{task['name']}** (Assigned to: {task.get('assignee', 'N/A')})")
        if blocked_tasks:
            st.error("🛑 **Blocked Tasks:**")
            for task in blocked_tasks:
                st.markdown(f"- **{task['name']}** (Blocker: {task.get('blocker', 'N/A')}, Assigned to: {task.get('assignee', 'N/A')})")
    else:
        st.info("No tasks found in current sprint data.")

    st.markdown("---")
    st.subheader("Generate Weekly Progress Report")

    if st.button("Generate Report", use_container_width=True, type="primary"):
        with st.spinner("Generating weekly summary report..."):
            report_prompt_template = PromptTemplate(
                input_variables=["sprint_data", "current_date"],
                template="""You are a Product Manager generating a weekly progress report for stakeholders.
                Based on the following sprint and task data, summarize key achievements, identify current blockers,
                highlight any overdue tasks, and provide an overall assessment of progress and risks.
                The report should be concise, professional, and actionable.
                Assume today's date is {current_date}.

                Sprint Data:
                {sprint_data}

                Weekly Progress Report:
                """
            )

            report_chain = LLMChain(llm=llm, prompt=report_prompt_template)

            try:
                # Convert sprint_data to a string for the LLM
                sprint_data_str = json.dumps(current_sprint, indent=2)
                current_date = datetime.now().strftime("%Y-%m-%d")

                report_output = report_chain.run(sprint_data=sprint_data_str, current_date=current_date)

                st.subheader("📰 Weekly Progress Report:")
                st.markdown(report_output)
                st.success("Report generated successfully!")

                st.subheader("Next Steps:")
                st.markdown("""
                -   (Conceptual) Send this report via email/Slack to relevant stakeholders.
                -   (Conceptual) Report progress and delay risks to the **Go-To-Market Agent**.
                """)

                # --- Simulated Alerts/Notifications ---
                st.subheader("🔔 Simulated Alerts & Notifications")
                st.write("In a real system, this agent would also:")
                st.markdown("""
                -   **Send real-time alerts to the PM** (e.g., via Slack) for overdue or blocked tasks.
                -   **Notify relevant team members** about task status changes.
                -   **Update project management tools** (Jira/Trello) with summarized progress.
                """)
                st.info("For this demonstration, imagine alerts for critical issues have been sent!")


            except Exception as e:
                st.error(f"An error occurred while generating the report: {e}")
                st.warning("Ensure your GOOGLE_API_KEY is correct and you have an active internet connection. Try regenerating.")