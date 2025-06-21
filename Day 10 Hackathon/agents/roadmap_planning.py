# agents/roadmap_planning.py
import streamlit as st
import os
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
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7) # Slightly higher temp for creativity
except Exception as e:
    st.error(f"Failed to initialize Google Generative AI model. Check your API key and network connection: {e}")
    st.stop()

def roadmap_planning_ui():
    """
    Streamlit UI and logic for the Roadmap Planning Agent.
    Generates and updates the product roadmap.
    """
    st.header("Generate Product Roadmap")
    st.write("Input market insights, feature requests, and team capacity to generate a draft roadmap.")

    # Inputs for Roadmap Agent
    market_research_data = st.text_area(
        "Market Research & User Feedback (e.g., top pain points, competitor analysis):",
        height=250,
        value=open("./data/mock_data/market_research.txt", "r").read() if os.path.exists("./data/mock_data/market_research.txt") else ""
    )

    feature_requests = st.text_area(
        "New Feature Requests & Ideas (one per line):",
        height=150,
        value=open("./data/mock_data/feature_requests.txt", "r").read() if os.path.exists("./data/mock_data/feature_requests.txt") else ""
    )

    team_capacity = st.slider(
        "Estimated Team Capacity (in story points or person-weeks per quarter):",
        min_value=10, max_value=500, value=150, step=10,
        help="Estimate the team's capacity for development over the next 3 months."
    )

    roadmap_duration = st.selectbox(
        "Roadmap Duration:",
        options=["3 Months", "6 Months", "12 Months"],
        index=0
    )

    if st.button("Generate Draft Roadmap", use_container_width=True, type="primary"):
        if not market_research_data and not feature_requests:
            st.warning("Please provide either market research data or feature requests to generate a roadmap.")
            return

        with st.spinner("Generating product roadmap... This may take a moment."):
            roadmap_prompt_template = PromptTemplate(
                input_variables=["market_research", "feature_requests", "team_capacity", "roadmap_duration"],
                template="""You are an experienced Product Manager leading a SaaS startup.
                Your task is to generate a {roadmap_duration} product roadmap for a new SaaS product,
                focusing on the most impactful features based on the provided inputs.
                Prioritize features using RICE/ICE scoring principles implicitly (Reach, Impact, Confidence, Effort).
                The output should be in a Kanban-like format, clearly showing features, their priority, estimated timeline, and a brief description.
                Consider the team's estimated capacity when suggesting timelines.

                Market Research & User Feedback:
                {market_research}

                Feature Requests:
                {feature_requests}

                Estimated Team Capacity (e.g., story points/person-weeks): {team_capacity}

                Roadmap Format (use Markdown tables):
                ### Product Roadmap - {roadmap_duration}

                | Feature Name | Priority (High/Medium/Low) | Estimated Timeline | Key Value Proposition / Description |
                |---|---|---|---|
                | Feature A | High | Month 1-2 | Addresses critical pain point X, expected to increase user retention by Y%. |
                | Feature B | Medium | Month 2 | Enhances user experience, adds Z integration. |
                | ... | ... | ... | ... |

                Goals for this roadmap: [Summarize 2-3 high-level goals based on inputs]
                Assumptions: [List 2-3 key assumptions]
                Risks: [List 1-2 potential risks]
                """
            )

            roadmap_chain = LLMChain(llm=llm, prompt=roadmap_prompt_template)

            try:
                roadmap_output = roadmap_chain.run(
                    market_research=market_research_data,
                    feature_requests=feature_requests,
                    team_capacity=team_capacity,
                    roadmap_duration=roadmap_duration
                )
                st.subheader("🗺️ Draft Product Roadmap:")
                st.markdown(roadmap_output) # Render markdown output directly
                st.success("Roadmap generated successfully!")

                st.subheader("Next Steps:")
                st.markdown("""
                -   Review and refine the generated roadmap.
                -   Collaborate with engineering and design for feasibility.
                -   (Conceptual) Send the roadmap to the **Progress Monitoring Agent** to start tracking.
                """)

            except Exception as e:
                st.error(f"An error occurred while generating the roadmap: {e}")
                st.warning("Ensure your GOOGLE_API_KEY is correct and you have an active internet connection. Also, try simplifying the input.")