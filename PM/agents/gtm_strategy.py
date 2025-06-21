# agents/gtm_strategy.py
import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    st.error("Google API Key not found. Please set it in your .env file.")
    st.stop()

# Initialize LLM
try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7) # Higher temperature for creative content
except Exception as e:
    st.error(f"Failed to initialize Google Generative AI model. Check your API key and network connection: {e}")
    st.stop()

MOCK_USER_PERSONAS_PATH = "./data/mock_data/user_personas.txt"

def load_mock_user_personas():
    """Loads mock user personas from a text file."""
    if os.path.exists(MOCK_USER_PERSONAS_PATH):
        try:
            with open(MOCK_USER_PERSONAS_PATH, 'r') as f:
                return f.read()
        except Exception as e:
            st.error(f"Error reading mock user personas: {e}")
            return ""
    else:
        st.warning(f"Mock user personas file not found at: {MOCK_USER_PERSONAS_PATH}")
        return ""

def gtm_strategy_ui():
    """
    Streamlit UI and logic for the Go-To-Market (GTM) Strategy Agent.
    Plans and executes launch campaigns.
    """
    st.header("Plan Product Launch Campaigns")
    st.write("This agent helps plan and execute effective Go-To-Market campaigns for your product launch.")
    st.info("Input product features, user personas, and your launch timeline to generate marketing content and a launch plan.")

    # Input fields
    product_feature_list = st.text_area(
        "Key Product Features (one per line or comma-separated):",
        height=150,
        value="AI-powered analytics dashboard, Real-time data streaming, Predictive churn modeling, Customizable reports, Salesforce integration"
    )

    user_personas = st.text_area(
        "Target User Personas (describe your ideal users):",
        height=250,
        value=load_mock_user_personas()
    )

    launch_timeline = st.date_input(
        "Target Launch Date:",
        value=datetime.now().date() + timedelta(days=60), # Default to 2 months from now
        min_value=datetime.now().date()
    )

    campaign_focus = st.multiselect(
        "Select Campaign Content Types to Generate:",
        options=["Marketing Email", "Social Media Post", "Press Release Headline", "Blog Post Idea", "Ad Copy"],
        default=["Marketing Email", "Social Media Post"]
    )

    if st.button("Generate Launch Campaign Kit", use_container_width=True, type="primary"):
        if not product_feature_list or not user_personas or not campaign_focus:
            st.warning("Please provide product features, user personas, and select at least one content type.")
            return

        with st.spinner("Generating campaign content and launch plan..."):
            # Prompt for overall launch plan (high-level)
            launch_plan_prompt = PromptTemplate(
                input_variables=["features", "personas", "launch_date"],
                template="""You are an expert Marketing Strategist.
                Based on the following product features, target user personas, and launch date,
                propose a high-level Go-To-Market launch plan.
                Include key phases, suggested channels, and primary messaging themes.

                Product Features: {features}
                Target User Personas: {personas}
                Target Launch Date: {launch_date}

                High-Level GTM Launch Plan:
                """
            )
            launch_plan_chain = LLMChain(llm=llm, prompt=launch_plan_prompt)
            try:
                launch_plan_output = launch_plan_chain.run(
                    features=product_feature_list,
                    personas=user_personas,
                    launch_date=launch_timeline.strftime("%Y-%m-%d")
                )
                st.subheader("📋 Proposed High-Level Launch Plan:")
                st.markdown(launch_plan_output)
            except Exception as e:
                st.error(f"Error generating launch plan: {e}")

            st.markdown("---")
            st.subheader("✍️ Generated Campaign Content:")

            # Generate specific content based on selected types
            for content_type in campaign_focus:
                with st.expander(f"Generate {content_type}"):
                    if content_type == "Marketing Email":
                        email_prompt = PromptTemplate(
                            input_variables=["features", "personas", "company_name"],
                            template="""Write a concise marketing email for a new SaaS product launch.
                            Focus on the value proposition of the following features for the described user personas.
                            Make it engaging and include a clear Call-to-Action.
                            Company Name: OurSaaSPlatform

                            Product Features: {features}
                            Target User Personas: {personas}

                            Marketing Email:
                            """
                        )
                        email_chain = LLMChain(llm=llm, prompt=email_prompt)
                        email_output = email_chain.run(
                            features=product_feature_list,
                            personas=user_personas,
                            company_name="OurSaaSPlatform"
                        )
                        st.text_area(f"Marketing Email Draft:", email_output, height=300)

                    elif content_type == "Social Media Post":
                        social_prompt = PromptTemplate(
                            input_variables=["features", "personas"],
                            template="""Write a short and catchy social media post (e.g., for LinkedIn/Twitter)
                            announcing a new SaaS product launch. Include relevant hashtags.

                            Product Features: {features}
                            Target User Personas: {personas}

                            Social Media Post:
                            """
                        )
                        social_chain = LLMChain(llm=llm, prompt=social_prompt)
                        social_output = social_chain.run(
                            features=product_feature_list,
                            personas=user_personas
                        )
                        st.text_area(f"Social Media Post Draft:", social_output, height=150)

                    elif content_type == "Press Release Headline":
                        pr_headline_prompt = PromptTemplate(
                            input_variables=["features", "personas"],
                            template="""Generate 3 compelling and attention-grabbing press release headlines
                            for a new SaaS product launch with the following features and targeting these personas.

                            Product Features: {features}
                            Target User Personas: {personas}

                            Press Release Headlines:
                            """
                        )
                        pr_headline_chain = LLMChain(llm=llm, prompt=pr_headline_prompt)
                        pr_headline_output = pr_headline_chain.run(
                            features=product_feature_list,
                            personas=user_personas
                        )
                        st.text_area(f"Press Release Headlines:", pr_headline_output, height=150)

                    elif content_type == "Blog Post Idea":
                        blog_idea_prompt = PromptTemplate(
                            input_variables=["features", "personas"],
                            template="""Suggest 3 engaging blog post ideas for a new SaaS product launch,
                            focusing on how the following features benefit the described user personas.

                            Product Features: {features}
                            Target User Personas: {personas}

                            Blog Post Ideas:
                            """
                        )
                        blog_idea_chain = LLMChain(llm=llm, prompt=blog_idea_prompt)
                        blog_idea_output = blog_idea_chain.run(
                            features=product_feature_list,
                            personas=user_personas
                        )
                        st.text_area(f"Blog Post Ideas:", blog_idea_output, height=150)

                    elif content_type == "Ad Copy":
                        ad_copy_prompt = PromptTemplate(
                            input_variables=["features", "personas"],
                            template="""Write 3 short and impactful ad copies (e.g., for Google Ads or social media ads)
                            for a new SaaS product launch, highlighting the key benefits for the described user personas.

                            Product Features: {features}
                            Target User Personas: {personas}

                            Ad Copies:
                            """
                        )
                        ad_copy_chain = LLMChain(llm=llm, prompt=ad_copy_prompt)
                        ad_copy_output = ad_copy_chain.run(
                            features=product_feature_list,
                            personas=user_personas
                        )
                        st.text_area(f"Ad Copy Suggestions:", ad_copy_output, height=150)

            st.success("Launch campaign kit generated successfully!")

            st.subheader("Next Steps:")
            st.markdown("""
            -   Review and refine the generated content.
            -   (Conceptual) Integrate with distribution platforms (Mailchimp, Hootsuite).
            -   (Conceptual) Send feedback and launch impact to the **Sales & Feedback Agent**.
            """)