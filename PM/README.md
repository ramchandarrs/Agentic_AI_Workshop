# Product Lifecycle Management AI Assistant

## 🚀 Project Overview

This project implements an **End-to-End Agentic AI Workflow for Product Lifecycle Management (PLM)**, designed to automate and assist the core responsibilities of a Product Manager. [cite_start]From team hiring to product development oversight, Go-To-Market (GTM) strategy, and sales tracking, this system aims to simulate or support a PM's tasks in real-world scenarios.

[cite_start]Built for a hackathon, this multi-agent system mimics or assists the lifecycle of launching a new SaaS product, where the Product Manager is responsible for building the right team, managing the product roadmap, tracking progress, communicating with stakeholders, launching the product, and analyzing post-launch performance and revenue.

## 🧠 Agentic AI Workflow

The system is composed of five specialized AI agents, each handling a critical phase of the product lifecycle:

1.  [cite_start]**Talent Acquisition Agent:** Automates hiring tasks like resume screening, filtering, and interview scheduling.
2.  [cite_start]**Roadmap Planning Agent:** Generates and updates the product roadmap based on market insights, user feedback, and internal goals.
3.  [cite_start]**Progress Monitoring Agent:** Tracks execution, identifies blockers, and communicates progress to stakeholders.
4.  [cite_start]**Go-To-Market (GTM) Strategy Agent:** Plans and executes launch campaigns for the product.
5.  [cite_start]**Sales & Feedback Agent:** Monitors product usage, sales, and collects feedback for future iterations, completing the feedback loop.

### Agent Flow Diagram

[cite_start]The agents operate in a connected flow, with the Sales & Feedback Agent providing crucial insights back to the Roadmap Planning Agent, thereby closing the feedback loop for continuous product improvement.

![Agent Flow Diagram](https://i.imgur.com/your_agent_flow_diagram.png)
*(Replace this with an actual image of the agent flow diagram if you have one, otherwise, this is a conceptual placeholder.)*

## ✨ Key Features

* [cite_start]**Intelligent Resume Screening:** Utilizes Gemini embeddings and semantic similarity to rank candidates against a job description.
* [cite_start]**AI-Powered Roadmap Generation:** Leverages LLMs to prioritize features and create structured roadmaps based on diverse inputs.
* [cite_start]**Automated Progress Reporting:** Summarizes sprint data and identifies critical issues like overdue tasks and blockers.
* [cite_start]**Dynamic Marketing Content Creation:** Generates various GTM campaign materials (emails, social posts, press releases) tailored to product features and personas.
* [cite_start]**Data-Driven Insights from Feedback:** Analyzes simulated sales data, product usage, and user reviews to suggest actionable feature improvements.
* [cite_start]**Simulated Feedback Loop:** Demonstrates how insights from post-launch performance feed directly into future roadmap planning.
* **Interactive Streamlit UI:** A user-friendly web interface for interacting with each AI agent.

## 🛠️ Technologies Used

* **Streamlit:** For rapidly building the interactive web application UI.
* **Google Gemini API:** The core Large Language Model (LLM) powering the intelligence of each agent for tasks like content generation, semantic analysis, and data summarization.
* **Langchain:** A framework used to orchestrate complex LLM workflows, manage prompts, integrate with vector stores, and chain operations.
* **RAG (Retrieval-Augmented Generation):** Employed in the Talent Acquisition Agent for retrieving relevant information from resumes based on job descriptions.
* **ChromaDB:** A lightweight vector database used for storing and querying embeddings of resume data.
* **PyPDFLoader:** For extracting text content from PDF resumes.
* **python-dotenv:** For securely managing API keys and environment variables.

## 🚀 How to Run the Application

Follow these steps to set up and run the PM AI Assistant locally.

### Prerequisites

* Python 3.8+ installed
* An active Google Gemini API Key (get one from [Google AI Studio](https://aistudio.google.com/))

### Setup Steps

1.  **Clone the repository (if applicable) or create the project directory:**
    ```bash
    mkdir your_pm_ai_project
    cd your_pm_ai_project
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows:
    venv\Scripts\activate
    ```

3.  **Create `requirements.txt` and install dependencies:**
    Create a file named `requirements.txt` in the root directory and paste the following content:
    ```
    streamlit
    google-generativeai
    langchain
    langchain-google-genai
    langchain-community
    pypdf
    chromadb
    python-dotenv
    ```
    Then, install them:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your Google Gemini API Key:**
    Create a file named `.env` in the root of your project directory and add your API key:
    ```
    GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
    ```
    **Important:** Replace `"YOUR_GEMINI_API_KEY"` with your actual key. Make sure to add `.env` to your `.gitignore` file if you are using Git to prevent exposing your API key.

5.  **Create the necessary folder structure:**
    ```bash
    mkdir .streamlit
    mkdir data
    mkdir data/resumes
    mkdir data/mock_data
    mkdir agents
    mkdir utils
    touch agents/__init__.py
    touch utils/__init__.py
    ```

6.  **Populate Mock Data:**
    Place the sample data (provided separately, e.g., `sample_job_description.txt`, `sprint_data.json`, `user_feedback.json`, `sales_data.json`, `market_research.txt`, `feature_requests.txt`, `user_personas.txt`, and sample PDF resumes) into the `data/` and `data/mock_data/` directories as instructed.

7.  **Place the Python code:**
    Ensure `app.py` is in the root directory, and the agent files (`talent_acquisition.py`, `roadmap_planning.py`, `progress_monitoring.py`, `gtm_strategy.py`, `sales_feedback.py`) are in the `agents/` directory.

### Running the Application

Once all steps above are complete and your virtual environment is active:

```bash
streamlit run app.py