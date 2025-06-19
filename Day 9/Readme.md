# Medical Credential Verifier (AI-Powered)

## Project Overview

The Medical Credential Verifier is an AI-powered application designed to automate and streamline the process of verifying medical credentials from various documents (resumes, certificates, professional profiles). Leveraging Google's Gemini LLM through CrewAI and a RAG (Retrieval-Augmented Generation) system, it extracts key information, verifies it against a mock medical registry, assesses risks, and provides a comprehensive credibility score and recommendation.

This application aims to enhance patient safety and regulatory compliance by ensuring the authenticity and validity of healthcare professionals' qualifications.

## Features

* **Document Upload:** Supports PDF, DOCX, and TXT file uploads for medical credential documents.
* **AI-Powered Extraction:** Utilizes a specialized CrewAI agent to accurately extract critical information such as:
    * Full Name
    * License Number(s)
    * Issuing Body/Authority
    * Expiration Date(s)
    * Specialty/Area of Practice
    * Any relevant status or remarks.
* **RAG-based Verification:** Integrates a Retrieval-Augmented Generation (RAG) system to search a local mock medical registry for verification data, which is then provided as context to the verification agent.
* **Multi-Agent Workflow:** Orchestrates a sequential workflow of specialized agents (Extraction, Verification, Eligibility/Risk, Credibility Scoring) using CrewAI.
* **Risk Assessment:** Identifies expired licenses, compliance issues, and general inconsistencies.
* **Credibility Scoring:** Calculates a comprehensive credibility score and provides a final recommendation (e.g., "Approved," "Approved with Caution," "Rejected").
* **Streamlit UI:** Provides an intuitive and interactive user interface for document upload and result display.

## Technologies Used

* **Python 3.9+**
* **CrewAI:** For orchestrating the multi-agent workflow.
* **Google Gemini (via LiteLLM provider):** The underlying Large Language Model powering the agents.
* **LangChain:** For document loading, text splitting, and RAG capabilities.
* **FAISS:** For efficient similarity search in the RAG system.
* **Streamlit:** For building the interactive web user interface.
* **python-dotenv:** For managing environment variables.
* **PyPDFLoader, Docx2txtLoader, TextLoader:** For handling various document formats.

## Setup Instructions

Follow these steps to get the project up and running on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/ramchandarrs/Agentic_AI_Workshop.git
cd Day 9
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Streamlit

```bash
streamlit run app.py
```
