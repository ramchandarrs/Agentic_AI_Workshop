import streamlit as st
import os
from agents import create_medical_verifier_crew, registry_rag_utility 
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

import tempfile
import json 
from crewai import Crew, Process 

st.set_page_config(page_title="Medical Credential Verifier", layout="wide")

st.markdown(
    """
    <style>
    .main-header {
        font-size: 3em;
        color: #2F80ED;
        text-align: center;
        margin-bottom: 0.5em;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        border-bottom: 2px solid #2F80ED;
        padding-bottom: 10px;
    }
    .sub-header {
        font-size: 1.5em;
        color: #555;
        text-align: center;
        margin-bottom: 1.5em;
    }
    .stButton>button {
        background-color: #2F80ED;
        color: white;
        border-radius: 12px;
        padding: 10px 20px;
        font-size: 1.1em;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 3px 3px 6px rgba(0,0,0,0.2);
    }
    .stButton>button:hover {
        background-color: #1A5FB4;
        transform: translateY(-2px);
        box-shadow: 5px 5px 10px rgba(0,0,0,0.3);
    }
    .stProgress > div > div > div > div {
        background-color: #2F80ED !important;
    }
    .stFileUploader label {
        font-weight: bold;
        color: #333;
        font-size: 1.1em;
    }
    .result-container {
        border: 2px solid #E0E0E0;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        background-color: #F9F9F9; /* Light background */
        box-shadow: inset 0px 0px 5px rgba(0,0,0,0.05);
        color: #333; /* Set a dark text color for the container */
    }
    .result-container h3 {
        color: #2F80ED;
        font-size: 1.8em;
        margin-bottom: 15px;
    }
    /* Add styling for the inner div where the result text is displayed */
    .result-container div {
        font-family: monospace;
        white-space: pre-wrap;
        background-color: #e6e6e6;
        padding: 15px;
        border-radius: 8px;
        color: #000000; /* Explicitly set text color to black for readability */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="main-header">Medical Credential Verifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Leveraging AI to ensure patient safety and compliance</p>', unsafe_allow_html=True)

st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader(
    "Upload a Doctor's Resume, Certificate, or Professional Profile (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"]
)

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_file_path = tmp_file.name

    document_text = ""
    try:
        if uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(temp_file_path)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            loader = Docx2txtLoader(temp_file_path)
        elif uploaded_file.type == "text/plain":
            loader = TextLoader(temp_file_path)
        else:
            st.error("Unsupported file type.")
            os.unlink(temp_file_path)
            st.stop()

        # Load and extract text from the document
        docs = loader.load()
        document_text = "\n".join([doc.page_content for doc in docs])

        st.sidebar.success(f"File '{uploaded_file.name}' uploaded and processed successfully!")
        
        st.info("Document content extracted. Ready to verify!")
        
        # Display extracted text in an expander for debugging/review
        with st.expander("Extracted Document Text"):
            st.text_area("Content", document_text, height=300)

        if st.button("Start Verification"):
            with st.spinner("Verification in progress..."):
                try:
                    # Step 1: Run the Credential Extraction Task first to get data for RAG query
                    # Create a temporary crew to access the first agent and task
                    temp_crew_for_extraction_setup = create_medical_verifier_crew(document_text) 

                    extraction_crew = Crew(
                        agents=[temp_crew_for_extraction_setup.agents[0]], # Get the extraction agent
                        tasks=[temp_crew_for_extraction_setup.tasks[0]], # Get the extraction task
                        process=Process.sequential,
                        verbose=0, # Less verbose for this internal step
                        full_output=True # Essential to get the CrewOutput object
                    )
                    st.info("Agent 1: Extracting credentials...")
                    
                    # kickoff returns a CrewOutput object
                    extraction_crew_output = extraction_crew.kickoff()
                    
                    # Access the raw string output of the extraction task
                    extracted_credentials_str = extraction_crew_output.raw 
                    
                    st.markdown("Extracted Credentials:")
                    # Attempt to display as JSON, fallback to text if not valid JSON
                    try:
                        st.json(json.loads(extracted_credentials_str))
                    except json.JSONDecodeError:
                        st.warning("Extracted credentials are not valid JSON. Displaying as plain text.")
                        st.text(extracted_credentials_str)


                    # Attempt to parse the extracted credentials to formulate a better RAG query
                    rag_query_terms = []
                    try:
                        parsed_extracted_credentials = json.loads(extracted_credentials_str)
                        if isinstance(parsed_extracted_credentials, list):
                            for item in parsed_extracted_credentials:
                                if isinstance(item, dict):
                                    if "License Number" in item:
                                        rag_query_terms.append(f"License Number {item['License Number']}")
                                    if "Name" in item:
                                        rag_query_terms.append(f"Name: {item['Name']}")
                                    if "Issuing Body" in item:
                                        rag_query_terms.append(f"Issuing Body: {item['Issuing Body']}")
                        elif isinstance(parsed_extracted_credentials, dict):
                             if "License Number" in parsed_extracted_credentials:
                                 rag_query_terms.append(f"License Number {parsed_extracted_credentials['License Number']}")
                             if "Name" in parsed_extracted_credentials:
                                 rag_query_terms.append(f"Name: {parsed_extracted_credentials['Name']}")
                             if "Issuing Body" in parsed_extracted_credentials:
                                 rag_query_terms.append(f"Issuing Body: {parsed_extracted_credentials['Issuing Body']}")
                        
                    except json.JSONDecodeError:
                        st.warning("Could not parse extracted credentials as JSON directly for RAG query. Trying keyword extraction.")
                        if "License Number:" in extracted_credentials_str:
                            for line in extracted_credentials_str.split('\n'):
                                if "License Number:" in line:
                                    rag_query_terms.append(line.replace("License Number:", "").strip())
                                if "Name:" in line:
                                    rag_query_terms.append(line.replace("Name:", "").strip())
                                if "Issuing Body:" in line:
                                    rag_query_terms.append(line.replace("Issuing Body:", "").strip())

                    if not rag_query_terms:
                        rag_query_terms.append(extracted_credentials_str) 
                        st.warning("No specific license numbers or names found for RAG query. Using full extracted text.")
                        
                    rag_query = " ".join(rag_query_terms)
                    st.info(f"RAG Lookup: Searching registry for: '{rag_query}'")
                    retrieved_rag_info = registry_rag_utility.search_registry(rag_query)
                    st.markdown("Retrieved Registry Information (for context):")
                    st.text(retrieved_rag_info)

                    # Now, create and run the full CrewAI crew with the RAG context
                    st.info("Agent 2-4: Verifying and scoring credentials...")
                    crew = create_medical_verifier_crew(document_text, rag_context_for_verification=retrieved_rag_info)
                    
                    # The kickoff will return a CrewOutput object for the main crew
                    main_crew_output = crew.kickoff()
                    
                    # Access the final output string of the entire main crew
                    final_verification_result = main_crew_output.raw 
                    
                    st.success("Verification Complete!")
                    st.markdown(
                        f"""
                        <div class="result-container">
                            <h3>Verification Results</h3>
                            <div style="font-family: monospace; white-space: pre-wrap; background-color: #e6e6e6; padding: 15px; border-radius: 8px;">{final_verification_result}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(f"An error occurred during verification: {e}")
                    st.exception(e) # Show full traceback for debugging

    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")
        st.exception(e)
    finally:
        # Clean up the temporary file
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

else:
    st.info("Please upload a document to begin the credential verification process.")