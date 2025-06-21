# agents/talent_acquisition.py (UPDATED)
import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import tempfile # For safer temporary file handling

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    st.error("Google API Key not found. Please set it in your .env file.")
    st.stop()

# Initialize LLM and Embeddings outside the function to avoid re-initialization on every rerun
try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
except Exception as e:
    st.error(f"Failed to initialize Google Generative AI models. Check your API key and network connection: {e}")
    st.stop()

# Define the path for temporary resume storage
# Using tempfile.mkdtemp() for robust temporary directory creation
# This will create a unique directory each session, safer than a fixed path
# but means you can't manually inspect easily unless you print the path.
# For hackathon, keeping it simple with a fixed sub-directory might be fine,
# but a truly robust solution would clean up temp files.
RESUME_UPLOAD_DIR = "./data/resumes_temp" # Using a different name to distinguish
os.makedirs(RESUME_UPLOAD_DIR, exist_ok=True)


# --- Candidate Evaluation Function ---
def evaluate_candidate_fit(job_description, candidate_text):
    """
    Uses an LLM to evaluate how well a candidate's text matches a job description.
    Returns a score and a brief reasoning.
    """
    evaluation_prompt = PromptTemplate(
        input_variables=["job_description", "candidate_text"],
        template="""You are an expert HR recruiter.
        Given the following Job Description and a candidate's resume snippet,
        evaluate the candidate's fit on a scale of 1 to 10 (10 being perfect fit).
        Provide a concise reason for the score, focusing on key matches or gaps.

        Job Description:
        {job_description}

        Candidate Resume Snippet:
        {candidate_text}

        Evaluation (Score: X/10, Reason: [Your concise reasoning]):
        """
    )
    evaluation_chain = LLMChain(llm=llm, prompt=evaluation_prompt)

    try:
        response = evaluation_chain.run(job_description=job_description, candidate_text=candidate_text)
        # Parse the score and reason from the response
        import re
        score_match = re.search(r"Score:\s*(\d+)/10", response)
        reason_match = re.search(r"Reason:\s*(.*)", response)

        score = int(score_match.group(1)) if score_match else 0
        reason = reason_match.group(1).strip() if reason_match else "N/A"
        return score, reason
    except Exception as e:
        st.warning(f"Could not evaluate candidate fit using LLM: {e}. Returning default score.")
        return 0, f"Error in LLM evaluation: {e}"


def talent_acquisition_ui():
    """
    Streamlit UI and logic for the Talent Acquisition Agent.
    Automates hiring tasks like resume screening, filtering, and interview scheduling.
    """
    st.header("Upload Job Description & Resumes")

    job_description = st.text_area("Enter Job Description:", height=200, help="Provide a detailed job description including responsibilities, required skills, and qualifications.")
    uploaded_resumes = st.file_uploader(
        "Upload Resumes (PDFs):",
        type="pdf",
        accept_multiple_files=True,
        help="Upload multiple PDF resumes for screening. The system will extract text and rank candidates."
    )

    candidate_data = [] # To store structured data for display

    if st.button("Analyze Candidates", use_container_width=True, type="primary"):
        if not job_description:
            st.warning("Please provide a Job Description to start the analysis.")
            return
        if not uploaded_resumes:
            st.warning("Please upload at least one resume (PDF) to analyze.")
            return

        with st.spinner("Processing resumes and evaluating candidates... This may take a few moments."):
            all_splits = []
            resume_metadata = {} # To map chunk sources back to original resume names

            for uploaded_file in uploaded_resumes:
                # Save the uploaded file temporarily in a unique directory for this session or a fixed one
                temp_file_path = os.path.join(RESUME_UPLOAD_DIR, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.write(f"📝 Processing: {uploaded_file.name}")

                try:
                    loader = PyPDFLoader(temp_file_path)
                    docs = loader.load()
                    # Add original file name to metadata for tracking
                    for doc in docs:
                        doc.metadata["original_filename"] = uploaded_file.name

                    # Split documents into chunks for embedding
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200) # Increased chunk size
                    splits = text_splitter.split_documents(docs)
                    all_splits.extend(splits)

                    # Store for later retrieval of full resume content if needed
                    # For a hackathon, simplicity, we're just extracting text
                    # For a real app, you might save processed data or a link to it
                    resume_metadata[uploaded_file.name] = "\n".join([d.page_content for d in docs])

                except Exception as e:
                    st.error(f"🚫 Error processing {uploaded_file.name}: {e}. Skipping this resume.")
                    # Clean up problematic file
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                    continue

            if not all_splits:
                st.error("No valid resume content could be extracted. Please ensure PDFs are text-searchable.")
                return

            st.success(f"Extracted text from {len(uploaded_resumes)} resume(s). Splitting into {len(all_splits)} chunks for analysis.")

            # Create Vector Store
            try:
                vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)
                st.success("Vector store created and populated with resume embeddings.")
            except Exception as e:
                st.error(f"Error creating vector store: {e}. Please try again.")
                return

            # Perform Semantic Search for relevant chunks
            st.subheader("🔎 Identifying Key Resume Sections...")
            # We retrieve a higher number of chunks to ensure we get relevant info from different candidates
            relevant_chunks = vectorstore.similarity_search(job_description, k=10) # Get top 10 most relevant chunks

            # --- Evaluate and Rank Candidates ---
            st.subheader("🧠 Evaluating Candidate Fit with AI...")
            evaluated_candidates = {} # {resume_filename: {"score": score, "reason": "...", "text_matches": []}}

            for i, chunk in enumerate(relevant_chunks):
                filename = chunk.metadata.get("original_filename", "Unknown Resume")
                page_content = chunk.page_content # The relevant text snippet

                # Use LLM to evaluate this chunk against the JD
                score, reason = evaluate_candidate_fit(job_description, page_content)

                if filename not in evaluated_candidates:
                    evaluated_candidates[filename] = {"total_score": 0, "match_count": 0, "reasons": [], "text_matches": []}

                evaluated_candidates[filename]["total_score"] += score
                evaluated_candidates[filename]["match_count"] += 1
                evaluated_candidates[filename]["reasons"].append(f"Score {score}/10: {reason} (from page {chunk.metadata.get('page', 'N/A')})")
                evaluated_candidates[filename]["text_matches"].append(page_content)


            # Calculate average score and prepare for display
            final_ranked_candidates = []
            for filename, data in evaluated_candidates.items():
                avg_score = data["total_score"] / data["match_count"] if data["match_count"] > 0 else 0
                final_ranked_candidates.append({
                    "Resume": filename,
                    "Average Fit Score": round(avg_score, 2),
                    "Key Reasons": " | ".join(data["reasons"][:3]) + ("..." if len(data["reasons"]) > 3 else ""), # Show top 3 reasons
                    "Matched Snippets": data["text_matches"]
                })

            # Sort candidates by average score in descending order
            final_ranked_candidates.sort(key=lambda x: x["Average Fit Score"], reverse=True)

            st.subheader("✅ Candidate Ranking Results:")
            if final_ranked_candidates:
                # Display in a table
                st.dataframe([
                    {"Resume": c["Resume"], "Average Fit Score": c["Average Fit Score"], "Key Reasons": c["Key Reasons"]}
                    for c in final_ranked_candidates
                ], use_container_width=True)

                st.markdown("---")
                st.subheader("Detailed Candidate Insights:")
                for candidate in final_ranked_candidates:
                    expander_title = f"{candidate['Resume']} (Score: {candidate['Average Fit Score']}/10)"
                    with st.expander(expander_title):
                        st.write(f"**Average Fit Score:** {candidate['Average Fit Score']}/10")
                        st.write("**Key Reasons/Matches:**")
                        for reason_detail in candidate["reasons"]:
                            st.markdown(f"- {reason_detail}")
                        st.write("**Relevant Text Snippets from Resume:**")
                        for snippet in candidate["Matched Snippets"]:
                            st.code(snippet[:500] + "...", language="text") # Show a preview
                            st.write("---")

                st.success("Candidate analysis complete!")
                st.write("---")

                # --- Simulated Interview Scheduling ---
                st.subheader("📅 Simulated Interview Scheduling")
                st.write("This agent would typically integrate with a Calendar API (e.g., Google Calendar) and HRMS to:")
                st.markdown("""
                -   **Auto-schedule interviews** with top candidates based on availability.
                -   **Send invitations** to candidates and interviewers.
                -   **Update candidate status** in an HR Management System.
                """)
                st.info("For this demonstration, imagine interview slots for the top 3 candidates are now reserved!")

            else:
                st.warning("No candidates were ranked. Please check your job description and resume content.")