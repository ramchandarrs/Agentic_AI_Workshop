import os
from crewai import Agent, Task, Crew, Process, LLM # Import LLM from crewai
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# --- 1. Define Utility for RAG (No longer a CrewAI Tool) ---

class RegistryRAG:
    """
    Utility class to load, embed, and search the mock medical registry.
    This replaces the crewai_tools.tool functionality.
    """
    def __init__(self, data_path="data"):
        self.data_path = data_path
        self.vector_store = self._load_and_embed_data()

    def _load_and_embed_data(self):
        """Loads text documents from the data directory and creates a FAISS vector store."""
        documents = []
        # Ensure 'data' directory exists
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            print(f"Created data directory: {self.data_path}")
            # Optionally, you might want to create a dummy file if the directory is empty
            # or provide instructions to the user to place files here.
            # For now, let's just proceed, the vector store will be empty if no files.

        for filename in os.listdir(self.data_path):
            filepath = os.path.join(self.data_path, filename)
            if filename.endswith(".txt"):
                loader = TextLoader(filepath)
                documents.extend(loader.load())
        
        if not documents:
            print(f"Warning: No text documents found in '{self.data_path}'. RAG will not retrieve any info.")
            # Return an empty FAISS index or handle appropriately
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            return FAISS.from_texts(["No documents loaded."], embeddings) # Create an empty-like index

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store

    def search_registry(self, query: str) -> str:
        """
        Searches the mock medical registry for information related to a given query.
        Returns the content of relevant documents.
        """
        if not self.vector_store:
            return "Registry not loaded. Please ensure data files are in the 'data' directory."
        
        # Perform similarity search to retrieve top relevant documents
        # Ensure that the vector store is not empty before searching
        if self.vector_store.index.ntotal == 0: # Check if the FAISS index has any vectors
            return "No information available in the registry for searching."

        docs = self.vector_store.similarity_search(query, k=3) 
        
        results = [doc.page_content for doc in docs]
        if results:
            return "\n---\n".join(results)
        else:
            return "No matching information found in the registry."

# Initialize RegistryRAG globally
registry_rag_utility = RegistryRAG()

# --- 2. Define Agents ---

# Base LLM for agents - CORRECTED WAY TO DEFINE LLM FOR CREWAI WITH GEMINI
llm = LLM(
    model="gemini/gemini-1.5-flash",
    api_key=GOOGLE_API_KEY # Use the GOOGLE_API_KEY directly here
)

credential_extraction_agent = Agent(
    role='Credential Extraction Agent',
    goal='Parse doctor resumes, certificates, and professional profiles to extract key credential details.',
    backstory=(
        "You are an expert data extractor, highly skilled in identifying and "
        "extracting critical information such as license numbers, issuing bodies, "
        "expiration dates, names, and specialties from unstructured text data."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

license_verification_agent = Agent(
    role='License Verification Agent (RAG-Enabled)',
    goal='Verify extracted credentials against medical registry information provided in context.',
    backstory=(
        "You are a meticulous verifier. Your primary task is to cross-reference "
        "provided credentials with registry data given to you in the task context "
        "to confirm their authenticity and status. You do not search the registry yourself, "
        "but analyze the information already retrieved for you."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

eligibility_risk_checker_agent = Agent(
    role='Eligibility & Risk Checker Agent',
    goal='Assess gaps, expired licenses, or country-specific compliance issues.',
    backstory=(
        "You are a compliance and risk assessment specialist. You analyze verified "
        "credential data for any discrepancies, expirations, or regulatory non-compliance issues. "
        "You are keen on identifying potential risks."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

credibility_scoring_agent = Agent(
    role='Credibility Scoring Agent',
    goal='Calculate a trust score per profile and flag high-risk credentials or unverifiable claims.',
    backstory=(
        "You are an impartial evaluator, responsible for synthesizing all verification "
        "and risk assessment findings into a concise credibility score. You clearly "
        "flag any suspicious or unverifiable claims."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# --- 3. Define Tasks ---
def define_tasks(document_text: str, rag_context_placeholder: str = ""): # Add a placeholder for RAG
    extract_credentials_task = Task(
        description=f"""
        Analyze the following document text to extract all relevant medical credential details.
        Identify:
        - Full Name of the Doctor
        - License Number(s)
        - Issuing Body/Authority (e.g., NMC, MCI, State Board, GMC)
        - Expiration Date(s) of the license(s)
        - Specialty/Area of Practice
        - Any other relevant status or remarks.

        Format the output clearly as a list of key-value pairs or a structured JSON-like format.
        Document Text:
        ---
        {document_text}
        ---
        """,
        agent=credential_extraction_agent,
        expected_output="A structured list or JSON of extracted credentials (Name, License Number, Issuing Body, Expiration Date, Specialty, Status)."
    )

    verify_licenses_task = Task(
        description=f"""
        Using the extracted credentials (provided in the context from the previous task) and the provided medical registry information below,
        meticulously verify each license number, its issuing body, and its expiration date.
        For each credential, state whether it is 'VERIFIED', 'NOT FOUND', or 'INFORMATION CONFLICT'.
        Focus on comparing the extracted details with the registry data provided to you.
        Provide verification status for each extracted credential.

        Medical Registry Information:
        ---
        {rag_context_placeholder} # This is where the RAG info will be injected
        ---
        """,
        agent=license_verification_agent,
        context=[extract_credentials_task], # The output of the extraction task will be in context
        expected_output="A detailed report on the verification status of each license, including findings from the provided registry data."
    )

    check_eligibility_risk_task = Task(
        description=f"""
        Based on the verification report (provided in context), assess the eligibility and identify any risks.
        Specifically, check for:
        - Expired licenses.
        - Gaps in licensing history (if discernible).
        - Compliance issues (e.g., if a license is active but with disciplinary remarks like 'Probation').
        - General consistency of information.
        
        Summarize all identified risks and compliance issues.
        """,
        agent=eligibility_risk_checker_agent,
        context=[verify_licenses_task], # This is correct
        expected_output="A summary of eligibility and risk assessment, detailing any issues found."
    )

    score_credibility_task = Task(
        description=f"""
        Synthesize all findings from the credential extraction, license verification, and
        eligibility/risk checking tasks (all provided in context).
        Calculate a credibility score (e.g., out of 100 or a High/Medium/Low rating).
        Clearly flag any high-risk credentials, unverifiable claims, or critical issues identified.
        Provide a final recommendation (e.g., 'Approved', 'Approved with Caution', 'Rejected').
        Explain the reasoning behind the score and recommendation.
        """,
        agent=credibility_scoring_agent,
        context=[
            extract_credentials_task,
            verify_licenses_task,
            check_eligibility_risk_task
        ],
        expected_output="A final credibility score, flags for high-risk items, and a clear recommendation with reasoning."
    )
    
    return [
        extract_credentials_task,
        verify_licenses_task,
        check_eligibility_risk_task,
        score_credibility_task
    ]

# --- 4. Define the Crew ---

def create_medical_verifier_crew(document_text: str, rag_context_for_verification: str = ""):
    """
    Creates the CrewAI crew for medical credential verification.
    rag_context_for_verification: Optional string containing relevant RAG search results to inject.
    """
    tasks = define_tasks(document_text, rag_context_for_verification) # Pass RAG context directly here
    
    medical_verifier_crew = Crew(
        agents=[
            credential_extraction_agent,
            license_verification_agent,
            eligibility_risk_checker_agent,
            credibility_scoring_agent
        ],
        tasks=tasks,
        process=Process.sequential, 
        verbose=True, 
        full_output=True, 
        manager_llm=llm 
    )
    return medical_verifier_crew

if __name__ == '__main__':
    # Create a dummy 'data' directory and a dummy text file for testing
    if not os.path.exists("data"):
        os.makedirs("data")
    with open("data/mock_registry_data.txt", "w") as f:
        f.write("License Number: TX123456, Name: Dr. Test User, Status: Active, Issuing Body: Texas Medical Board, Expiration: 2024-11-30\n")
        f.write("License Number: CA789012, Name: Dr. Test User, Status: Expired, Issuing Body: California Medical Licensing, Expiration: 2023-05-10\n")
        f.write("License Number: NY987654, Name: Dr. New York, Status: Active, Issuing Body: New York State Board, Expiration: 2025-12-31\n")
    
    dummy_doc = """
    Resume of Dr. Test User.
    Holds license TX123456 from Texas Medical Board, valid until 2024-11-30.
    Specializes in Dermatology.
    Previous experience in California, license CA789012, expired 2023-05-10.
    """
    print("Initializing RAG utility and crew with dummy document...")
    
    # Manual RAG lookup for the example (simulating app.py's process)
    # The extraction agent needs to run first to get these queries in a real scenario
    # For this __main__ block, we'll manually specify a relevant query based on the dummy doc.
    dummy_rag_query = "License Number TX123456" 
    retrieved_rag_info = registry_rag_utility.search_registry(dummy_rag_query)
    
    crew = create_medical_verifier_crew(dummy_doc, rag_context_for_verification=retrieved_rag_info)
    print("Crew initialized. Starting execution...")
    result = crew.kickoff()
    print("\n--- Crew Execution Finished ---")
    print(result)