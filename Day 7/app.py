# app.py
import os
import json
import re
import ast
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# === 1. Setup Gemini API Key ===
os.environ["GOOGLE_API_KEY"] = "AIzaSyBFwk5xfWX3xpFB1Etq-9xuKCU37uDS7y0"  # Replace with your key

# === 2. Load FAISS Vectorstore ===
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.load_local("faiss_registry_index", embedding, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# === 3. Setup Gemini Model ===
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# === 4. Define Agents ===

# Credential Extraction Agent
extraction_prompt = PromptTemplate(
    input_variables=["input"],
    template="""
Extract License Number, Issued By, and Expiry Date from the following text.
Output as JSON with these keys: License Number, Issued By, Expiry Date.

Text:
{input}
"""
)

# License Verification Agent (RAG)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)

# Eligibility & Risk Checker Agent
risk_check_prompt = PromptTemplate(
    input_variables=["input"],
    template="""
Given the following extracted credential info, check:
- If the license is expired (Today's date is 2025-06-17)
- Any missing fields
- If issuing body matches common authorities (NMC, MCI, State Medical Council)
Output as JSON: {{'Expired': True/False, 'Missing Fields': [], 'Invalid Issuing Body': True/False}}

Data:
{input}
"""
)

# Credibility Scoring Agent
score_prompt = PromptTemplate(
    input_variables=["input"],
    template="""
Based on this credential check, assign a Credibility Score out of 100.

Criteria:
- Valid License, Not Expired: +70
- Issued by recognized authority: +20
- All fields present: +10

Output JSON: {{'Credibility Score': 0-100}}

Data:
{input}
"""
)

# === 5. Utility Function ===
def safe_parse_json(text):
    try:
        # If Gemini returns JSON as dict-like Python literal, parse with ast
        result = ast.literal_eval(text)
        # Convert Python dict to JSON-compatible string, then load as JSON
        return json.loads(json.dumps(result))
    except Exception as e1:
        try:
            # If it fails, try directly as JSON
            return json.loads(text)
        except Exception as e2:
            st.error(f"❌ JSON Parse Error: {e1} | {e2}")
            return None


# === 6. Streamlit App ===
st.title("🔍 License & Credential Verifier (RAG + Gemini + Streamlit)")

resume_text = st.text_area("Paste Doctor Resume / Credential Text:")

if st.button("Verify Credentials") and resume_text:
    with st.spinner("Extracting Credentials..."):
        extraction_chain = extraction_prompt.format(input=resume_text)
        extraction_result = llm.invoke(extraction_chain)
        match = re.search(r'```json(.*?)```', extraction_result.content, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            credentials = safe_parse_json(json_str)
            if credentials:
                st.subheader("🔑 Extracted Credentials")
                st.json(credentials)
            else:
                st.stop()
        else:
            st.error("❌ Extraction Failed!")
            st.stop()

    with st.spinner("Verifying License with RAG..."):
        query = f"License Number: {credentials.get('License Number', '')} Issued By: {credentials.get('Issued By', '')}"
        rag_result = qa_chain.run(query)
        st.subheader("📄 License Verification Result (RAG)")
        st.write(rag_result)

    with st.spinner("Checking Risk & Eligibility..."):
        risk_chain = risk_check_prompt.format(input=json.dumps(credentials))
        risk_result = llm.invoke(risk_chain)
        risk_match = re.search(r'```json(.*?)```', risk_result.content, re.DOTALL)
        if risk_match:
            risk_str = risk_match.group(1).strip()
            risk_json = safe_parse_json(risk_str)
            if risk_json:
                st.subheader("⚠️ Eligibility & Risk Check")
                st.json(risk_json)
        else:
            st.error("❌ Risk Check Failed!")

    with st.spinner("Calculating Credibility Score..."):
        score_chain = score_prompt.format(input=json.dumps(credentials))
        score_result = llm.invoke(score_chain)
        score_match = re.search(r'```json(.*?)```', score_result.content, re.DOTALL)
        if score_match:
            score_str = score_match.group(1).strip()
            score_json = safe_parse_json(score_str)
            if score_json:
                st.subheader("⭐ Credibility Score")
                st.json(score_json)
        else:
            st.error("❌ Scoring Failed!")

else:
    st.info("👆 Please paste resume text and click 'Verify Credentials'.")
