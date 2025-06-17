import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

# === 1. Set Gemini API Key ===
os.environ["GOOGLE_API_KEY"] = "AIzaSyBFwk5xfWX3xpFB1Etq-9xuKCU37uDS7y0"  # Replace this with your actual Gemini API key

# === 2. Mock Medical License Registry Data (3 records) ===
data = [
    "License Number: NMC12345 Issued By: NMC Expiry Date: 2026-12-31",
    "License Number: MCI67890 Issued By: MCI Expiry Date: 2025-08-30",
    "License Number: STATE54321 Issued By: State Medical Council Expiry Date: 2024-11-15"
]

# === 3. Convert to Langchain Documents ===
docs = [Document(page_content=record) for record in data]

# === 4. Embeddings using Gemini ===
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# === 5. Build FAISS Index ===
db = FAISS.from_documents(docs, embedding)

# === 6. Save FAISS Index Locally ===
db.save_local("faiss_registry_index")

print("✅ FAISS Index successfully built and saved in 'faiss_registry_index/' folder.")
