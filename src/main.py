from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from groq import Groq
import os
from dotenv import load_dotenv

# Load env
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load PDF
pdf_path = "TDPOWERSYS_29012026162954_Outcome_of_Board_Meeting_Financial_Results.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

print(f"Total pages: {len(documents)}\n")

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300
)

chunks = splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}\n")

# Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

embeddings = embedding_model.embed_documents(
    [chunk.page_content for chunk in chunks]
)

print(f"Total embeddings: {len(embeddings)}")
print(f"Embedding size: {len(embeddings[0])}")

# Vector DB
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

print("Chroma DB created successfully")

# Query
query = "Profit After Tax value for latest quarter standalone financial results"

results = vector_store.similarity_search(query, k=5)

print("\nTop results:\n")

for i, res in enumerate(results):
    print(f"Result {i+1}:\n")
    print(res.page_content)
    print("\n" + "-"*50)

# ✅ FIX: create context
context = "\n\n".join([doc.page_content for doc in results])

# LLM call
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "system",
            "content": "You are an expert equity research analyst."
        },
        {
            "role": "user",
            "content":  f"""
Analyze the company's financial performance using ONLY the provided context.

STRICT RULES:
- Use ONLY the given context (no external knowledge)
- DO NOT assume or estimate missing values
- If something is not present → say "Not Available"
- Prefer exact numbers from context

Tasks:
1. Compare latest quarter vs previous year
2. Identify growth drivers or concerns
3. Comment on profitability (PAT vs revenue)
4. Detect margin expansion or contraction
5. Give final investment sentiment

Context:
{context}

Return format:

Trend:
- Revenue Growth:
- Profit Growth:

Key Insights:
- 
- 
- 

Risk/Concern:
- 

Final Sentiment:
Positive / Neutral / Negative

Confidence:
High / Medium / Low
   """     }
    ],
    temperature=0.2
)

print("\nFinal Answer:\n")
print(response.choices[0].message.content)