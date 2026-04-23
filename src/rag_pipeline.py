def run_rag_pipeline(pdf_path):
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

    # 1. Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )
    chunks = splitter.split_documents(documents)

    # 3. Embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # 4. Vector DB (Chroma)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    # 5. Retrieval query (analysis-focused)
    query = "Revenue, Profit Before Tax, Profit After Tax comparison latest quarter vs previous year financial results"

    results = vector_store.similarity_search(query, k=5)

    # 6. Build structured context
    context = "\n\n".join([
        f"[Chunk {i+1}]\n{doc.page_content}"
        for i, doc in enumerate(results)
    ])

    # 7. LLM call (analysis prompt)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are an expert equity research analyst."
            },
            {
                "role": "user",
                "content": f"""
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
"""
            }
        ],
        temperature=0.2
    )

    # 8. Return final result
    final_answer = response.choices[0].message.content

    return final_answer