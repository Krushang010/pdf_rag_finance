from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Put your PDF file in project root
pdf_path = "TDPOWERSYS_29012026162954_Outcome_of_Board_Meeting_Financial_Results.pdf"

loader = PyPDFLoader(pdf_path)

documents = loader.load()

# Print number of pages
print(f"Total pages: {len(documents)}\n")

# Print first page
print("First page content:\n")
print(documents[0].page_content)

# 🔹 Step 2: Chunking

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}\n")

# Print first chunk
print("First chunk:\n")
print(chunks[0].page_content)




# Step 3: Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)

# Convert chunks into embeddings
embeddings = embedding_model.embed_documents(
    [chunk.page_content for chunk in chunks]
)

print(f"\nTotal embeddings created: {len(embeddings)}")
print(f"Embedding dimension: {len(embeddings[0])}")