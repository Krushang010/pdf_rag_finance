from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load PDF
pdf_path = "TDPOWERSYS_29012026162954_Outcome_of_Board_Meeting_Financial_Results.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

print(f"Total pages: {len(documents)}\n")

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
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