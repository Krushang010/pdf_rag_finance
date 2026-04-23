import streamlit as st
import tempfile
from rag_pipeline import run_rag_pipeline

st.set_page_config(page_title="Financial RAG Analyzer", layout="wide")

st.title("📊 Financial Report Analyzer (RAG)")
st.write("Upload a financial PDF and get insights")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    st.success("PDF uploaded successfully!")

    if st.button("Analyze"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        with st.spinner("Processing PDF..."):
            result = run_rag_pipeline(pdf_path)

        st.subheader("📈 Analysis Result")
        st.write(result)