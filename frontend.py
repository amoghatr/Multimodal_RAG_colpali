import json
import uuid
from typing import Dict, List, Optional

import requests
import streamlit as st

# Configure the page
st.set_page_config(
    page_title="ColPali RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API Configuration
API_BASE_URL = "http://localhost:8000"


def get_or_create_session_id() -> str:
    """Get or create a session ID for the user"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id


def upload_pdfs(files: List, session_id: str) -> Dict:
    """Upload PDF files to the backend"""
    files_data = []
    for file in files:
        files_data.append(("files", (file.name, file.getvalue(), "application/pdf")))

    try:
        response = requests.post(
            f"{API_BASE_URL}/ingest-pdfs/",
            files=files_data,
            params={"session_id": session_id},
            timeout=300,  # 5 minutes timeout for large files
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error uploading files: {str(e)}")
        return {"error": str(e)}


def query_documents(query: str, top_k: int, session_id: str) -> Optional[str]:
    """Query the documents"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query/",
            params={"query": query, "top_k": top_k, "session_id": session_id},
            timeout=60,
            stream=True,
        )
        response.raise_for_status()

        # Collect streaming response
        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    if "answer" in data and data["answer"]:
                        full_response = data["answer"]
                except json.JSONDecodeError:
                    continue

        return full_response
    except requests.exceptions.RequestException as e:
        st.error(f"Error querying documents: {str(e)}")
        return None


def main():
    # Header
    st.title("📚 ColPali RAG System")
    st.markdown(
        "*Upload PDFs and query them with natural language using ColPali embeddings*"
    )

    # Sidebar
    with st.sidebar:
        st.header("🔧 Settings")
        session_id = get_or_create_session_id()
        st.info(f"Session ID: `{session_id[:8]}...`")

        if st.button("🔄 New Session", help="Start a new session"):
            del st.session_state.session_id
            st.rerun()

        st.markdown("---")
        st.markdown("### 📖 How it works")
        st.markdown(
            """
        1. **Upload PDFs**: Upload one or more PDF documents
        2. **Processing**: Documents are converted to images and embedded using ColPali
        3. **Query**: Ask questions about your documents in natural language
        4. **Results**: Get AI-powered answers based on document content
        """
        )

    # Initialize session state
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "query_history" not in st.session_state:
        st.session_state.query_history = []

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 Upload Documents")

        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF documents to analyze",
        )

        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} file(s):")
            for file in uploaded_files:
                st.write(f"• {file.name} ({file.size:,} bytes)")

            if st.button("🚀 Process Documents", type="primary"):
                with st.spinner("Processing documents... This may take a few minutes."):
                    result = upload_pdfs(uploaded_files, session_id)

                    if "error" not in result:
                        st.success("✅ Documents processed successfully!")
                        st.session_state.uploaded_files.extend(
                            [f.name for f in uploaded_files]
                        )

                        # Show processing results
                        if "results" in result:
                            st.write("**Processing Summary:**")
                            for item in result["results"]:
                                if "error" in item:
                                    st.error(f"❌ {item['filename']}: {item['error']}")
                                else:
                                    st.success(
                                        f"✅ {item['filename']}: {item['num_pages']} pages processed"
                                    )
                    else:
                        st.error("❌ Error processing documents")

        # Show uploaded files
        if st.session_state.uploaded_files:
            st.write("**Uploaded Files:**")
            for filename in st.session_state.uploaded_files:
                st.write(f"📄 {filename}")

    with col2:
        st.header("🔍 Query Documents")

        if not st.session_state.uploaded_files:
            st.info("👆 Please upload and process some documents first!")
        else:
            query = st.text_area(
                "Ask a question about your documents:",
                placeholder="e.g., What are the main topics discussed in the documents?",
                height=100,
            )

            col2a, col2b = st.columns([2, 1])
            with col2a:
                top_k = st.slider("Number of results to consider", 1, 10, 5)
            with col2b:
                query_button = st.button(
                    "🔍 Search", type="primary", disabled=not query.strip()
                )

            if query_button and query.strip():
                with st.spinner("Searching documents..."):
                    answer = query_documents(query, top_k, session_id)

                    if answer:
                        st.markdown("### 💬 Answer")
                        st.markdown(answer)

                        # Add to history
                        st.session_state.query_history.append(
                            {
                                "query": query,
                                "answer": answer,
                                "timestamp": str(uuid.uuid4())[:8],
                            }
                        )
                    else:
                        st.error("❌ No answer received. Please try again.")

    # Query History
    if st.session_state.query_history:
        st.markdown("---")
        st.header("📜 Query History")

        for i, item in enumerate(
            reversed(st.session_state.query_history[-5:])
        ):  # Show last 5
            with st.expander(
                f"Query {len(st.session_state.query_history) - i}: {item['query'][:50]}..."
            ):
                st.markdown(f"**Question:** {item['query']}")
                st.markdown(f"**Answer:** {item['answer']}")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Powered by ColPali, FastAPI, and Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
