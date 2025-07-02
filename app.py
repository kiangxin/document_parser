import streamlit as st
import os
import logging
from typing import List, Dict, Any
from src.services.document_processor import DocumentProcessor
from src.services.vector_store import VectorStoreManager
from src.utils.file_handler import FileHandler

logger = logging.getLogger(__name__)

def main():
    st.set_page_config(
        page_title="Document Parser & Vector Store",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Document Parser & Vector Store")
    st.markdown("Upload documents and convert them to LLM-ready format for vector storage")
    
    # Initialize services
    if 'doc_processor' not in st.session_state:
        st.session_state.doc_processor = DocumentProcessor()
    
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = VectorStoreManager()
    
    if 'file_handler' not in st.session_state:
        st.session_state.file_handler = FileHandler()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Vector store settings
        st.subheader("Vector Store Settings")
        index_name = st.text_input("Pinecone Index Name", value="document-parser")
        namespace = st.text_input("Namespace (optional)", value="")
        
        # Processing settings
        st.subheader("Processing Settings")
        chunk_size = st.slider("Chunk Size", min_value=100, max_value=2000, value=1000)
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=200, value=100)
        
        # Model settings
        st.subheader("Model Settings")
        embedding_model = st.selectbox(
            "Embedding Model",
            ["text-embedding-3-small"]
        )
    
    # Main content area - Upload Documents Section
    st.header("📄 Step 1: Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'txt', 'md', 'html', 'xlsx', 'pptx']
    )
    
    # Clear processed documents and status when no files are uploaded
    if not uploaded_files:
        if 'processed_documents' in st.session_state:
            del st.session_state.processed_documents
        if 'processing_status' in st.session_state:
            del st.session_state.processing_status
        if 'just_processed' in st.session_state:
            del st.session_state.just_processed
    else:
        # Check if processed documents match current uploaded files
        if 'processed_documents' in st.session_state:
            uploaded_file_names = {file.name for file in uploaded_files}
            processed_file_names = set(st.session_state.processed_documents.keys())
            
            # If files don't match, clear processed documents
            if uploaded_file_names != processed_file_names:
                if 'processed_documents' in st.session_state:
                    del st.session_state.processed_documents
                if 'processing_status' in st.session_state:
                    del st.session_state.processing_status
                if 'just_processed' in st.session_state:
                    del st.session_state.just_processed
    
    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} file(s)")
        
        # Display file information in columns for better layout
        cols = st.columns(min(3, len(uploaded_files)))
        for i, file in enumerate(uploaded_files):
            with cols[i % 3]:
                with st.expander(f"📄 {file.name}"):
                    st.write(f"**Type:** {file.type}")
                    st.write(f"**Size:** {file.size:,} bytes")
        
        # Process button - placed after the file information
        if st.button(
            "🚀 Process Documents", 
            type="primary", 
            key="process_btn",
            help="Process documents with Docling"
        ):
            process_documents_only(uploaded_files, chunk_size, chunk_overlap)
        
        # Show processing status if available
        if 'processing_status' in st.session_state:
            status = st.session_state.processing_status
            if status.get('completed'):
                st.success(f"✅ Successfully processed {len(status.get('processed_files', []))} documents")
            elif status.get('error'):
                st.error(f"❌ Processing failed: {status['error']}")
            elif status.get('processing'):
                st.info("⏳ Processing in progress...")
    
    # Step 2: Document Preview Section (shows processed results)
    if uploaded_files and 'processed_documents' in st.session_state and st.session_state.processed_documents:
        st.divider()
        
        # Auto-expand preview section after processing, but allow collapse
        preview_expanded = st.session_state.get('just_processed', False)
        if preview_expanded:
            st.session_state.just_processed = False  # Reset flag
        
        with st.expander("👁️ Step 2: Preview Processed Documents", expanded=preview_expanded):
            st.info("📄 Review parsed content before uploading to vector store")
            
            # Select file to preview from processed documents
            processed_files = list(st.session_state.processed_documents.keys())
            selected_file_name = st.selectbox("Select a processed file to preview:", processed_files, key="preview_select")
            
            if selected_file_name and selected_file_name in st.session_state.processed_documents:
                preview_data = st.session_state.processed_documents[selected_file_name]
                
                # Preview tabs
                tab1, tab2, tab3, tab4 = st.tabs(["📝 Extracted Text", "📊 Tables", "🖼️ Images", "🧩 Chunks"])
                
                with tab1:
                    if preview_data.get('full_text'):
                        st.subheader("Full Document Content")
                        preview_text = preview_data['full_text'][:3000]  # Show more for processed content
                        if len(preview_data['full_text']) > 3000:
                            preview_text += "\n\n... (content truncated for preview)"
                        st.text_area("Document Content", preview_text, height=400, disabled=True)
                        st.info(f"Total text length: {len(preview_data['full_text']):,} characters")
                    else:
                        st.info("No text content found in document")
                
                with tab2:
                    if preview_data.get('tables'):
                        st.subheader(f"Found {len(preview_data['tables'])} table(s)")
                        for i, table in enumerate(preview_data['tables']):
                            with st.expander(f"Table {i+1}"):
                                st.markdown(table)
                    else:
                        st.info("No tables found in document")
                
                with tab3:
                    if preview_data.get('images'):
                        st.subheader(f"Found {len(preview_data['images'])} image(s)")
                        for i, img_desc in enumerate(preview_data['images']):
                            st.write(f"**Image {i+1}:** {img_desc}")
                    else:
                        st.info("No images found in document")
                
                with tab4:
                    if preview_data.get('chunks'):
                        st.subheader(f"Document divided into {len(preview_data['chunks'])} chunks")
                        
                        # Show chunk statistics
                        chunk_lengths = [len(chunk.get('text', '')) for chunk in preview_data['chunks']]
                        avg_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Chunks", len(preview_data['chunks']))
                        with col2:
                            st.metric("Avg Chunk Size", f"{avg_length:.0f} chars")
                        with col3:
                            st.metric("Max Chunk Size", f"{max(chunk_lengths) if chunk_lengths else 0} chars")
                        
                        # Show individual chunks
                        for i, chunk in enumerate(preview_data['chunks'][:5]):  # Show first 5 chunks
                            with st.expander(f"Chunk {i+1} ({len(chunk.get('text', ''))} characters)"):
                                st.write(f"**Type:** {chunk.get('chunk_type', 'text')}")
                                st.write(f"**Index:** {chunk.get('chunk_index', i)}")
                                st.text_area(f"Content {i+1}", chunk.get('text', ''), height=150, disabled=True)
                        
                        if len(preview_data['chunks']) > 5:
                            st.info(f"Showing first 5 chunks. Total: {len(preview_data['chunks'])} chunks")
                    else:
                        st.info("No chunks found in processed document")
        
        st.divider()
        
        # Step 3: Upload to Pinecone Section
        st.header("📤 Step 3: Upload to Pinecone Vector Store")
        st.info("🗺️ Generate embeddings and store in Pinecone for semantic search")
        
        # Show what will be uploaded
        total_chunks = sum(len(data['chunks']) for data in st.session_state.processed_documents.values())
        st.write(f"Ready to upload **{len(st.session_state.processed_documents)} documents** with **{total_chunks} chunks** to Pinecone")
        
        # Upload to Pinecone button
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🗺️ Upload to Pinecone Vector Store", type="primary", key="upload_pinecone_btn"):
                upload_to_pinecone(index_name, namespace, embedding_model)
        
        with col2:
            if st.button("🗑️ Clear Index", key="clear_index_btn", help="Clear all vectors from Pinecone index"):
                clear_pinecone_index(index_name, namespace, embedding_model)
    
    elif uploaded_files and 'processing_status' in st.session_state and st.session_state.processing_status.get('completed'):
        st.divider()
        st.header("👁️ Step 2: Preview Processed Documents")
        st.info("⚠️ No processed documents found. Please try processing again.")
    
    elif uploaded_files:
        st.divider()
        st.header("👁️ Step 2: Preview Processed Documents")
        st.info("⏳ Please process documents first to see preview")
        st.markdown("Click the **🚀 Process Documents** button above to continue.")

    
    # Processing Status Section
    if 'processing_results' in st.session_state:
        st.header("📊 Processing Status")
        results = st.session_state.processing_results
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Documents", len(results.get('processed_files', [])))
        with col2:
            st.metric("Total Chunks", results.get('total_chunks', 0))
        with col3:
            st.metric("Vector Store Entries", results.get('vectors_stored', 0))
        
        if results.get('errors'):
            st.error("Some files failed to process:")
            for error in results['errors']:
                st.write(f"- {error}")
    
    st.divider()
    
    # AI Chatbot Section
    st.header("🤖 AI Document Chatbot")
    st.info("💬 Ask questions about your documents and get AI-powered answers with source citations")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Ask a question about your documents", placeholder="e.g., What are the main findings? Summarize the key points...")
    with col2:
        top_k = st.slider("Sources to consider", min_value=1, max_value=20, value=5, help="Number of document chunks to use as context")
    
    if st.button("💬 Ask AI Assistant", disabled=not query, type="primary") and query:
        with st.spinner("Searching documents..."):
            search_results = search_documents(query, index_name, namespace, top_k, embedding_model)
        
        # Debug info for troubleshooting
        if search_results:
            st.success(f"Found {len(search_results)} relevant document chunks")
            
            # Check if any results have text content
            has_text = any(len(result.get('text', '')) > 0 for result in search_results)
            if not has_text:
                st.error("⚠️ Search found documents but they contain no text content!")
                st.write("**Possible issues:**")
                st.write("- Documents may not have been processed correctly")
                st.write("- Namespace mismatch between upload and search")
                st.write("- Text content wasn't stored in metadata during upload")
            
            # Debug: Show what we actually got from search
            with st.expander("🔧 Debug: Search Results Structure", expanded=True if not has_text else False):
                st.write(f"**Search Configuration:**")
                st.write(f"- Index: {index_name}")
                st.write(f"- Namespace: '{namespace}' (empty means default)")
                st.write(f"- Embedding model: {embedding_model}")
                st.write(f"- Top K: {top_k}")
                st.write("---")
                
                for i, result in enumerate(search_results[:3]):  # Show first 3 results
                    st.write(f"**Result {i+1}:**")
                    st.write(f"- ID: {result.get('id', 'N/A')}")
                    st.write(f"- Score: {result.get('score', 0):.4f}")
                    text_len = len(result.get('text', ''))
                    st.write(f"- Text length: {text_len}")
                    
                    if text_len > 0:
                        st.write(f"- Text preview: '{result.get('text', '')[:200]}...'")
                        st.success("✅ This result has text content")
                    else:
                        st.error("❌ This result has NO text content")
                    
                    st.write(f"- Metadata keys: {list(result.get('metadata', {}).keys())}")
                    
                    # Show relevant metadata
                    metadata = result.get('metadata', {})
                    relevant_meta = {k: v for k, v in metadata.items() 
                                   if k in ['filename', 'source_file', 'chunk_type', 'chunk_index', 'text_length']}
                    st.json(relevant_meta)
                    st.write("---")
        else:
            st.warning("No search results found. Make sure documents are uploaded to Pinecone and the index name matches.")
            
        display_search_results(query, search_results)

def process_documents_only(uploaded_files, chunk_size: int, chunk_overlap: int):
    """Process documents with Docling only (no vector store upload)"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Initialize processing status
    st.session_state.processing_status = {'processing': True, 'completed': False, 'error': None}
    
    try:
        # Initialize processed documents storage for preview
        if 'processed_documents' not in st.session_state:
            st.session_state.processed_documents = {}
        
        processed_files = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name} with Docling...")
            
            try:
                # Save uploaded file temporarily
                temp_path = st.session_state.file_handler.save_uploaded_file(uploaded_file)
                
                # Process document with Docling
                processed_content = st.session_state.doc_processor.process_document(
                    temp_path,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
                # Store processed content for preview
                full_text = ""
                tables = []
                images = []
                
                for chunk in processed_content:
                    if chunk.get('chunk_type') == 'text':
                        full_text += chunk.get('text', '') + "\n\n"
                    elif chunk.get('chunk_type') == 'table':
                        tables.append(chunk.get('text', ''))
                    elif chunk.get('chunk_type') == 'image':
                        images.append(chunk.get('text', ''))
                
                # Store for preview
                st.session_state.processed_documents[uploaded_file.name] = {
                    'full_text': full_text.strip(),
                    'tables': tables,
                    'images': images,
                    'chunks': processed_content,
                    'file_type': uploaded_file.type,
                    'chunk_count': len(processed_content)
                }
                
                processed_files.append(uploaded_file.name)
                
                # Clean up temporary file
                st.session_state.file_handler.cleanup_temp_file(temp_path)
                
            except Exception as e:
                st.error(f"Failed to process {uploaded_file.name}: {str(e)}")
                continue
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        # Update processing status
        st.session_state.processing_status = {
            'processing': False, 
            'completed': True, 
            'processed_files': processed_files,
            'error': None
        }
        
        # Set flag to auto-expand preview section
        st.session_state.just_processed = True
        
        status_text.text("✅ Processing complete!")
        st.success(f"Successfully processed {len(processed_files)} documents with Docling")
        st.rerun()  # Refresh to show preview section
        
    except Exception as e:
        st.session_state.processing_status = {
            'processing': False, 
            'completed': False, 
            'error': str(e)
        }
        st.error(f"Processing failed: {str(e)}")

def clear_pinecone_index(index_name: str, namespace: str, embedding_model: str):
    """Clear all vectors from Pinecone index"""
    try:
        # Configure vector store
        st.session_state.vector_store.configure(
            index_name=index_name,
            namespace=namespace,
            embedding_model=embedding_model
        )
        
        # Clear the index
        st.session_state.vector_store.clear_index(namespace)
        st.success(f"✅ Successfully cleared all vectors from index '{index_name}' in namespace '{namespace or 'default'}'")
        
    except Exception as e:
        st.error(f"Failed to clear index: {str(e)}")

def upload_to_pinecone(index_name: str, namespace: str, embedding_model: str):
    """Upload processed documents to Pinecone vector store"""
    
    if 'processed_documents' not in st.session_state or not st.session_state.processed_documents:
        st.error("No processed documents found. Please process documents first.")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = {
        'processed_files': [],
        'total_chunks': 0,
        'vectors_stored': 0,
        'errors': []
    }
    
    try:
        # Configure vector store
        status_text.text("Configuring vector store...")
        st.session_state.vector_store.configure(
            index_name=index_name,
            namespace=namespace,
            embedding_model=embedding_model
        )
        
        processed_docs = list(st.session_state.processed_documents.items())
        
        for i, (filename, doc_data) in enumerate(processed_docs):
            status_text.text(f"Uploading {filename} to Pinecone...")
            
            try:
                # Get chunks from processed document
                processed_content = doc_data['chunks']
                
                # Debug: Check what we're about to upload
                st.write(f"**Debug - Uploading {filename}:**")
                st.write(f"- Total chunks: {len(processed_content)}")
                for j, chunk in enumerate(processed_content[:2]):  # Show first 2 chunks
                    st.write(f"- Chunk {j+1}: {len(chunk.get('text', ''))} characters")
                    st.write(f"  Preview: '{chunk.get('text', '')[:100]}...'")
                
                # Store in vector database
                vector_ids = st.session_state.vector_store.store_documents(
                    processed_content,
                    metadata={'filename': filename, 'file_type': doc_data['file_type']}
                )
                
                results['processed_files'].append(filename)
                results['total_chunks'] += len(processed_content)
                results['vectors_stored'] += len(vector_ids)
                
            except Exception as e:
                results['errors'].append(f"{filename}: {str(e)}")
                st.error(f"Failed to upload {filename}: {str(e)}")
            
            progress_bar.progress((i + 1) / len(processed_docs))
        
        st.session_state.processing_results = results
        status_text.text("✅ Upload to Pinecone complete!")
        st.success(f"Successfully uploaded {len(results['processed_files'])} documents to Pinecone")
        
        if results['errors']:
            st.warning(f"Some uploads failed: {len(results['errors'])} errors")
            with st.expander("View errors"):
                for error in results['errors']:
                    st.write(f"- {error}")
        
    except Exception as e:
        st.error(f"Upload to Pinecone failed: {str(e)}")


def search_documents(query: str, index_name: str, namespace: str, top_k: int, embedding_model: str):
    """Search documents in vector store"""
    try:
        # Configure vector store before searching
        st.session_state.vector_store.configure(
            index_name=index_name,
            namespace=namespace,
            embedding_model=embedding_model
        )
        
        results = st.session_state.vector_store.search(
            query=query,
            top_k=top_k,
            index_name=index_name,
            namespace=namespace
        )
        
        # Debug: Log search results
        logger.info(f"Search returned {len(results)} results for query: '{query}'")
        for i, result in enumerate(results[:3]):  # Log first 3 results
            text_preview = result.get('text', '')[:100] + "..." if len(result.get('text', '')) > 100 else result.get('text', '')
            logger.info(f"Result {i+1}: Score={result.get('score', 0):.3f}, Text preview='{text_preview}'")
        
        return results
    except Exception as e:
        st.error(f"Search failed: {str(e)}")
        return []

def generate_rag_response(query: str, search_results: List[Dict[str, Any]]) -> str:
    """Generate response using GPT-4o-mini with RAG context"""
    try:
        from openai import OpenAI
        
        # Initialize OpenAI client
        client = OpenAI()
        
        # Debug: Check if we have search results
        if not search_results:
            return "No relevant documents found to answer your question. Please make sure you have uploaded and processed documents first."
        
        # Prepare context from search results
        context_parts = []
        for i, result in enumerate(search_results[:5]):  # Use top 5 results
            text = result.get('text', '').strip()
            metadata = result.get('metadata', {})
            filename = metadata.get('filename', f'Document {i+1}')
            
            # Debug: Check if text is empty
            if not text:
                logger.warning(f"Empty text found in search result {i+1}")
                continue
                
            context_parts.append(f"[Source: {filename}]\n{text}")
        
        # Debug: Check if we have any valid context
        if not context_parts:
            return "The documents were found but appear to contain no readable text content. Please check if your documents were processed correctly."
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Debug: Log context length
        logger.info(f"Generated context with {len(context_parts)} sources, total length: {len(context)} characters")
        
        # Create system prompt for RAG
        system_prompt = """You are an intelligent document assistant. You have access to relevant document excerpts that can help answer the user's question. 

Your task is to:
1. Analyze the provided document context carefully
2. Answer the user's question based primarily on the information in the documents
3. If the documents don't contain enough information to fully answer the question, clearly state what information is missing
4. Cite which documents or sources support your answer when relevant
5. Be concise but comprehensive in your response
6. If the question cannot be answered from the provided context, say so clearly

Always prioritize accuracy over completeness. If you're uncertain about something, acknowledge the uncertainty."""

        # Create user prompt with context
        user_prompt = f"""Based on the following document excerpts, please answer this question: "{query}"

Document Context:
{context}

Please provide a helpful and accurate response based on the information above."""

        # Generate response using GPT-4o-mini
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Lower temperature for more focused responses
            max_tokens=1500
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Failed to generate RAG response: {str(e)}")
        return f"Sorry, I encountered an error while generating a response: {str(e)}"

def display_search_results(query: str, results: List[Dict[str, Any]]):
    """Display AI response and search results"""
    if not results:
        st.info("No results found")
        return
    
    # Generate and display AI response
    st.subheader("🤖 AI Assistant Response")
    
    with st.spinner("Generating AI response based on your documents..."):
        ai_response = generate_rag_response(query, results)
    
    # Display AI response in a nice container
    with st.container():
        st.markdown("**Question:** " + query)
        st.markdown("**Answer:**")
        st.write(ai_response)
    
    st.divider()
    
    # Display raw search results in expandable section
    with st.expander(f"🔍 View Source Documents ({len(results)} results)", expanded=False):
        st.info("These are the document excerpts that were used to generate the AI response above.")
        
        for i, result in enumerate(results):
            with st.expander(f"Source {i+1} - Score: {result.get('score', 0):.3f}"):
                st.write("**Content:**")
                st.write(result.get('text', ''))
                
                if result.get('metadata'):
                    st.write("**Metadata:**")
                    metadata_display = {k: v for k, v in result['metadata'].items() 
                                      if k not in ['text_length', 'created_at', 'embedding_model']}
                    st.json(metadata_display)


if __name__ == "__main__":
    main()