# Document Parser & Vector Store

A sophisticated AI/ML document processing system that converts various document formats into LLM-ready format and stores them in Pinecone vector database for semantic search and retrieval.

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit     │    │    Docling       │    │   Pinecone      │
│   Frontend      │◄──►│  Document        │◄──►│ Vector Store    │
│                 │    │  Processor       │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ File Upload     │    │ Text Extraction  │    │ Semantic Search │
│ Management      │    │ & Chunking       │    │ & Retrieval     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Features

### Document Processing
- **Multi-format Support**: PDF, DOCX, PPTX, XLSX, HTML, Markdown, TXT, and more
- **Advanced OCR**: Optical Character Recognition for scanned documents
- **Table Extraction**: Intelligent table detection and conversion to structured format
- **Image Processing**: Extract and describe images within documents
- **Smart Chunking**: Context-aware text segmentation with configurable overlap

### Vector Storage
- **Pinecone Integration**: Scalable vector database for similarity search
- **Multiple Embedding Models**: Support for Sentence Transformers and OpenAI embeddings
- **Metadata Management**: Rich metadata storage for filtering and organization
- **Namespace Support**: Logical separation of document collections

### User Interface
- **Streamlit Web App**: Intuitive drag-and-drop file upload interface
- **Real-time Processing**: Live progress tracking and status updates
- **Search Interface**: Semantic search with configurable result ranking
- **Configuration Panel**: Dynamic settings adjustment without code changes

## 📁 Project Structure

```
document_parser/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .env.example                   # Environment variables template
├── README.md                      # Project documentation
│
├── src/
│   ├── services/
│   │   ├── document_processor.py  # Docling integration
│   │   └── vector_store.py        # Pinecone vector operations
│   │
│   └── utils/
│       ├── file_handler.py        # File upload/management utilities
│       └── logger.py              # Logging configuration
│
├── config/
│   └── settings.py                # Configuration management
│
└── tests/                         # Unit tests (to be implemented)
```

## 🔧 System Components

### 1. Document Processor (`src/services/document_processor.py`)
- **Docling Integration**: Leverages Docling's advanced document conversion capabilities
- **Content Extraction**: Extracts text, tables, and images from various document formats
- **Intelligent Chunking**: Splits documents into semantically meaningful chunks
- **Metadata Enrichment**: Adds document-level and chunk-level metadata

### 2. Vector Store Manager (`src/services/vector_store.py`)
- **Pinecone Operations**: Create, update, delete, and search vector embeddings
- **Embedding Generation**: Support for multiple embedding models
- **Batch Processing**: Efficient bulk operations for large document sets
- **Search Functionality**: Semantic similarity search with metadata filtering

### 3. File Handler (`src/utils/file_handler.py`)
- **Upload Management**: Secure temporary file storage for uploaded documents
- **File Validation**: Size and format validation before processing
- **Cleanup Operations**: Automatic temporary file cleanup

### 4. Configuration System (`config/settings.py`)
- **Environment Variables**: Centralized configuration management
- **Default Values**: Sensible defaults with override capabilities
- **Validation**: Configuration validation and error handling

## ⚙️ Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Pinecone Configuration (Required)
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=document-index
PINECONE_ENVIRONMENT=us-east-1

# Embedding Model (Optional)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Processing Settings (Optional)
CHUNK_SIZE=500
CHUNK_OVERLAP=50
EXTRACT_TABLES=true
EXTRACT_IMAGES=false

# OpenAI (Optional, for better embeddings)
OPENAI_API_KEY=your_openai_api_key_here
```

### Supported Embedding Models

1. **Sentence Transformers** (Default):
   - `sentence-transformers/all-MiniLM-L6-v2` (Fast, 384 dimensions)
   - `sentence-transformers/all-mpnet-base-v2` (Better quality, 768 dimensions)
   - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Multilingual)

2. **OpenAI Embeddings**:
   - `text-embedding-ada-002` (High quality, 1536 dimensions)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd document_parser

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your API keys
```

### 2. Pinecone Setup

1. Create account at [Pinecone](https://pinecone.io)
2. Create a new index with:
   - **Dimension**: 384 (for default model) or 1536 (for OpenAI)
   - **Metric**: Cosine
   - **Cloud**: AWS (recommended)
   - **Region**: us-east-1 (default)

### 3. Run Application

```bash
streamlit run app.py
```

Access the application at `http://localhost:8501`

## 📊 Usage Workflow

### 1. Document Upload
- Upload one or multiple documents via the web interface
- Supported formats: PDF, DOCX, PPTX, XLSX, HTML, MD, TXT, etc.
- Real-time file validation and size checking

### 2. Processing Configuration
- **Chunk Size**: Control text segment length (100-2000 characters)
- **Chunk Overlap**: Set overlap between chunks (0-200 characters)
- **Embedding Model**: Choose between different embedding models
- **Extraction Options**: Enable/disable table and image extraction

### 3. Document Processing
- Docling converts documents to structured format
- Text extraction with OCR support for scanned documents
- Table detection and conversion to markdown format
- Image description extraction (optional)
- Smart chunking with context preservation

### 4. Vector Storage
- Generate embeddings for each text chunk
- Store vectors in Pinecone with rich metadata
- Batch processing for efficiency
- Automatic cleanup of temporary files

### 5. Search & Retrieval
- Semantic search across processed documents
- Adjustable result count and relevance scoring
- Metadata filtering capabilities
- Result highlighting and source attribution

