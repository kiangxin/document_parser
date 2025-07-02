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

## 🔍 Technical Details

### Document Processing Pipeline

1. **File Upload**: Temporary storage with validation
2. **Format Detection**: Automatic file type identification
3. **Content Extraction**: Docling processes the document
4. **Text Segmentation**: Smart chunking with overlap
5. **Metadata Enrichment**: Document and chunk-level metadata
6. **Embedding Generation**: Vector representation creation
7. **Vector Storage**: Pinecone database insertion
8. **Cleanup**: Temporary file removal

### Embedding Strategy

- **Chunk-level Embeddings**: Each text chunk gets its own vector
- **Metadata Preservation**: Source document info maintained
- **Batch Processing**: Efficient API usage
- **Deduplication**: Content-based vector ID generation

### Search Implementation

- **Semantic Similarity**: Vector-based document matching
- **Hybrid Search**: Combine vector and metadata filtering
- **Result Ranking**: Configurable relevance scoring
- **Context Preservation**: Maintain document relationships

## 🛠️ Development

### Adding New Document Formats

1. Update `supported_formats` in `ProcessingConfig`
2. Modify `DocumentProcessor` to handle new format
3. Test with sample documents
4. Update documentation

### Custom Embedding Models

1. Implement model loading in `VectorStoreManager`
2. Update configuration options
3. Handle dimension compatibility
4. Test embedding quality

### Performance Optimization

- **Batch Processing**: Process multiple documents simultaneously
- **Caching**: Cache embeddings for duplicate content
- **Async Operations**: Non-blocking file processing
- **Memory Management**: Efficient handling of large documents

## 📈 Monitoring & Logging

- **Structured Logging**: Comprehensive error tracking
- **Processing Metrics**: Document count, chunk statistics
- **Performance Monitoring**: Processing time tracking
- **Error Handling**: Graceful failure recovery

## 🔒 Security Considerations

- **File Validation**: Size and type restrictions
- **Temporary Storage**: Secure file handling
- **API Key Management**: Environment variable protection
- **Data Privacy**: Local processing, external storage

## 🚨 Troubleshooting

### Common Issues

1. **Pinecone Connection Errors**
   - Verify API key and index configuration
   - Check network connectivity
   - Ensure index exists and is ready

2. **Document Processing Failures**
   - Validate file format and size
   - Check OCR requirements for scanned PDFs
   - Review error logs for specific issues

3. **Embedding Model Issues**
   - Verify model availability and downloads
   - Check memory requirements
   - Ensure proper model configuration

4. **Search Performance**
   - Optimize chunk size and overlap
   - Consider index optimization
   - Review metadata filtering logic

## 🔮 Future Enhancements

- **Multi-modal Search**: Support for image and table search
- **Document Comparison**: Similarity analysis between documents
- **Batch APIs**: RESTful API for programmatic access
- **Advanced Analytics**: Document insights and statistics
- **Integration Options**: Webhook and API integrations
- **Cloud Deployment**: Docker and cloud platform support

## 📝 License

[Add your license information here]

## 🤝 Contributing

[Add contribution guidelines here]