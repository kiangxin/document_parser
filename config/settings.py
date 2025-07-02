import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class PineconeConfig:
    """Pinecone configuration settings"""
    api_key: str
    environment: str = "us-east-1"
    index_name: str = "document-index"
    namespace: str = ""
    metric: str = "cosine"
    dimension: int = 1536

@dataclass
class EmbeddingConfig:
    """Embedding model configuration"""
    model_name: str = "text-embedding-3-small"
    batch_size: int = 32
    max_seq_length: int = 512

@dataclass
class ProcessingConfig:
    """Document processing configuration"""
    chunk_size: int = 500
    chunk_overlap: int = 50
    extract_tables: bool = True
    extract_images: bool = False
    ocr_enabled: bool = True
    supported_formats: list = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = [
                '.pdf', '.docx', '.doc', '.pptx', '.ppt',
                '.xlsx', '.xls', '.html', '.htm', '.md',
                '.txt', '.rtf', '.odt', '.odp', '.ods'
            ]

@dataclass
class AppConfig:
    """Application configuration"""
    title: str = "Document Parser & Vector Store"
    page_icon: str = "📄"
    layout: str = "wide"
    max_upload_size: int = 200  # MB
    temp_dir: str = "temp"
    log_level: str = "INFO"

class Settings:
    """
    Application settings manager
    """
    
    def __init__(self):
        self.pinecone = self._load_pinecone_config()
        self.embedding = self._load_embedding_config()
        self.processing = self._load_processing_config()
        self.app = self._load_app_config()
    
    def _load_pinecone_config(self) -> PineconeConfig:
        """Load Pinecone configuration from environment variables"""
        api_key = os.getenv('PINECONE_API_KEY')
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        
        return PineconeConfig(
            api_key=api_key,
            environment=os.getenv('PINECONE_ENVIRONMENT', 'us-east-1'),
            index_name=os.getenv('PINECONE_INDEX_NAME', 'document-index'),
            namespace=os.getenv('PINECONE_NAMESPACE', ''),
            metric=os.getenv('PINECONE_METRIC', 'cosine'),
            dimension=int(os.getenv('PINECONE_DIMENSION', '1536'))
        )
    
    def _load_embedding_config(self) -> EmbeddingConfig:
        """Load embedding configuration from environment variables"""
        return EmbeddingConfig(
            model_name=os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
            batch_size=int(os.getenv('EMBEDDING_BATCH_SIZE', '32')),
            max_seq_length=int(os.getenv('EMBEDDING_MAX_SEQ_LENGTH', '512'))
        )
    
    def _load_processing_config(self) -> ProcessingConfig:
        """Load processing configuration from environment variables"""
        return ProcessingConfig(
            chunk_size=int(os.getenv('CHUNK_SIZE', '500')),
            chunk_overlap=int(os.getenv('CHUNK_OVERLAP', '50')),
            extract_tables=os.getenv('EXTRACT_TABLES', 'true').lower() == 'true',
            extract_images=os.getenv('EXTRACT_IMAGES', 'false').lower() == 'true',
            ocr_enabled=os.getenv('OCR_ENABLED', 'true').lower() == 'true'
        )
    
    def _load_app_config(self) -> AppConfig:
        """Load application configuration from environment variables"""
        return AppConfig(
            title=os.getenv('APP_TITLE', 'Document Parser & Vector Store'),
            page_icon=os.getenv('APP_PAGE_ICON', '📄'),
            layout=os.getenv('APP_LAYOUT', 'wide'),
            max_upload_size=int(os.getenv('MAX_UPLOAD_SIZE_MB', '200')),
            temp_dir=os.getenv('TEMP_DIR', 'temp'),
            log_level=os.getenv('LOG_LEVEL', 'INFO')
        )
    
    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI configuration if available"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return {}
        
        return {
            'api_key': api_key,
            'organization': os.getenv('OPENAI_ORGANIZATION'),
            'model': os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
            'max_tokens': int(os.getenv('OPENAI_MAX_TOKENS', '1000')),
            'temperature': float(os.getenv('OPENAI_TEMPERATURE', '0.7'))
        }
    
    def validate_config(self) -> bool:
        """Validate configuration settings"""
        try:
            # Check required environment variables
            required_vars = ['PINECONE_API_KEY']
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            
            if missing_vars:
                raise ValueError(f"Missing required environment variables: {missing_vars}")
            
            # Validate numeric values
            if self.processing.chunk_size <= 0:
                raise ValueError("CHUNK_SIZE must be positive")
            
            if self.processing.chunk_overlap < 0:
                raise ValueError("CHUNK_OVERLAP must be non-negative")
            
            if self.processing.chunk_overlap >= self.processing.chunk_size:
                raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE")
            
            if self.app.max_upload_size <= 0:
                raise ValueError("MAX_UPLOAD_SIZE_MB must be positive")
            
            return True
            
        except Exception as e:
            print(f"Configuration validation failed: {str(e)}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            'pinecone': {
                'environment': self.pinecone.environment,
                'index_name': self.pinecone.index_name,
                'namespace': self.pinecone.namespace,
                'metric': self.pinecone.metric,
                'dimension': self.pinecone.dimension
            },
            'embedding': {
                'model_name': self.embedding.model_name,
                'batch_size': self.embedding.batch_size,
                'max_seq_length': self.embedding.max_seq_length
            },
            'processing': {
                'chunk_size': self.processing.chunk_size,
                'chunk_overlap': self.processing.chunk_overlap,
                'extract_tables': self.processing.extract_tables,
                'extract_images': self.processing.extract_images,
                'ocr_enabled': self.processing.ocr_enabled,
                'supported_formats': self.processing.supported_formats
            },
            'app': {
                'title': self.app.title,
                'page_icon': self.app.page_icon,
                'layout': self.app.layout,
                'max_upload_size': self.app.max_upload_size,
                'temp_dir': self.app.temp_dir,
                'log_level': self.app.log_level
            }
        }

# Global settings instance
settings = Settings()