from typing import List, Dict, Any, Optional
import os
import logging
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import time
import hashlib

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    Vector store manager for Pinecone integration with document embeddings
    """
    
    def __init__(self):
        self.pc = None
        self.index = None
        self.index_name = None
        self.embedding_model = None
        self.embedding_model_name = None
        self.dimension = None
        self._initialize_pinecone()
    
    def _initialize_pinecone(self):
        """Initialize Pinecone client"""
        try:
            api_key = os.getenv('PINECONE_API_KEY')
            if not api_key:
                raise ValueError("PINECONE_API_KEY environment variable not set")
            
            self.pc = Pinecone(api_key=api_key)
            logger.info("Pinecone client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise
    
    def configure(
        self,
        index_name: str,
        namespace: str = "",
        embedding_model: str = "text-embedding-3-small",
        dimension: Optional[int] = None
    ):
        """
        Configure vector store with index and embedding model
        
        Args:
            index_name: Name of the Pinecone index
            namespace: Namespace for vectors
            embedding_model: Name of the embedding model
            dimension: Vector dimension (auto-detected if not provided)
        """
        try:
            # Load embedding model
            if self.embedding_model_name != embedding_model:
                logger.info(f"Loading embedding model: {embedding_model}")
                
                if embedding_model.startswith("sentence-transformers/"):
                    self.embedding_model = SentenceTransformer(embedding_model)
                    self.dimension = self.embedding_model.get_sentence_embedding_dimension()
                elif embedding_model in ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]:
                    # OpenAI embeddings
                    from openai import OpenAI
                    self.embedding_model = OpenAI()
                    if embedding_model == "text-embedding-3-small":
                        self.dimension = 1536  # text-embedding-3-small dimension
                    elif embedding_model == "text-embedding-3-large":
                        self.dimension = 3072  # text-embedding-3-large dimension
                    else:  # text-embedding-ada-002
                        self.dimension = 1536  # Ada-002 dimension
                else:
                    raise ValueError(f"Unsupported embedding model: {embedding_model}")
                
                self.embedding_model_name = embedding_model
            
            # Set dimension if provided
            if dimension:
                self.dimension = dimension
            
            # Connect to or create index
            self._setup_index(index_name, namespace)
            
        except Exception as e:
            logger.error(f"Failed to configure vector store: {str(e)}")
            raise
    
    def _setup_index(self, index_name: str, namespace: str):
        """Setup Pinecone index"""
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if index_name not in existing_indexes:
                logger.info(f"Creating new index: {index_name}")
                self.pc.create_index(
                    name=index_name,
                    dimension=self.dimension,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                
                # Wait for index to be ready
                while not self.pc.describe_index(index_name).status['ready']:
                    time.sleep(1)
            
            # Connect to index
            self.index = self.pc.Index(index_name)
            self.index_name = index_name  # Store index name separately
            self.namespace = namespace
            
            logger.info(f"Connected to index: {index_name} (namespace: {namespace or 'default'})")
            
        except Exception as e:
            logger.error(f"Failed to setup index {index_name}: {str(e)}")
            raise
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        try:
            if self.embedding_model_name.startswith("sentence-transformers/"):
                embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
                return embeddings.tolist()
            
            elif self.embedding_model_name in ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]:
                embeddings = []
                for text in texts:
                    response = self.embedding_model.embeddings.create(
                        input=text,
                        model=self.embedding_model_name
                    )
                    embeddings.append(response.data[0].embedding)
                return embeddings
            
            else:
                raise ValueError(f"Unsupported embedding model: {self.embedding_model_name}")
                
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise
    
    def store_documents(
        self,
        documents: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Store documents in vector store
        
        Args:
            documents: List of document chunks with text and metadata
            metadata: Additional metadata to add to all documents
            
        Returns:
            List of vector IDs
        """
        try:
            if not self.index:
                raise ValueError("Vector store not configured. Call configure() first.")
            
            texts = [doc['text'] for doc in documents]
            embeddings = self._generate_embeddings(texts)
            
            # Prepare vectors for upsert
            vectors = []
            vector_ids = []
            
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                # Generate unique ID
                vector_id = self._generate_vector_id(doc['text'], doc.get('metadata', {}))
                vector_ids.append(vector_id)
                
                # Combine metadata
                combined_metadata = doc.get('metadata', {}).copy()
                if metadata:
                    combined_metadata.update(metadata)
                
                # Add processing metadata
                combined_metadata.update({
                    'text_length': len(doc['text']),
                    'chunk_type': doc.get('chunk_type', 'text'),
                    'chunk_index': doc.get('chunk_index', i),
                    'embedding_model': self.embedding_model_name,
                    'created_at': time.time()
                })
                
                # Store the actual text content in metadata (essential for retrieval)
                combined_metadata['text'] = doc['text']
                
                vectors.append({
                    'id': vector_id,
                    'values': embedding,
                    'metadata': combined_metadata
                })
            
            # Upsert vectors in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(
                    vectors=batch,
                    namespace=self.namespace
                )
            
            logger.info(f"Stored {len(vectors)} vectors in Pinecone")
            return vector_ids
            
        except Exception as e:
            logger.error(f"Failed to store documents: {str(e)}")
            raise
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        index_name: Optional[str] = None,
        namespace: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            index_name: Index name (if different from configured)
            namespace: Namespace (if different from configured)
            filter_metadata: Metadata filter
            
        Returns:
            List of search results with text, metadata, and scores
        """
        try:
            if not self.index:
                raise ValueError("Vector store not configured. Call configure() first.")
            
            # Use different index if specified
            index = self.index
            if index_name and index_name != self.index_name:
                index = self.pc.Index(index_name)
            
            # Use different namespace if specified
            search_namespace = namespace if namespace is not None else self.namespace
            
            # Generate query embedding
            query_embedding = self._generate_embeddings([query])[0]
            
            # Search
            results = index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=search_namespace,
                filter=filter_metadata,
                include_metadata=True
            )
            
            # Format results
            formatted_results = []
            for i, match in enumerate(results['matches']):
                text_content = match['metadata'].get('text', '')
                
                # Debug logging
                logger.info(f"Match {i+1}: ID={match['id'][:8]}..., Score={match['score']:.4f}")
                logger.info(f"  Raw metadata keys: {list(match['metadata'].keys())}")
                logger.info(f"  Text content length: {len(text_content)}")
                if text_content:
                    logger.info(f"  Text preview: '{text_content[:100]}...'")
                else:
                    logger.warning(f"  EMPTY TEXT FOUND! This might be an old document.")
                    # Check if there are other text-related fields
                    text_fields = [k for k in match['metadata'].keys() if 'text' in k.lower()]
                    logger.info(f"  Available text-related fields: {text_fields}")
                
                formatted_results.append({
                    'id': match['id'],
                    'text': text_content,
                    'score': match['score'],
                    'metadata': {k: v for k, v in match['metadata'].items() if k != 'text'}
                })
            
            logger.info(f"Returning {len(formatted_results)} formatted results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search documents: {str(e)}")
            raise
    
    def delete_documents(
        self,
        vector_ids: List[str],
        namespace: Optional[str] = None
    ) -> bool:
        """
        Delete documents from vector store
        
        Args:
            vector_ids: List of vector IDs to delete
            namespace: Namespace (if different from configured)
            
        Returns:
            True if successful
        """
        try:
            if not self.index:
                raise ValueError("Vector store not configured. Call configure() first.")
            
            delete_namespace = namespace if namespace is not None else self.namespace
            
            # Delete vectors
            self.index.delete(
                ids=vector_ids,
                namespace=delete_namespace
            )
            
            logger.info(f"Deleted {len(vector_ids)} vectors from Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {str(e)}")
            raise
    
    def get_index_stats(self, index_name: Optional[str] = None) -> Dict[str, Any]:
        """Get index statistics"""
        try:
            index = self.index
            if index_name and index_name != self.index_name:
                index = self.pc.Index(index_name)
            
            stats = index.describe_index_stats()
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get index stats: {str(e)}")
            raise
    
    def _generate_vector_id(self, text: str, metadata: Dict[str, Any]) -> str:
        """Generate unique vector ID based on content and metadata"""
        content = text + str(sorted(metadata.items()))
        return hashlib.md5(content.encode()).hexdigest()
    
    def clear_index(self, namespace: Optional[str] = None) -> bool:
        """Clear all vectors from the index"""
        try:
            if not self.index:
                raise ValueError("Vector store not configured. Call configure() first.")
            
            clear_namespace = namespace if namespace is not None else self.namespace
            
            # Delete all vectors in the namespace
            self.index.delete(delete_all=True, namespace=clear_namespace)
            
            logger.info(f"Cleared all vectors from index in namespace: {clear_namespace or 'default'}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear index: {str(e)}")
            raise

    def list_available_models(self) -> List[str]:
        """List available embedding models"""
        return [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ]