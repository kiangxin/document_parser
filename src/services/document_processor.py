from typing import List, Dict, Any, Optional
import os
from pathlib import Path
import logging
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Document processor using Docling for converting various document formats
    to LLM-ready text with proper chunking and metadata extraction.
    """
    
    def __init__(self):
        self._setup_pipeline_options()
    
    def _setup_pipeline_options(self):
        """Setup pipeline options for different document formats"""
        try:
            # Configure PDF pipeline options
            pipeline_options = PdfPipelineOptions(
                do_ocr=True,
                do_table_structure=True
            )
            
            # Configure table structure options if available
            try:
                pipeline_options.table_structure_options.do_cell_matching = True
            except AttributeError:
                # Table structure options may not be available in all versions
                pass
            
            # Use PdfFormatOption to wrap pipeline options
            pdf_format_option = PdfFormatOption(pipeline_options=pipeline_options)
            
            # Configure converter with format options
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: pdf_format_option,
                }
            )
            logger.info("DocumentConverter initialized with PdfFormatOption and pipeline options")
            
        except Exception as e:
            # Handle any configuration errors
            logger.warning(f"Failed to configure PDF pipeline options: {str(e)}. Trying simplified configuration.")
            try:
                # Try simplified pipeline options
                pipeline_options = PdfPipelineOptions(do_ocr=True)
                pdf_format_option = PdfFormatOption(pipeline_options=pipeline_options)
                
                self.converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: pdf_format_option,
                    }
                )
                logger.info("DocumentConverter initialized with simplified PdfFormatOption")
                
            except Exception as e2:
                # Final fallback to default converter
                logger.warning(f"Simplified configuration also failed: {str(e2)}. Using default converter.")
                self.converter = DocumentConverter()
    
    def process_document(
        self,
        file_path: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        extract_tables: bool = True,
        extract_images: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Process a document and return chunked content with metadata
        
        Args:
            file_path: Path to the document file
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            extract_tables: Whether to extract table data
            extract_images: Whether to extract image data
            
        Returns:
            List of dictionaries containing chunked content and metadata
        """
        try:
            # Convert document
            result = self.converter.convert(file_path)
            
            # Extract content
            content_chunks = []
            
            # Process main text content
            full_text = result.document.export_to_markdown()
            text_chunks = self._chunk_text(full_text, chunk_size, chunk_overlap)
            
            for i, chunk in enumerate(text_chunks):
                content_chunks.append({
                    'text': chunk,
                    'chunk_index': i,
                    'chunk_type': 'text',
                    'metadata': {
                        'source_file': os.path.basename(file_path),
                        'file_type': self._get_file_type(file_path),
                        'total_chunks': len(text_chunks),
                        'document_title': self._extract_title(result.document),
                        'page_count': getattr(result.document, 'page_count', 1)
                    }
                })
            
            # Process tables if requested
            if extract_tables:
                table_chunks = self._extract_tables(result.document, file_path)
                content_chunks.extend(table_chunks)
            
            # Process images if requested
            if extract_images:
                image_chunks = self._extract_images(result.document, file_path)
                content_chunks.extend(image_chunks)
            
            logger.info(f"Successfully processed {file_path} into {len(content_chunks)} chunks")
            return content_chunks
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise
    
    def _chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start:
                    end = sentence_end + 1
                else:
                    # Look for paragraph breaks
                    para_end = text.rfind('\n\n', start, end)
                    if para_end > start:
                        end = para_end + 2
                    else:
                        # Look for any line break
                        line_end = text.rfind('\n', start, end)
                        if line_end > start:
                            end = line_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - chunk_overlap
            
            # Prevent infinite loops
            if start >= len(text):
                break
        
        return chunks
    
    def _extract_tables(self, document, file_path: str) -> List[Dict[str, Any]]:
        """Extract table data from document"""
        table_chunks = []
        
        try:
            # Extract tables from document
            tables = getattr(document, 'tables', [])
            
            for i, table in enumerate(tables):
                # Convert table to markdown format
                table_markdown = self._table_to_markdown(table)
                
                if table_markdown:
                    table_chunks.append({
                        'text': table_markdown,
                        'chunk_index': i,
                        'chunk_type': 'table',
                        'metadata': {
                            'source_file': os.path.basename(file_path),
                            'file_type': self._get_file_type(file_path),
                            'table_index': i,
                            'content_type': 'table'
                        }
                    })
        
        except Exception as e:
            logger.warning(f"Error extracting tables from {file_path}: {str(e)}")
        
        return table_chunks
    
    def _extract_images(self, document, file_path: str) -> List[Dict[str, Any]]:
        """Extract image data and descriptions from document"""
        image_chunks = []
        
        try:
            # Extract images from document
            images = getattr(document, 'images', [])
            
            for i, image in enumerate(images):
                # Get image description or caption
                description = getattr(image, 'description', '') or getattr(image, 'caption', '')
                
                if description:
                    image_chunks.append({
                        'text': f"Image {i+1}: {description}",
                        'chunk_index': i,
                        'chunk_type': 'image',
                        'metadata': {
                            'source_file': os.path.basename(file_path),
                            'file_type': self._get_file_type(file_path),
                            'image_index': i,
                            'content_type': 'image'
                        }
                    })
        
        except Exception as e:
            logger.warning(f"Error extracting images from {file_path}: {str(e)}")
        
        return image_chunks
    
    def _table_to_markdown(self, table) -> str:
        """Convert table object to markdown format"""
        try:
            # This is a simplified implementation
            # Actual implementation would depend on Docling's table structure
            if hasattr(table, 'to_markdown'):
                return table.to_markdown()
            elif hasattr(table, 'data'):
                # Convert table data to markdown
                rows = table.data
                if not rows:
                    return ""
                
                # Create markdown table
                markdown_lines = []
                
                # Header row
                if len(rows) > 0:
                    header = " | ".join(str(cell) for cell in rows[0])
                    markdown_lines.append(f"| {header} |")
                    markdown_lines.append("|" + " --- |" * len(rows[0]))
                
                # Data rows
                for row in rows[1:]:
                    row_str = " | ".join(str(cell) for cell in row)
                    markdown_lines.append(f"| {row_str} |")
                
                return "\n".join(markdown_lines)
            
            return str(table)
            
        except Exception as e:
            logger.warning(f"Error converting table to markdown: {str(e)}")
            return str(table)
    
    def _extract_title(self, document) -> str:
        """Extract document title"""
        try:
            if hasattr(document, 'title') and document.title:
                return document.title
            elif hasattr(document, 'metadata') and document.metadata.get('title'):
                return document.metadata['title']
            else:
                return "Untitled Document"
        except:
            return "Untitled Document"
    
    def _get_file_type(self, file_path: str) -> str:
        """Get file type from file extension"""
        return Path(file_path).suffix.lower()
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported document formats"""
        return [
            '.pdf', '.docx', '.doc', '.pptx', '.ppt',
            '.xlsx', '.xls', '.html', '.htm', '.md',
            '.txt', '.rtf', '.odt', '.odp', '.ods'
        ]
    
    def is_supported_format(self, file_path: str) -> bool:
        """Check if file format is supported"""
        file_ext = Path(file_path).suffix.lower()
        return file_ext in self.get_supported_formats()