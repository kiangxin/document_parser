import os
import tempfile
import shutil
from typing import Optional, List
from pathlib import Path
import logging
import streamlit as st

logger = logging.getLogger(__name__)

class FileHandler:
    """
    File handling utilities for uploaded documents
    """
    
    def __init__(self, temp_dir: str = "temp"):
        self.temp_dir = temp_dir
        self._ensure_temp_dir()
    
    def _ensure_temp_dir(self):
        """Ensure temporary directory exists"""
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
    
    def save_uploaded_file(self, uploaded_file) -> str:
        """
        Save Streamlit uploaded file to temporary location
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            Path to saved temporary file
        """
        try:
            # Create unique filename
            file_extension = Path(uploaded_file.name).suffix
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=file_extension,
                dir=self.temp_dir
            )
            
            # Write file content
            temp_file.write(uploaded_file.getbuffer())
            temp_file.close()
            
            logger.info(f"Saved uploaded file to: {temp_file.name}")
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Failed to save uploaded file {uploaded_file.name}: {str(e)}")
            raise
    
    def cleanup_temp_file(self, file_path: str) -> bool:
        """
        Clean up temporary file
        
        Args:
            file_path: Path to temporary file
            
        Returns:
            True if successful
        """
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to cleanup temporary file {file_path}: {str(e)}")
            return False
    
    def cleanup_temp_dir(self) -> bool:
        """
        Clean up entire temporary directory
        
        Returns:
            True if successful
        """
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                self._ensure_temp_dir()
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to cleanup temporary directory {self.temp_dir}: {str(e)}")
            return False
    
    def get_file_info(self, file_path: str) -> dict:
        """
        Get file information
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with file information
        """
        try:
            stat = os.stat(file_path)
            return {
                'name': os.path.basename(file_path),
                'path': file_path,
                'size': stat.st_size,
                'extension': Path(file_path).suffix.lower(),
                'created': stat.st_ctime,
                'modified': stat.st_mtime
            }
            
        except Exception as e:
            logger.error(f"Failed to get file info for {file_path}: {str(e)}")
            return {}
    
    def validate_file_size(self, file_path: str, max_size_mb: float = 200) -> bool:
        """
        Validate file size
        
        Args:
            file_path: Path to file
            max_size_mb: Maximum file size in MB
            
        Returns:
            True if file size is valid
        """
        try:
            file_size = os.path.getsize(file_path)
            max_size_bytes = max_size_mb * 1024 * 1024
            return file_size <= max_size_bytes
            
        except Exception as e:
            logger.error(f"Failed to validate file size for {file_path}: {str(e)}")
            return False
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions"""
        return [
            '.pdf', '.docx', '.doc', '.pptx', '.ppt',
            '.xlsx', '.xls', '.html', '.htm', '.md',
            '.txt', '.rtf', '.odt', '.odp', '.ods'
        ]
    
    def is_supported_file(self, file_path: str) -> bool:
        """
        Check if file type is supported
        
        Args:
            file_path: Path to file
            
        Returns:
            True if file type is supported
        """
        extension = Path(file_path).suffix.lower()
        return extension in self.get_supported_extensions()