import streamlit as st
import json
import os
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your existing modules (adjust paths as needed)
try:
    # Assuming your modules are in the same directory or properly installed
    import sys
    sys.path.append('.')  # Add current directory to path
    
    from pdf_extractor import DocumentExtractor, ExtractionConfig
    from multi_modal_rag import AnthropicOnlyRAG
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Multimodal RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stAlert > div {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .processing-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
    .query-response {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables - SAFE initialization"""
    defaults = {
        'documents': {},
        'rag_system': None,
        'processed_files': [],
        'current_doc_id': None,
        'query_history': [],
        'processing_status': 'ready',
        'extraction_config': None,
        'api_key_valid': False,
        'last_query_result': None,
        'processing_metrics': {},
        'debug_mode': False,
        'current_api_key': '',
        'show_image_debug': False,
        'show_image_test': False,
        'show_force_reprocess': False
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def validate_api_key(api_key: str) -> bool:
    """Validate Anthropic API key - SIMPLIFIED to prevent recursion"""
    if not api_key or not api_key.startswith('sk-'):
        return False
    
    # Simple format validation only to prevent recursion issues
    # Real validation happens during first API call
    return len(api_key) > 20  # Basic length check

def create_extraction_config(use_ai: bool = False, ai_key: str = None) -> ExtractionConfig:
    """Create extraction configuration - MAINTAINS YOUR EXACT DIRECTORY STRUCTURE"""
    return ExtractionConfig(
        use_ai_service=use_ai,
        ai_api_key=ai_key,
        ai_service_url="https://api.anthropic.com/v1/messages" if use_ai else None,
        output_dir="extracted_content",  # YOUR EXACT DIRECTORY
        save_images=True,
        ocr_language="eng",
        table_extraction_method="auto",
        min_table_rows=2,
        min_table_cols=2,
        image_dpi=300,
        enable_preprocessing=True
    )

def process_document(uploaded_file, api_key: str) -> Dict[str, Any]:
    """Process uploaded document through extraction and RAG preparation"""
    
    with st.spinner("Processing document..."):
        start_time = time.time()
        
        # Ensure directories exist with YOUR EXACT STRUCTURE
        os.makedirs("extracted_content", exist_ok=True)
        os.makedirs("extracted_content/images", exist_ok=True)
        os.makedirs("extracted_content/JSON", exist_ok=True)
        
        # CRITICAL FIX: Use original filename, not temp file name
        original_filename = uploaded_file.name
        file_stem = Path(original_filename).stem  # This becomes part of image names
        
        # Clear any existing images for this document to prevent conflicts
        clean_existing_images(file_stem)
        
        # Create temporary file in a way that preserves original naming for extraction
        temp_dir = tempfile.mkdtemp()
        temp_file_path = Path(temp_dir) / original_filename  # Keep original name!
        
        try:
            # Write uploaded file to temp location with original name
            with open(temp_file_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            # Step 1: Extract document content using original filename
            st.write("📄 Extracting document content...")
            config = create_extraction_config(use_ai=True, ai_key=api_key)
            extractor = DocumentExtractor(config)
            
            # Pass the properly named file for extraction
            extraction_result = extractor.extract_document(str(temp_file_path))
            extraction_time = time.time() - start_time
            
            # CRITICAL: Save JSON in your exact format and location
            json_output_path = Path("extracted_content/JSON") / f"{file_stem}.json"
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(extraction_result, f, indent=2, ensure_ascii=False)
            
            st.write(f"💾 Saved extraction to: {json_output_path}")
            
            # Step 2: Initialize RAG system with EXACT SAME LOGIC
            st.write("🧠 Preparing RAG system...")
            rag_start = time.time()
            
            # IMPORTANT: Reset RAG system for new document to prevent conflicts
            st.session_state.rag_system = AnthropicOnlyRAG(api_key)
            
            # Process document through RAG - THIS IS WHERE IMAGE PATHS ARE CRITICAL
            chunks = st.session_state.rag_system.process_json_document(extraction_result)
            rag_time = time.time() - rag_start
            
            # VERIFY IMAGE PATHS WERE CREATED CORRECTLY
            image_verification = verify_image_paths(extraction_result, file_stem)
            st.write(f"🖼️ Images verified: {image_verification['found']}/{image_verification['expected']}")
            
            # Show actual image files found
            if image_verification['expected'] > 0:
                show_extracted_images(file_stem, image_verification)
            
            total_time = time.time() - start_time
            
            # Create document metadata
            doc_id = f"doc_{int(time.time())}"
            doc_metadata = {
                'id': doc_id,
                'filename': uploaded_file.name,
                'file_stem': file_stem,  # Store for image path reconstruction
                'upload_time': datetime.now().isoformat(),
                'extraction_result': extraction_result,
                'json_path': str(json_output_path),
                'chunk_count': len(chunks),
                'processing_time': total_time,
                'extraction_time': extraction_time,
                'rag_time': rag_time,
                'file_size': len(uploaded_file.getvalue()),
                'pages': len(extraction_result.get('pages', [])),
                'tables': sum(len(page.get('tables', [])) for page in extraction_result.get('pages', [])),
                'images': sum(len(page.get('images', [])) for page in extraction_result.get('pages', [])),
                'image_verification': image_verification
            }
            
            # Store in session state
            st.session_state.documents[doc_id] = doc_metadata
            st.session_state.processed_files.append(uploaded_file.name)
            st.session_state.current_doc_id = doc_id
            st.session_state.processing_metrics[doc_id] = doc_metadata
            
            return doc_metadata
            
        finally:
            # Clean up temporary directory
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except:
                pass

def clean_existing_images(file_stem: str):
    """Clean any existing images for this document to prevent conflicts"""
    image_dir = Path("extracted_content/images")
    if image_dir.exists():
        # Remove any existing images for this document
        for img_file in image_dir.glob(f"{file_stem}_page_*_img_*.png"):
            try:
                img_file.unlink()
                st.write(f"🗑️ Removed existing image: {img_file.name}")
            except Exception as e:
                st.warning(f"Could not remove {img_file.name}: {e}")

def show_extracted_images(file_stem: str, verification: Dict[str, int]):
    """Show the actual extracted images"""
    image_dir = Path("extracted_content/images")
    found_images = list(image_dir.glob(f"{file_stem}_page_*_img_*.png"))
    
    if found_images:
        with st.expander(f"🖼️ Extracted Images ({len(found_images)} found)"):
            cols = st.columns(min(3, len(found_images)))
            for i, img_path in enumerate(sorted(found_images)[:6]):  # Show first 6
                with cols[i % 3]:
                    try:
                        st.image(str(img_path), caption=img_path.name, width=150)
                        st.caption(f"Size: {img_path.stat().st_size:,} bytes")
                    except Exception as e:
                        st.error(f"Could not display {img_path.name}: {e}")
            
            if len(found_images) > 6:
                st.info(f"... and {len(found_images) - 6} more images")
    else:
        st.warning("No images found in expected location")

def verify_image_paths(extraction_result: Dict, file_stem: str) -> Dict[str, int]:
    """Verify that images were saved with correct paths matching your naming convention"""
    expected_images = 0
    found_images = 0
    
    for page in extraction_result.get('pages', []):
        page_num = page.get('page_number', 1) - 1  # Convert to 0-based
        images = page.get('images', [])
        
        for img_idx, image in enumerate(images):
            expected_images += 1
            
            # YOUR EXACT NAMING PATTERN
            expected_path = f"extracted_content/images/{file_stem}_page_{page_num}_img_{img_idx}.png"
            
            if Path(expected_path).exists():
                found_images += 1
            else:
                # Check alternative paths that might have been created
                alternatives = [
                    f"extracted_content/images/{file_stem}_page_{page_num}_image_{img_idx}.png",
                    f"extracted_content/images/{file_stem}_{page_num}_{img_idx}.png"
                ]
                for alt_path in alternatives:
                    if Path(alt_path).exists():
                        found_images += 1
                        break
    
    return {"expected": expected_images, "found": found_images}

def display_processing_results(doc_metadata: Dict[str, Any]):
    """Display processing results in a nice format"""
    st.markdown("### 📊 Processing Complete!")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Pages", doc_metadata['pages'])
    with col2:
        st.metric("Tables", doc_metadata['tables'])
    with col3:
        st.metric("Images", doc_metadata['images'])
    with col4:
        st.metric("Chunks Created", doc_metadata['chunk_count'])
    
    # Image verification status
    if 'image_verification' in doc_metadata:
        verification = doc_metadata['image_verification']
        if verification['expected'] > 0:
            success_rate = (verification['found'] / verification['expected']) * 100
            if success_rate == 100:
                st.success(f"✅ All {verification['expected']} images saved successfully!")
            else:
                st.warning(f"⚠️ {verification['found']}/{verification['expected']} images saved ({success_rate:.0f}%)")
    
    # Processing time breakdown
    st.markdown("#### ⏱️ Processing Time Breakdown")
    time_col1, time_col2, time_col3 = st.columns(3)
    
    with time_col1:
        st.metric("Extraction", f"{doc_metadata['extraction_time']:.2f}s")
    with time_col2:
        st.metric("RAG Preparation", f"{doc_metadata['rag_time']:.2f}s")
    with time_col3:
        st.metric("Total Time", f"{doc_metadata['processing_time']:.2f}s")
    
    # File paths info
    if st.session_state.debug_mode:
        with st.expander("📁 File System Info"):
            st.write(f"**JSON saved to:** `{doc_metadata.get('json_path', 'N/A')}`")
            st.write(f"**Images directory:** `extracted_content/images/`")
            st.write(f"**Naming pattern:** `{doc_metadata.get('file_stem', 'filename')}_page_X_img_Y.png`")

def query_documents(query: str, doc_id: str = None) -> Dict[str, Any]:
    """Query the processed documents"""
    
    if st.session_state.rag_system is None:
        st.error("No RAG system initialized. Please process a document first.")
        return None
    
    with st.spinner("Searching documents..."):
        start_time = time.time()
        
        # Perform the query
        result = st.session_state.rag_system.query_with_selective_image_processing(
            query, top_k=5
        )
        
        query_time = time.time() - start_time
        
        # Add metadata to result
        result['query_time'] = query_time
        result['timestamp'] = datetime.now().isoformat()
        result['doc_id'] = doc_id or st.session_state.current_doc_id
        
        # Store in query history
        st.session_state.query_history.append({
            'query': query,
            'timestamp': result['timestamp'],
            'doc_id': result['doc_id'],
            'images_analyzed': result['images_analyzed'],
            'query_time': query_time
        })
        
        st.session_state.last_query_result = result
        return result

def display_query_result(result: Dict[str, Any]):
    """Display query results in a formatted way"""
    
    st.markdown('<div class="query-response">', unsafe_allow_html=True)
    
    # Query info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Query Time", f"{result['query_time']:.2f}s")
    with col2:
        st.metric("Images Analyzed", result['images_analyzed'])
    with col3:
        st.metric("Relevant Chunks", len(result['relevant_chunks']))
    
    # Main answer
    st.markdown("#### 🎯 Answer")
    st.markdown(result['answer'])
    
    # Show relevant chunks if debug mode is on
    if st.session_state.debug_mode:
        with st.expander("🔍 Relevant Content Chunks"):
            for i, chunk in enumerate(result['relevant_chunks']):
                st.markdown(f"**Chunk {i+1}** (Type: {chunk.chunk_type}, Page: {chunk.page})")
                st.text(chunk.content[:300] + "..." if len(chunk.content) > 300 else chunk.content)
                st.markdown("---")
    
    # Show processed images if any
    if result['processed_images']:
        with st.expander("🖼️ Image Analysis"):
            for img_id, description in result['processed_images'].items():
                st.markdown(f"**{img_id}:**")
                st.write(description)
    
    st.markdown('</div>', unsafe_allow_html=True)

def sidebar_content():
    """Render sidebar content - FIXED to prevent recursion"""
    st.sidebar.markdown("## ⚙️ Configuration")
    
    # API Key input with key to prevent re-creation
    api_key = st.sidebar.text_input(
        "Anthropic API Key", 
        type="password",
        help="Enter your Anthropic API key to enable document processing",
        key="api_key_input"  # Fixed key to prevent recreation
    )
    
    # Only update if key actually changed
    if api_key and api_key != st.session_state.get('current_api_key', ''):
        st.session_state.current_api_key = api_key
        st.session_state.api_key_valid = validate_api_key(api_key)
        # Don't reset rag_system here to prevent issues
    
    # API Key status
    if st.session_state.get('api_key_valid', False):
        st.sidebar.success("✅ API Key Valid")
    elif api_key:
        st.sidebar.warning("⚠️ API Key Format Check - Will validate on first use")
    else:
        st.sidebar.warning("⚠️ API Key Required")
    
    st.sidebar.markdown("---")
    
    # Document management
    st.sidebar.markdown("## 📁 Documents")
    
    if st.session_state.get('processed_files', []):
        st.sidebar.markdown("### Processed Files:")
        for i, filename in enumerate(st.session_state.processed_files):
            if st.sidebar.button(f"📄 {filename}", key=f"doc_btn_{i}"):
                # Find doc_id for this filename
                for doc_id, doc_data in st.session_state.documents.items():
                    if doc_data['filename'] == filename:
                        st.session_state.current_doc_id = doc_id
                        break
    else:
        st.sidebar.info("No documents processed yet")
    
    st.sidebar.markdown("---")
    
    # Debug mode toggle
    st.session_state.debug_mode = st.sidebar.checkbox(
        "🐛 Debug Mode", 
        value=st.session_state.get('debug_mode', False),
        help="Show detailed chunk information and processing details",
        key="debug_mode_checkbox"  # Fixed key
    )
    
    # Image debugging utilities
    if st.session_state.debug_mode and st.session_state.get('current_doc_id'):
        st.sidebar.markdown("### 🖼️ Image Debug Tools")
        
        if st.sidebar.button("🔍 Debug Image Paths", key="debug_paths_btn"):
            st.session_state.show_image_debug = True
        
        if st.sidebar.button("🧪 Test Image Processing", key="test_image_btn"):
            st.session_state.show_image_test = True
            
        if st.sidebar.button("🔄 Force Reprocess Images", key="force_reprocess_btn"):
            st.session_state.show_force_reprocess = True
    
        # Clear all data - ENHANCED with image cleanup
        if st.sidebar.button("🗑️ Clear All Data", type="secondary", key="clear_data_btn"):
            # Clear session state safely
            keys_to_clear = ['documents', 'processed_files', 'query_history', 'current_doc_id', 'last_query_result', 'rag_system']
            for key in keys_to_clear:
                if key in st.session_state:
                    if key == 'documents':
                        st.session_state[key] = {}
                    elif key in ['processed_files', 'query_history']:
                        st.session_state[key] = []
                    else:
                        st.session_state[key] = None
            
            # IMPORTANT: Also clear extracted images and JSONs
            clear_extracted_content()
            
            st.sidebar.success("All data and files cleared!")
            time.sleep(1)  # Brief pause before rerun
            st.rerun()

def clear_extracted_content():
    """Clear all extracted content files"""
    try:
        # Clear images
        image_dir = Path("extracted_content/images")
        if image_dir.exists():
            for img_file in image_dir.glob("*.png"):
                img_file.unlink()
            st.sidebar.info(f"🖼️ Cleared images")
        
        # Clear JSONs
        json_dir = Path("extracted_content/JSON")
        if json_dir.exists():
            for json_file in json_dir.glob("*.json"):
                json_file.unlink()
            st.sidebar.info(f"📄 Cleared JSON files")
                
    except Exception as e:
        st.sidebar.warning(f"Could not clear some files: {e}")
    
    # Query history
    if st.session_state.get('query_history', []):
        st.sidebar.markdown("## 📝 Recent Queries")
        for i, query_data in enumerate(reversed(st.session_state.query_history[-5:])):
            with st.sidebar.expander(f"Query {len(st.session_state.query_history) - i}"):
                st.write(f"**Q:** {query_data['query'][:50]}...")
                st.write(f"**Time:** {query_data['query_time']:.2f}s")
                st.write(f"**Images:** {query_data['images_analyzed']}")

def force_image_reprocessing():
    """Force reprocessing of images with Claude Vision to get detailed descriptions"""
    if not st.session_state.current_doc_id or not st.session_state.rag_system:
        st.error("No document or RAG system available")
        return
    
    current_doc = st.session_state.documents[st.session_state.current_doc_id]
    extraction_result = current_doc.get('extraction_result', {})
    file_stem = current_doc.get('file_stem', '')
    
    st.markdown("### 🔄 Force Image Reprocessing")
    
    total_images = 0
    processed_images = 0
    
    with st.spinner("Reprocessing all images with Claude Vision..."):
        for page in extraction_result.get('pages', []):
            page_num = page.get('page_number', 1) - 1
            images = page.get('images', [])
            
            for img_idx, image in enumerate(images):
                total_images += 1
                image_id = f"page_{page_num}_img_{img_idx}"
                
                st.write(f"Processing {image_id}...")
                
                # Find the chunk for this image
                for chunk in st.session_state.rag_system.chunks:
                    if chunk.chunk_type == 'image_placeholder' and image_id in chunk.related_images:
                        # Force Claude to process this image
                        description = st.session_state.rag_system._process_image_with_claude(chunk, image_id)
                        
                        if description and len(description) > 50:  # Ensure we got a real description
                            processed_images += 1
                            st.success(f"✅ {image_id}: {description[:100]}...")
                            
                            # Store the description for future use
                            st.session_state.rag_system.image_descriptions[image_id] = description
                        else:
                            st.error(f"❌ Failed to process {image_id}")
                        break
    
    st.write(f"**Results:** {processed_images}/{total_images} images successfully processed with Claude")

def debug_current_document_images():
    """Debug image paths for current document"""
    if not st.session_state.current_doc_id:
        st.error("No document selected")
        return
    
    doc_metadata = st.session_state.documents[st.session_state.current_doc_id]
    
    st.markdown("### 🔍 Image Path Debug Report")
    
    # Use your RAG system's debug function
    if st.session_state.rag_system:
        with st.expander("📋 RAG System Debug Output"):
            try:
                st.session_state.rag_system.debug_image_paths()
                st.success("Debug function executed - check console output")
            except Exception as e:
                st.error(f"Debug function failed: {e}")
    
    # Additional verification with visual feedback
    file_stem = doc_metadata.get('file_stem', 'unknown')
    extraction_result = doc_metadata.get('extraction_result', {})
    
    st.markdown("#### 📂 Expected vs Actual Image Paths")
    
    for page in extraction_result.get('pages', []):
        page_num = page.get('page_number', 1) - 1
        images = page.get('images', [])
        
        if images:
            st.markdown(f"**Page {page_num + 1}:**")
            
            for img_idx, image in enumerate(images):
                expected_path = f"extracted_content/images/{file_stem}_page_{page_num}_img_{img_idx}.png"
                stored_path = image.get('file_path', 'Not stored')
                
                exists = Path(expected_path).exists()
                status = "✅" if exists else "❌"
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.write(f"{status} Image {img_idx}")
                with col2:
                    st.code(f"Expected: {expected_path}")
                    st.code(f"Stored:   {stored_path}")
                    
                    if exists:
                        # Show file size
                        file_size = Path(expected_path).stat().st_size
                        st.write(f"File size: {file_size:,} bytes")
                        
                        # Show thumbnail
                        try:
                            st.image(expected_path, width=150, caption=f"Page {page_num+1}, Image {img_idx}")
                        except Exception as e:
                            st.warning(f"Could not display image: {e}")

def test_image_processing_current_doc():
    """Test image processing for current document"""
    if not st.session_state.current_doc_id or not st.session_state.rag_system:
        st.error("No document or RAG system available")
        return
    
    st.markdown("### 🧪 Testing Image Processing")
    
    # Use your RAG system's test function
    with st.spinner("Testing image processing..."):
        try:
            result = st.session_state.rag_system.test_image_processing()
            
            if result and len(result) > 50:  # Check if we got a real description
                st.success("✅ Image processing test successful!")
                st.text_area("Claude's Image Description:", result, height=150)
            else:
                st.error("❌ Image processing test failed - got basic description only")
                st.write("**Possible issues:**")
                st.write("- Image file not found at expected path")
                st.write("- Claude Vision API not working")
                st.write("- Image format not supported")
                
                if st.button("🔄 Force Reprocess All Images"):
                    force_image_reprocessing()
                
        except Exception as e:
            st.error(f"Test failed with error: {e}")
            st.write("Check the debug output for details")

def main_content():
    """Render main content area"""
    
    # Header
    st.markdown('<h1 class="main-header">📚 Multimodal RAG Assistant</h1>', unsafe_allow_html=True)
    
    # Check if API key is valid
    if not st.session_state.api_key_valid:
        st.warning("Please enter a valid Anthropic API key in the sidebar to begin.")
        st.markdown("""
        ### 🚀 Getting Started
        
        1. **Enter your Anthropic API Key** in the sidebar
        2. **Upload a document** (PDF, DOCX, or image)
        3. **Ask questions** about your document content
        
        ### ✨ Features
        - **Multimodal Analysis**: Text, tables, and images
        - **Smart Image Processing**: Only analyzes images when relevant to your query
        - **Context-Aware**: Understands document structure and relationships
        - **Multiple Formats**: Supports PDF, Word documents, and images
        """)
        return
    
    # File upload section
    st.markdown("## 📤 Upload Document")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'docx', 'doc', 'jpg', 'jpeg', 'png', 'tiff'],
        help="Upload PDF, Word document, or image file"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.info(f"📄 **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")
        
        with col2:
            if st.button("🔄 Process Document", type="primary"):
                try:
                    doc_metadata = process_document(uploaded_file, st.session_state.current_api_key)
                    st.success("Document processed successfully!")
                    display_processing_results(doc_metadata)
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
                    logger.error(f"Document processing error: {e}")
    
    # Query section
    if st.session_state.current_doc_id:
        st.markdown("---")
        st.markdown("## 💬 Ask Questions")
        
        # Current document info
        current_doc = st.session_state.documents[st.session_state.current_doc_id]
        st.info(f"📄 Currently querying: **{current_doc['filename']}**")
        
        # Query input
        col1, col2 = st.columns([4, 1])
        
        with col1:
            query = st.text_input(
                "Enter your question:",
                placeholder="What is the main topic of this document?",
                key="query_input"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            search_clicked = st.button("🔍 Search", type="primary")
        
        # Example queries - FIXED: Use unique keys and proper state handling
        with st.expander("💡 Example Queries"):
            example_queries = [
                "What is the main topic of this document?",
                "Summarize the key findings or conclusions", 
                "What data is shown in the tables?",
                "Describe any charts or graphs in the document",
                "What are the important metrics or numbers mentioned?",
                "Who are the key people or organizations mentioned?",
                "What recommendations or next steps are provided?"
            ]
            
            for i, example in enumerate(example_queries):
                if st.button(example, key=f"example_btn_{i}"):
                    # Store the selected query in session state
                    st.session_state.selected_example_query = example
                    st.rerun()
        
        # Handle selected example query
        if hasattr(st.session_state, 'selected_example_query'):
            query = st.session_state.selected_example_query
            # Clear the selected query to prevent reuse
            del st.session_state.selected_example_query
        
        # Process query - FIXED: Handle both manual input and example queries
        if (search_clicked and query) or (hasattr(st.session_state, 'selected_example_query')):
            # Use example query if available, otherwise use manual input
            query_to_process = getattr(st.session_state, 'selected_example_query', query)
            
            # Validate API key before processing
            if not st.session_state.get('current_api_key'):
                st.error("Please enter your Anthropic API key in the sidebar first.")
            else:
                try:
                    result = query_documents(query_to_process, st.session_state.current_doc_id)
                    if result:
                        display_query_result(result)
                        
                        # ENHANCED: Check if images were actually processed with Claude
                        if result['images_analyzed'] > 0:
                            st.success(f"🖼️ Successfully analyzed {result['images_analyzed']} images with Claude Vision!")
                            
                            # Show what Claude actually saw in the images
                            if result['processed_images']:
                                with st.expander("🔍 Detailed Image Analysis"):
                                    for img_id, description in result['processed_images'].items():
                                        st.markdown(f"**Image: {img_id}**")
                                        st.write(description)
                                        
                                        # Try to find and display the actual image
                                        if st.session_state.current_doc_id:
                                            current_doc = st.session_state.documents[st.session_state.current_doc_id]
                                            file_stem = current_doc.get('file_stem', '')
                                            
                                            # Parse image ID to get page and img numbers
                                            if 'page_' in img_id and '_img_' in img_id:
                                                try:
                                                    parts = img_id.split('_')
                                                    page_idx = parts.index('page')
                                                    img_idx = parts.index('img')
                                                    page_num = parts[page_idx + 1]
                                                    img_num = parts[img_idx + 1]
                                                    
                                                    expected_path = f"extracted_content/images/{file_stem}_page_{page_num}_img_{img_num}.png"
                                                    
                                                    if Path(expected_path).exists():
                                                        st.image(expected_path, caption=f"Claude analyzed this image: {img_id}")
                                                    else:
                                                        st.warning(f"Image file not found: {expected_path}")
                                                except (ValueError, IndexError):
                                                    st.warning(f"Could not parse image ID: {img_id}")
                        else:
                            st.info("ℹ️ No images were analyzed for this query. Try asking about charts, graphs, or visual elements.")
                            
                    # Clear the example query from session state
                    if hasattr(st.session_state, 'selected_example_query'):
                        del st.session_state.selected_example_query
                        
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    logger.error(f"Query processing error: {e}")
                    
                    # Enhanced error diagnostics for image issues
                    if "image" in str(e).lower():
                        st.error("🖼️ **Image Processing Error Detected**")
                        
                        if st.session_state.get('rag_system'):
                            st.write("Running image diagnostics...")
                            with st.expander("🔧 Image Diagnostics"):
                                try:
                                    # Run your debug function to show image path issues
                                    st.session_state.rag_system.debug_image_paths()
                                    st.info("Debug function executed - check console for details")
                                except Exception as debug_error:
                                    st.error(f"Debug failed: {debug_error}")
        
        # Show last result if available
        elif st.session_state.last_query_result and not query:
            st.markdown("### 📋 Last Query Result")
            display_query_result(st.session_state.last_query_result)
    
    # Debug panels - Show if requested
    if st.session_state.get('show_image_debug', False):
        debug_current_document_images()
        st.session_state.show_image_debug = False
    
    if st.session_state.get('show_image_test', False):
        test_image_processing_current_doc()
        st.session_state.show_image_test = False
        
    if st.session_state.get('show_force_reprocess', False):
        force_image_reprocessing()
        st.session_state.show_force_reprocess = False
    
    # Processing metrics (if available)
    if st.session_state.processing_metrics and st.session_state.debug_mode:
        st.markdown("---")
        st.markdown("## 📊 Processing Metrics")
        
        df_data = []
        for doc_id, metrics in st.session_state.processing_metrics.items():
            df_data.append({
                'Document': metrics['filename'],
                'Pages': metrics['pages'],
                'Tables': metrics['tables'],
                'Images': metrics['images'],
                'Chunks': metrics['chunk_count'],
                'Processing Time (s)': f"{metrics['processing_time']:.2f}",
                'File Size (KB)': f"{metrics['file_size'] / 1024:.1f}"
            })
        
        if df_data:
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)

def main():
    """Main application function - SAFE execution"""
    try:
        # Initialize session state first
        initialize_session_state()
        
        # Create layout with error handling
        try:
            sidebar_content()
        except Exception as e:
            st.error(f"Sidebar error: {e}")
            logger.error(f"Sidebar error: {e}")
        
        try:
            main_content()
        except Exception as e:
            st.error(f"Main content error: {e}")
            logger.error(f"Main content error: {e}")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.8em;'>
            Built with Streamlit • Powered by Anthropic Claude • Multimodal RAG System
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Critical application error: {e}")
        logger.error(f"Critical application error: {e}")
        
        # Emergency reset option
        if st.button("🔄 Emergency Reset Application"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()