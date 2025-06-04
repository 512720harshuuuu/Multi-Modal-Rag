# Multimodal RAG System

A comprehensive document processing and retrieval system that combines PDF extraction, multimodal analysis, and Retrieval-Augmented Generation (RAG) using Anthropic's Claude.

## 🎯 Project Overview

This system processes documents (PDFs, Word files, images) and creates an intelligent Q&A interface that can understand and analyze both text and visual content. It uses Claude for enhanced text processing, image analysis, and generating contextual responses.


## 📁 Project Structure

```
project/
├── document_extractor.py     # PDF/Document extraction and processing
├── anthropic_rag.py         # RAG system with Claude integration
├── app.py                   # Streamlit web interface
├── requirements.txt         # Python dependencies
├── extracted_content/       # Output directory
│   ├── images/             # Extracted images (filename_page_X_img_Y.png)
│   └── JSON/               # Processed document JSON files
└── PDFS/                   # Input documents directory
```
-- Please make sure you create extracted_content/ in the structure shown above in your local repository

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python3.12 -m venv pdf_extractor_venv

# Activate virtual environment
# On macOS/Linux:
source pdf_extractor_venv/bin/activate
# On Windows:
pdf_extractor_venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Required Libraries:**
```
streamlit
anthropic
sentence-transformers
PyMuPDF
pillow
pytesseract
opencv-python
numpy
pandas
python-docx
pdfplumber
pathlib
python-dotenv
```

### 3. API Key Setup

Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

Or set it directly in the Streamlit interface.

## 🧪 Testing Individual Components

### 1. Test Document Extraction

```bash
python pdf_extractor.py
```

**What it does:**
- Processes a sample PDF from `PDFS/` directory
- Extracts text, tables, and images
- Saves results to `extracted_content/JSON/`
- Images saved to `extracted_content/images/`

**Expected Output:**
- JSON file with structured document content
- Individual image files with naming pattern: `filename_page_X_img_Y.png`
- Console output showing extraction metrics

### 2. Test RAG System

```bash
python multi_modal_rag.py
```

**What it does:**
- Loads a processed JSON file
- Creates embeddings and chunks
- Tests Claude image processing
- Runs sample queries

**Expected Output:**
- Chunk processing statistics
- Image path verification
- Sample Q&A responses
- Debug information about image locations

### 3. Run Full Application

```bash
streamlit run app.py
```

**What it does:**
- Launches web interface on `http://localhost:8501`
- Provides document upload and processing
- Interactive Q&A interface
- Real-time processing feedback

## 📊 Results and Output

### Extraction Results
- **Location:** `extracted_content/JSON/{filename}.json`
- **Content:** Structured document data with text, tables, images
- **Format:** JSON with pages, metadata, and content hierarchy

### Image Processing
- **Location:** `extracted_content/images/`
- **Naming:** `{document_name}_page_{page_number}_img_{image_number}.png`
- **Example:** `report-2024_page_0_img_0.png`

### Query Results
- **Interface:** Streamlit web app
- **Features:** 
  - Real-time responses
  - Image analysis when relevant
  - Source chunk references
  - Processing metrics

## 🔧 Configuration Options

### Document Extractor Config
```python
config = ExtractionConfig(
    use_ai_service=True,           # Enable Claude for table/image analysis
    ai_api_key="your_key",         # Anthropic API key
    output_dir="extracted_content", # Output directory
    save_images=True,              # Save extracted images
    table_extraction_method="auto", # Table extraction method
    image_dpi=300                  # Image resolution
)
```

### RAG System Config
```python
rag = AnthropicOnlyRAG(
    anthropic_api_key="your_key",
    embedding_model="all-MiniLM-L6-v2"  # Local embedding model
)
```

## 🐛 Debugging and Troubleshooting

### Common Issues

1. **Image Not Found Errors**
   - Check `extracted_content/images/` directory exists
   - Verify filename pattern matches: `{name}_page_{X}_img_{Y}.png`
   - Use debug tools in Streamlit app

2. **API Key Issues**
   - Verify Anthropic API key is valid
   - Check `.env` file or Streamlit input
   - Ensure sufficient API credits

3. **Processing Failures**
   - Check file format compatibility (PDF, DOCX, images)
   - Verify file isn't corrupted or password-protected
   - Check console output for detailed errors

### Debug Tools

The Streamlit app includes built-in debugging:
- **Debug Mode:** Toggle in sidebar for detailed output
- **Image Path Debug:** Verify image file locations
- **Force Reprocess:** Re-analyze images with Claude
- **Processing Metrics:** Performance and quality tracking

## 🏗️ Architecture Overview

### 1. Document Processing Pipeline
```
Upload → Extract → Structure → Enhance → Chunk → Embed → Index
```

### 2. Query Processing Pipeline
```
Query → Retrieve → Analyze Images (if needed) → Generate Response
```

### 3. Key Components
- **Extractor:** Multi-format document processing
- **RAG System:** Intelligent retrieval and generation
- **Web Interface:** User-friendly interaction layer

## 📝 Usage Examples

### Basic Document Processing
```python
from document_extractor import DocumentExtractor, ExtractionConfig

config = ExtractionConfig(use_ai_service=True, ai_api_key="your_key")
extractor = DocumentExtractor(config)
result = extractor.extract_document("document.pdf")
```

### RAG Query
```python
from anthropic_rag import AnthropicOnlyRAG

rag = AnthropicOnlyRAG("your_api_key")
chunks = rag.process_json_document(extraction_result)
response = rag.query_with_selective_image_processing("What does the chart show?")
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

## 📄 License

[Add your license information here]

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Enable debug mode for detailed logs
3. Review console output for error details
4. Create GitHub issue with reproduction steps
