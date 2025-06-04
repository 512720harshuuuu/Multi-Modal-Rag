import json
import anthropic
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import base64
from pathlib import Path
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import hashlib
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

@dataclass
class ProcessedChunk:
    chunk_id: str
    content: str
    context_enhanced_content: str
    embedding: List[float]
    chunk_type: str
    page: int
    related_images: List[str]
    metadata: Dict[str, Any]

class AnthropicOnlyRAG:
    def __init__(self, anthropic_api_key: str, embedding_model: str = "all-MiniLM-L6-v2"):
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
        
        # Use sentence-transformers for embeddings (free, local)
        self.embedding_model = SentenceTransformer(embedding_model)
        
        self.chunks = []
        self.image_descriptions = {}
        
        # Query patterns that suggest image relevance
        self.image_trigger_patterns = [
            r'\b(chart|graph|diagram|figure|image|visual|table|data)\b',
            r'\b(show|display|illustrate|demonstrate)\b',
            r'\b(trend|pattern|comparison|structure)\b',
            r'\b(layout|design|format|appearance)\b'
        ]
    
    def process_json_document(self, json_data: Dict[str, Any]) -> List[ProcessedChunk]:
        """Main processing function using Anthropic for context enhancement"""
        
        # Extract and store document filename for image path construction
        self.set_document_filename(json_data)
        
        document_context = self._extract_document_context(json_data)
        
        for page in json_data['pages']:
            page_context = self._extract_page_context(page, document_context)
            
            # Process text sections with Claude-enhanced context
            text_chunks = self._process_text_sections_with_claude(page, page_context)
            
            # Process tables with Claude-enhanced context  
            table_chunks = self._process_tables_with_claude(page, page_context)
            
            # Create image-aware chunks
            image_chunks = self._process_images(page, page_context)
            
            self.chunks.extend(text_chunks + table_chunks + image_chunks)
        
        return self.chunks
    
    def _extract_document_context(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract high-level document context"""
        metadata = json_data.get('document_metadata', {})
        summary = json_data.get('summary', {})
        
        return {
            'document_type': '',
            'filename': metadata.get('filename', ''),
            'main_topics': summary.get('main_topics', []),
            'total_pages': metadata.get('total_pages', 0),
            'key_metrics': summary.get('key_metrics', [])
        }
    
    def _extract_page_context(self, page: Dict, doc_context: Dict) -> Dict[str, Any]:
        """Extract page-level context"""
        title = page.get('text_content', {}).get('title', '')
        content_type = page.get('content_type', '')
        page_num = page.get('page_number', 0)
        
        return {
            'page_number': page_num,
            'section_title': title,
            'content_type': content_type,
            'has_images': len(page.get('images', [])) > 0,
            'has_tables': len(page.get('tables', [])) > 0,
            'doc_context': doc_context
        }
    
    def _process_text_sections_with_claude(self, page: Dict, page_context: Dict) -> List[ProcessedChunk]:
        """Process text sections using Claude for context enhancement"""
        chunks = []
        text_content = page.get('text_content', {})
        sections = text_content.get('sections', [])
        
        # Batch process sections for efficiency
        section_batch = []
        for i, section in enumerate(sections):
            subtitle = section.get('subtitle', '')
            content = section.get('content', '')
            
            if not content or len(content.strip()) < 50:
                continue
                
            section_batch.append({
                'index': i,
                'subtitle': subtitle,
                'content': content
            })
        
        if not section_batch:
            return chunks
        
        # Use Claude to enhance multiple sections at once
        enhanced_sections = self._batch_enhance_with_claude(section_batch, page_context)
        
        for section_data, enhanced_content in zip(section_batch, enhanced_sections):
            # Generate embedding for enhanced content
            embedding = self._get_embedding(enhanced_content)
            
            # Identify related images
            related_images = self._find_related_images(page, section_data['content'])
            
            chunk = ProcessedChunk(
                chunk_id=f"text_{page_context['page_number']}_{section_data['index']}",
                content=section_data['content'],
                context_enhanced_content=enhanced_content,
                embedding=embedding.tolist(),
                chunk_type='text_section',
                page=page_context['page_number'],
                related_images=related_images,
                metadata={
                    'subtitle': section_data['subtitle'],
                    'section_title': page_context['section_title'],
                    'content_type': page_context['content_type'],
                    'image_count': len(related_images)
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _batch_enhance_with_claude(self, sections: List[Dict], page_context: Dict) -> List[str]:
        """Use Claude to enhance sections with contextual information"""
        
        # Build context prompt
        doc_context = page_context['doc_context']
        context_prompt = f"""
You are enhancing text chunks for a RAG system. Add contextual information that will help with retrieval.

Document: {doc_context['filename']}
Document Type: {doc_context['document_type']}
Page: {page_context['page_number']} of {doc_context['total_pages']}
Section: {page_context['section_title']}
Content Type: {page_context['content_type']}
Main Topics: {', '.join(doc_context['main_topics'])}

For each section below, prepend appropriate context that explains:
1. What document this is from
2. What page/section this appears in
3. How it relates to the overall document structure
4. Any relevant metadata about the content

Make the context concise but informative for search retrieval.

Sections to enhance:
"""
        
        for i, section in enumerate(sections):
            context_prompt += f"\n--- Section {i+1} ---\n"
            if section['subtitle']:
                context_prompt += f"Subtitle: {section['subtitle']}\n"
            context_prompt += f"Content: {section['content']}\n"
        
        context_prompt += "\nReturn the enhanced sections in the same order, separated by '---SECTION_BREAK---'"
        
        try:
            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",  # Latest Claude Sonnet 4
                max_tokens=4000,
                messages=[{
                    "role": "user",
                    "content": context_prompt
                }]
            )
            
            enhanced_text = response.content[0].text
            enhanced_sections = enhanced_text.split('---SECTION_BREAK---')
            
            # Clean up and ensure we have the right number
            enhanced_sections = [s.strip() for s in enhanced_sections if s.strip()]
            
            # If something went wrong, fall back to basic enhancement
            if len(enhanced_sections) != len(sections):
                return [self._basic_enhance(s, page_context) for s in sections]
            
            return enhanced_sections
            
        except Exception as e:
            print(f"Claude enhancement failed: {e}, falling back to basic enhancement")
            return [self._basic_enhance(s, page_context) for s in sections]
    
    def _basic_enhance(self, section: Dict, page_context: Dict) -> str:
        """Fallback enhancement if Claude fails"""
        doc_context = page_context['doc_context']
        
        context_prefix = (
            f"Document: {doc_context['filename']}. "
            f"Page {page_context['page_number']} of {doc_context['total_pages']}. "
            f"Section: {page_context['section_title']}. "
        )
        
        if section['subtitle']:
            context_prefix += f"Subsection: {section['subtitle']}. "
        
        return context_prefix + section['content']
    
    def _process_tables_with_claude(self, page: Dict, page_context: Dict) -> List[ProcessedChunk]:
        """Process tables using Claude for natural language conversion"""
        chunks = []
        tables = page.get('tables', [])
        
        for i, table in enumerate(tables):
            # Use Claude to convert table to natural language
            table_text = self._table_to_text_with_claude(table, page_context)
            
            # Generate embedding
            embedding = self._get_embedding(table_text)
            
            chunk = ProcessedChunk(
                chunk_id=f"table_{page_context['page_number']}_{i}",
                content=table_text,
                context_enhanced_content=table_text,  # Already enhanced by Claude
                embedding=embedding.tolist(),
                chunk_type='table',
                page=page_context['page_number'],
                related_images=[],
                metadata={
                    'table_id': table.get('table_id', f'table_{i}'),
                    'headers': table.get('headers', []),
                    'row_count': len(table.get('rows', [])),
                    'raw_table': table
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _table_to_text_with_claude(self, table: Dict, page_context: Dict) -> str:
        """Use Claude to convert table to natural language with context"""
        
        headers = table.get('headers', [])
        rows = table.get('rows', [])
        
        if not rows:
            return f"Empty table with headers: {', '.join(headers)}"
        
        # Prepare table data for Claude
        table_prompt = f"""
Convert this table data into a natural language description that preserves all important information for search and retrieval.

Context: This is from {page_context['doc_context']['filename']}, page {page_context['page_number']}, section "{page_context['section_title']}".

Table Headers: {', '.join(headers)}

Table Data:
"""
        
        for i, row in enumerate(rows[:10]):  # Limit to first 10 rows for token efficiency
            row_data = []
            for header, value in row.items():
                if value and str(value).strip():
                    row_data.append(f"{header}: {value}")
            if row_data:
                table_prompt += f"Row {i+1}: {' | '.join(row_data)}\n"
        
        if len(rows) > 10:
            table_prompt += f"... and {len(rows) - 10} more rows\n"
        
        table_prompt += "\nCreate a comprehensive description that includes context about what this table shows and its key data points."
        
        try:
            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",  # Latest Claude Sonnet 4
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": table_prompt
                }]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            print(f"Claude table conversion failed: {e}, using basic conversion")
            return self._basic_table_to_text(table)
    
    def _basic_table_to_text(self, table: Dict) -> str:
        """Fallback table conversion"""
        headers = table.get('headers', [])
        rows = table.get('rows', [])
        
        text_parts = [f"Table with columns: {', '.join(headers)}."]
        
        for i, row in enumerate(rows[:5]):
            row_text = []
            for header, value in row.items():
                if value and str(value).strip():
                    row_text.append(f"{header}: {value}")
            
            if row_text:
                text_parts.append(f"Row {i+1}: {', '.join(row_text)}")
        
        if len(rows) > 5:
            text_parts.append(f"... and {len(rows) - 5} more rows")
        
        return ' '.join(text_parts)
    
    def _process_images(self, page: Dict, page_context: Dict) -> List[ProcessedChunk]:
        """Create image placeholder chunks with correct path handling"""
        chunks = []
        images = page.get('images', [])
        
        for i, image in enumerate(images):
            image_desc = image.get('description', 'image')
            size = image.get('estimated_size', 'unknown size')
            image_id = image.get('image_id', f'img_{page_context["page_number"]}_{i}')
            
            # Store the original path and also construct expected path
            original_path = image.get('file_path', '')
            
            placeholder_content = (
                f"Image {i+1} from {page_context['doc_context']['filename']}, "
                f"page {page_context['page_number']}: {image_desc} ({size}). "
                f"This image may contain charts, diagrams, or visual information "
                f"relevant to {page_context['section_title']}."
            )
            
            embedding = self._get_embedding(placeholder_content)
            
            chunk = ProcessedChunk(
                chunk_id=f"image_{page_context['page_number']}_{i}",
                content=placeholder_content,
                context_enhanced_content=placeholder_content,
                embedding=embedding.tolist(),
                chunk_type='image_placeholder',
                page=page_context['page_number'],
                related_images=[image_id],
                metadata={
                    'image_path': original_path,
                    'image_id': image_id,
                    'description': image.get('description', ''),
                    'size': image.get('estimated_size', ''),
                    'processed': False,
                    # Add debug info
                    'width': image.get('metadata', {}).get('width'),
                    'height': image.get('metadata', {}).get('height'),
                    'source_document': image.get('metadata', {}).get('source_document')
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using sentence-transformers"""
        return self.embedding_model.encode(text, convert_to_numpy=True)
    
    def _find_related_images(self, page: Dict, content: str) -> List[str]:
        """Identify images related to text content"""
        images = page.get('images', [])
        related = []
        
        visual_keywords = ['chart', 'graph', 'figure', 'diagram', 'table', 'image']
        content_lower = content.lower()
        
        if any(keyword in content_lower for keyword in visual_keywords):
            related = [img.get('image_id', '') for img in images]
        
        return related
    
    def query_with_selective_image_processing(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Main query function with Claude-powered analysis"""
        
        # Check if query suggests need for visual analysis
        needs_images = self._query_suggests_images(query)
        
        # Get text-based retrieval
        query_embedding = self._get_embedding(query)
        relevant_chunks = self._retrieve_similar_chunks(query_embedding, top_k)
        
        # Process images if needed
        processed_images = {}
        if needs_images:
            image_chunks = [c for c in relevant_chunks if c.related_images]
            for chunk in image_chunks:
                for image_id in chunk.related_images:
                    if image_id not in self.image_descriptions:
                        description = self._process_image_with_claude(chunk, image_id)
                        if description:
                            self.image_descriptions[image_id] = description
                            processed_images[image_id] = description
        
        # Generate comprehensive answer using Claude
        answer = self._generate_answer_with_claude(query, relevant_chunks, processed_images)
        
        return {
            'query': query,
            'answer': answer,
            'relevant_chunks': relevant_chunks,
            'processed_images': processed_images,
            'images_analyzed': len(processed_images)
        }
    
    def _query_suggests_images(self, query: str) -> bool:
        """Determine if query likely needs visual analysis"""
        query_lower = query.lower()
        
        for pattern in self.image_trigger_patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False
    
    def _retrieve_similar_chunks(self, query_embedding: np.ndarray, top_k: int) -> List[ProcessedChunk]:
        """Cosine similarity retrieval"""
        similarities = []
        
        for chunk in self.chunks:
            chunk_embedding = np.array(chunk.embedding)
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, chunk_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
            )
            similarities.append((similarity, chunk))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in similarities[:top_k]]
    
    def _process_image_with_claude(self, chunk: ProcessedChunk, image_id: str) -> Optional[str]:
        """Process image using Claude Vision"""
        # Get the correct image path from metadata or construct it
        image_path = self._get_correct_image_path(chunk, image_id)
        
        if not image_path or not Path(image_path).exists():
            print(f"Image not found: {image_path}")
            return None
        
        try:
            # Read and encode image
            with open(image_path, 'rb') as img_file:
                image_data = base64.b64encode(img_file.read()).decode()
            
            # All extracted images are PNG format based on your naming convention
            media_type = 'image/png'
            
            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",  # Latest Claude Sonnet 4 with vision
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": f"Analyze this image. This image appears in the context: {chunk.content[:200]}... Describe what you see, any data, charts, trends, or key information visible. Focus on details that would help answer questions about the document content."
                        }
                    ]
                }]
            )
            
            return response.content[0].text
            
        except Exception as e:
            print(f"Error processing image {image_id}: {e}")
            return None
    
    def _get_correct_image_path(self, chunk: ProcessedChunk, image_id: str) -> Optional[str]:
        """Get the correct image path based on your naming convention"""
        
        # Method 1: Try the stored path from metadata
        stored_path = chunk.metadata.get('image_path', '')
        if stored_path and Path(stored_path).exists():
            return stored_path
        
        # Method 2: Construct path based on document filename + page + image pattern
        # Get the document filename from chunk metadata
        doc_filename = None
        
        # Try to get from chunk metadata or global context
        if hasattr(self, 'document_filename'):
            doc_filename = self.document_filename
        else:
            # Extract from stored path or try to infer
            if stored_path:
                # Extract filename from path like: extracted_content/images/adm-2022-short_page_0_img_0.png
                path_parts = Path(stored_path).stem  # removes .png
                # Split by underscore and reconstruct base filename
                parts = path_parts.split('_')
                if len(parts) >= 3:  # should have at least filename_page_X_img_Y
                    # Find where 'page' appears and take everything before it
                    try:
                        page_idx = parts.index('page')
                        doc_filename = '_'.join(parts[:page_idx])
                    except ValueError:
                        # Fallback: assume first parts are filename
                        doc_filename = '_'.join(parts[:-3])  # Remove last 3 parts (page_X_img_Y)
        
        if not doc_filename:
            print(f"Could not determine document filename for image {image_id}")
            return self._fallback_image_search(image_id)
        
        # Method 2a: Standard naming pattern
        base_image_dir = Path("extracted_content/images")
        
        # Extract page and image numbers from image_id
        # image_id should be like: page_0_img_0, page_1_img_2, etc.
        page_num, img_num = self._extract_page_img_numbers(image_id)
        
        if page_num is not None and img_num is not None:
            # Try different naming patterns
            naming_patterns = [
                f"{doc_filename}_page_{page_num}_img_{img_num}",  # adm-2022-short_page_0_img_0
                f"{doc_filename}_page_{page_num}_image_{img_num}", # adm-2022-short_page_0_image_0
                f"{doc_filename}_{page_num}_{img_num}",            # adm-2022-short_0_0
                f"{doc_filename}_p{page_num}_i{img_num}",          # adm-2022-short_p0_i0
            ]
            
            for pattern in naming_patterns:
                for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                    img_path = base_image_dir / f"{pattern}{ext}"
                    if img_path.exists():
                        return str(img_path)
        
        # Method 3: Fallback search
        return self._fallback_image_search(image_id)
    
    def _extract_page_img_numbers(self, image_id: str) -> tuple:
        """Extract page and image numbers from image_id"""
        
        # Try different patterns
        patterns = [
            r'page_(\d+)_img_(\d+)',      # page_0_img_0
            r'page_(\d+)_image_(\d+)',    # page_0_image_0  
            r'p(\d+)_i(\d+)',             # p0_i0
            r'(\d+)_(\d+)',               # 0_0
        ]
        
        for pattern in patterns:
            match = re.search(pattern, image_id)
            if match:
                return int(match.group(1)), int(match.group(2))
        
        # If image_id is just the full filename, try to parse it
        if '_page_' in image_id and '_img_' in image_id:
            try:
                parts = image_id.split('_')
                page_idx = parts.index('page')
                img_idx = parts.index('img')
                if page_idx + 1 < len(parts) and img_idx + 1 < len(parts):
                    page_num = int(parts[page_idx + 1])
                    img_num = int(parts[img_idx + 1])
                    return page_num, img_num
            except (ValueError, IndexError):
                pass
        
        return None, None
    
    def _fallback_image_search(self, image_id: str) -> Optional[str]:
        """Fallback method to search for images"""
        
        base_image_dir = Path("extracted_content/images")
        
        # Try exact match with extensions
        for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
            img_path = base_image_dir / f"{image_id}{ext}"
            if img_path.exists():
                return str(img_path)
        
        # Try partial matching
        if base_image_dir.exists():
            for img_file in base_image_dir.glob("*"):
                if img_file.is_file() and image_id in img_file.stem:
                    return str(img_file)
        
        # Try other possible base directories
        for potential_base in ["./extracted_content/images", "./images", "."]:
            base_path = Path(potential_base)
            if base_path.exists():
                for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                    img_path = base_path / f"{image_id}{ext}"
                    if img_path.exists():
                        return str(img_path)
        
        print(f"Could not find image file for {image_id}")
        print(f"Searched in: {base_image_dir}")
        print(f"Looking for files like: {image_id}.png")
        
        return None
    
    def set_document_filename(self, json_data: Dict[str, Any]):
        """Extract and store document filename for image path construction"""
        filename = json_data.get('document_metadata', {}).get('filename', '')
        if filename:
            # Remove extension to get base name
            self.document_filename = Path(filename).stem
        else:
            self.document_filename = None
    
    def debug_image_paths(self):
        """Debug function to check image availability"""
        print("=== IMAGE PATH DEBUGGING ===")
        
        # Check base directories
        possible_dirs = [
            "extracted_content/images",
            "./extracted_content/images", 
            "images",
            "./images",
            "."
        ]
        
        print("Checking directories:")
        for dir_path in possible_dirs:
            path_obj = Path(dir_path)
            exists = path_obj.exists()
            print(f"  {dir_path}: {'EXISTS' if exists else 'NOT FOUND'}")
            
            if exists and path_obj.is_dir():
                # List image files
                image_files = list(path_obj.glob("*.png")) + list(path_obj.glob("*.jpg")) + list(path_obj.glob("*.jpeg"))
                print(f"    Image files found: {len(image_files)}")
                for img_file in image_files[:5]:  # Show first 5
                    print(f"      {img_file.name}")
                if len(image_files) > 5:
                    print(f"      ... and {len(image_files) - 5} more")
        
        print("\nImage chunks and their expected paths:")
        for chunk in self.chunks:
            if chunk.chunk_type == 'image_placeholder':
                image_id = chunk.related_images[0] if chunk.related_images else "unknown"
                stored_path = chunk.metadata.get('image_path', 'No path stored')
                print(f"  Image ID: {image_id}")
                print(f"    Stored path: {stored_path}")
                
                # Show document filename being used
                doc_filename = getattr(self, 'document_filename', 'Not set')
                print(f"    Document filename: {doc_filename}")
                
                # Extract page/img numbers
                page_num, img_num = self._extract_page_img_numbers(image_id)
                print(f"    Extracted page: {page_num}, img: {img_num}")
                
                # Show expected paths
                if doc_filename and doc_filename != 'Not set' and page_num is not None and img_num is not None:
                    expected_patterns = [
                        f"{doc_filename}_page_{page_num}_img_{img_num}.png",
                        f"{doc_filename}_page_{page_num}_image_{img_num}.png",
                    ]
                    print(f"    Expected patterns: {expected_patterns}")
                
                # Try to find the actual path
                actual_path = self._get_correct_image_path(chunk, image_id)
                print(f"    Found at: {actual_path if actual_path else 'NOT FOUND'}")
                print()
    
    def test_image_processing(self, image_id: str = None):
        """Test image processing for a specific image"""
        
        if not image_id:
            # Find first image chunk
            image_chunks = [c for c in self.chunks if c.chunk_type == 'image_placeholder']
            if not image_chunks:
                print("No image chunks found!")
                return
            
            chunk = image_chunks[0]
            image_id = chunk.related_images[0] if chunk.related_images else None
        else:
            # Find chunk with this image_id
            chunk = None
            for c in self.chunks:
                if c.chunk_type == 'image_placeholder' and image_id in c.related_images:
                    chunk = c
                    break
            
            if not chunk:
                print(f"No chunk found for image_id: {image_id}")
                return
        
        print(f"Testing image processing for: {image_id}")
        
        # Try to process the image
        description = self._process_image_with_claude(chunk, image_id)
        
        if description:
            print("SUCCESS! Image processed.")
            print(f"Description: {description[:200]}...")
        else:
            print("FAILED to process image.")
            print("Running debug to find the issue...")
            
            # Debug the specific image
            image_path = self._get_correct_image_path(chunk, image_id)
            if image_path:
                print(f"Found image at: {image_path}")
                if Path(image_path).exists():
                    file_size = Path(image_path).stat().st_size
                    print(f"File size: {file_size} bytes")
                    print("File exists but Claude processing failed - check API key or file format")
                else:
                    print("Path found but file doesn't exist!")
            else:
                print("Could not determine correct image path")
        
        return description
    
    def test_image_naming_convention(self, json_filename: str):
        """Test the image naming convention with your actual files"""
        print(f"Testing image naming for JSON file: {json_filename}")
        
        # Extract base filename
        base_name = Path(json_filename).stem
        print(f"Base filename: {base_name}")
        
        # Check what images exist
        image_dir = Path("extracted_content/images")
        if image_dir.exists():
            print(f"\nImages found in {image_dir}:")
            image_files = sorted(image_dir.glob(f"{base_name}*"))
            for img_file in image_files:
                print(f"  {img_file.name}")
                
                # Try to parse the naming pattern
                stem = img_file.stem
                if '_page_' in stem and '_img_' in stem:
                    parts = stem.split('_')
                    try:
                        page_idx = parts.index('page')
                        img_idx = parts.index('img')
                        if page_idx + 1 < len(parts) and img_idx + 1 < len(parts):
                            page_num = parts[page_idx + 1]
                            img_num = parts[img_idx + 1]
                            print(f"    -> Page: {page_num}, Image: {img_num}")
                    except (ValueError, IndexError):
                        print(f"    -> Could not parse naming pattern")
        else:
            print(f"Image directory {image_dir} not found!")
        
        return image_files if image_dir.exists() else []
    
    def _generate_answer_with_claude(self, query: str, chunks: List[ProcessedChunk], 
                                   image_descriptions: Dict[str, str]) -> str:
        """Generate comprehensive answer using Claude"""
        
        # Prepare context from chunks
        context_parts = []
        for chunk in chunks:
            context_parts.append(f"Content: {chunk.context_enhanced_content}")
            
            # Add image descriptions if available
            for image_id in chunk.related_images:
                if image_id in image_descriptions:
                    context_parts.append(f"Related Image Analysis: {image_descriptions[image_id]}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""
Based on the following content from an document, answer the user's question comprehensively.

User Question: {query}

Relevant Content:
{context}

Please provide a detailed, accurate answer based on the provided content. If the content includes data, charts, or visual information, incorporate that into your response. If you cannot answer based on the provided content, say so clearly.
"""
        
        try:
            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",  # Latest Claude Sonnet 4
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            return response.content[0].text
            
        except Exception as e:
            return f"Error generating answer: {e}"

# Usage example
def main():
    # Initialize with only Anthropic API key
    rag = AnthropicOnlyRAG(os.getenv("ANTHROPIC_API_KEY"))
    
    # Test image naming convention first
    json_filename = "extracted_content/JSON/adm.json"  # Replace with your actual JSON filename
    print("="*60)
    rag.test_image_naming_convention(json_filename)
    
    # Load your JSON document
    with open(json_filename, 'r') as f:
        json_data = json.load(f)
    
    # Process document
    chunks = rag.process_json_document(json_data)
    print(f"\nProcessed {len(chunks)} chunks")
    
    # Debug image paths
    print("\n" + "="*50)
    rag.debug_image_paths()
    
    # Test image processing
    print("\n" + "="*50)
    print("Testing image processing...")
    rag.test_image_processing()
    
    # Example queries
    queries = [
            "Who is the contractor, what does he do, where does he work, which company is he working for, how many hours did he work on 5/27/2025"
    ]
    
    for query in queries:
        print(f"\n{'='*50}")
        result = rag.query_with_selective_image_processing(query)
        print(f"Q: {query}")
        print(f"A: {result['answer']}")
        print(f"Images analyzed: {result['images_analyzed']}")
        if result['processed_images']:
            print("Image descriptions:")
            for img_id, desc in result['processed_images'].items():
                print(f"  {img_id}: {desc[:100]}...")

if __name__ == "__main__":
    main()
