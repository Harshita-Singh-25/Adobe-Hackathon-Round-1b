import os
import re
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict, Counter
import fitz  # PyMuPDF

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class TextElement:
    """Enhanced text element with comprehensive metadata"""
    text: str
    page: int
    bbox: List[float]
    font_size: float
    font_name: str
    is_bold: bool
    is_italic: bool
    color: int
    spacing_above: float
    spacing_below: float
    column: int
    block_type: str
    confidence: float = 0.0

class PDFStructureExtractor:
    """Robust PDF structure extractor that handles diverse document types"""
    
    def __init__(self):
        self.min_heading_score = 0.4
        self.semantic_patterns = {
            'numbered': [
                (r'^\d+\.\s+[A-Z]', 0.9, 'H1'),
                (r'^\d+\.\d+\s+[A-Z]', 0.8, 'H2'),
                (r'^\d+\.\d+\.\d+\s+[A-Z]', 0.7, 'H3'),
                (r'^[A-Z]\.\s+[A-Z]', 0.6, 'H2'),
                (r'^\([a-z]\)\s+[A-Z]', 0.5, 'H3')
            ],
            'structural': [
                (r'^(Chapter|Section|Part)\s+\d+', 0.9, 'H1'),
                (r'^(Appendix|Annex)\s+[A-Z]', 0.8, 'H1'),
                (r'^(Abstract|Introduction|Conclusion|Summary)$', 0.8, 'H1'),
                (r'^(Background|Methodology|Results|Discussion)$', 0.7, 'H1'),
                (r'^(Overview|Objectives|Requirements)$', 0.6, 'H2')
            ],
            'business': [
                (r'^(Executive Summary|Financial|Market|Strategy)', 0.8, 'H1'),
                (r'^(Revenue|Profit|Cost|Budget)', 0.7, 'H2'),
                (r'^(Q[1-4]|Quarter)', 0.6, 'H2')
            ],
            'guide_content': [
                (r'^(Hotels?|Restaurants?|Activities|Transportation)', 0.7, 'H2'),
                (r'^(Budget|Luxury|Mid-range)', 0.6, 'H2'),
                (r'^(Tips|Recommendations|Guide)', 0.5, 'H3')
            ]
        }
    
    def extract_outline(self, pdf_path: str) -> Dict[str, Any]:
        """Main extraction function for Round 1A compatibility"""
        try:
            with fitz.open(pdf_path) as doc:
                # Extract and analyze text elements
                elements = self._extract_text_elements(doc)
                if not elements:
                    return {"title": "", "outline": []}
                
                # Detect document characteristics
                doc_analysis = self._analyze_document_characteristics(elements)
                
                # Extract title
                title = self._extract_title(elements, doc_analysis)
                
                # Identify headings using adaptive approach
                headings = self._identify_headings_adaptive(elements, doc_analysis)
                
                # Build hierarchical outline
                outline = self._build_hierarchical_outline(headings, doc_analysis)
                
                return {
                    "title": title,
                    "outline": outline
                }
        except Exception as e:
            logger.error(f"Error extracting PDF structure: {e}")
            return {"title": "", "outline": []}
    
    def _extract_text_elements(self, doc: fitz.Document) -> List[TextElement]:
        """Enhanced text extraction with multi-column support and spacing analysis"""
        elements = []
        prev_page_bottom = {}
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            # Group blocks by columns
            column_groups = self._group_blocks_by_columns(blocks, page.rect.width)
            
            for col_idx, col_blocks in enumerate(column_groups):
                # Sort blocks vertically within column
                col_blocks.sort(key=lambda b: b["bbox"][1])
                
                prev_bottom = prev_page_bottom.get((page_num, col_idx), 0)
                
                for block in col_blocks:
                    if block["type"] != 0 or "lines" not in block:
                        continue
                    
                    for line in block["lines"]:
                        element = self._create_text_element(
                            line, page_num, col_idx, prev_bottom, page.rect
                        )
                        
                        if element and self._is_valid_element(element):
                            elements.append(element)
                            prev_bottom = element.bbox[3]
                
                prev_page_bottom[(page_num, col_idx)] = prev_bottom
        
        # Filter repetitive headers/footers
        elements = self._filter_repetitive_elements(elements, len(doc))
        
        return elements
    
    def _group_blocks_by_columns(self, blocks: List[Dict], page_width: float) -> List[List[Dict]]:
        """Intelligent column detection and grouping"""
        text_blocks = [b for b in blocks if b["type"] == 0 and b.get("lines")]
        
        if not text_blocks:
            return []
        
        # Calculate x-positions for column detection
        x_positions = [block["bbox"][0] for block in text_blocks]
        
        # Use clustering to identify columns
        if len(set(x_positions)) <= 2:
            return [text_blocks]  # Single column
        
        # Simple clustering by x-coordinate
        x_positions.sort()
        columns = []
        current_column = []
        current_x = x_positions[0]
        threshold = page_width * 0.15  # 15% of page width
        
        for block in text_blocks:
            block_x = block["bbox"][0]
            
            if abs(block_x - current_x) > threshold:
                if current_column:
                    columns.append(current_column)
                current_column = [block]
                current_x = block_x
            else:
                current_column.append(block)
        
        if current_column:
            columns.append(current_column)
        
        return columns if len(columns) > 1 else [text_blocks]
    
    def _create_text_element(self, line: Dict, page_num: int, col_idx: int, 
                           prev_bottom: float, page_rect: fitz.Rect) -> Optional[TextElement]:
        """Create enhanced text element with comprehensive metadata"""
        spans = line.get("spans", [])
        if not spans:
            return None
        
        # Aggregate span information
        text_parts = []
        font_sizes = []
        font_names = []
        colors = []
        is_bold = False
        is_italic = False
        
        for span in spans:
            text = span.get("text", "").strip()
            if text:
                text_parts.append(text)
                font_sizes.append(span.get("size", 12))
                font_names.append(span.get("font", ""))
                colors.append(span.get("color", 0))
                
                flags = span.get("flags", 0)
                if flags & 4:  # Bold
                    is_bold = True
                if flags & 2:  # Italic
                    is_italic = True
        
        if not text_parts:
            return None
        
        full_text = " ".join(text_parts).strip()
        avg_font_size = np.mean(font_sizes) if font_sizes else 12
        dominant_font = max(set(font_names), key=font_names.count) if font_names else ""
        
        # Calculate spacing
        bbox = line.get("bbox", [0, 0, 0, 0])
        spacing_above = max(0, bbox[1] - prev_bottom) if prev_bottom > 0 else 0
        
        return TextElement(
            text=full_text,
            page=page_num + 1,  # 1-based
            bbox=bbox,
            font_size=round(avg_font_size, 1),
            font_name=dominant_font,
            is_bold=is_bold,
            is_italic=is_italic,
            color=colors[0] if colors else 0,
            spacing_above=spacing_above,
            spacing_below=0,  # Will be calculated later
            column=col_idx,
            block_type=self._classify_block_type(full_text, bbox, page_rect)
        )
    
    def _classify_block_type(self, text: str, bbox: List[float], page_rect: fitz.Rect) -> str:
        """Classify the type of text block"""
        # Position-based classification
        rel_y = bbox[1] / page_rect.height if page_rect.height > 0 else 0
        
        if rel_y < 0.1:
            return "header"
        elif rel_y > 0.9:
            return "footer"
        elif len(text.split()) < 3 and any(c.isdigit() for c in text):
            return "page_number"
        else:
            return "content"
    
    def _is_valid_element(self, element: TextElement) -> bool:
        """Validate text element"""
        return (len(element.text.strip()) >= 2 and 
                not element.text.isspace() and
                element.block_type not in ["page_number"])
    
    def _filter_repetitive_elements(self, elements: List[TextElement], total_pages: int) -> List[TextElement]:
        """Filter out repetitive headers/footers"""
        text_frequency = Counter(elem.text for elem in elements)
        threshold = max(2, total_pages * 0.7)  # Appears on 70%+ of pages
        
        filtered = []
        for element in elements:
            # Skip highly repetitive elements
            if text_frequency[element.text] > threshold:
                continue
            
            # Skip obvious headers/footers
            if (element.block_type in ["header", "footer"] and 
                len(element.text.split()) < 5):
                continue
            
            filtered.append(element)
        
        return filtered
    
    def _analyze_document_characteristics(self, elements: List[TextElement]) -> Dict[str, Any]:
        """Analyze document to understand its structure and type"""
        if not elements:
            return {}
        
        # Font size analysis
        font_sizes = [elem.font_size for elem in elements]
        size_counter = Counter(font_sizes)
        
        # Determine body text size
        body_size = size_counter.most_common(1)[0][0] if size_counter else 12
        
        # Identify potential heading sizes
        unique_sizes = sorted(set(font_sizes), reverse=True)
        heading_sizes = [size for size in unique_sizes if size > body_size]
        
        # Spacing analysis
        spacings = [elem.spacing_above for elem in elements if elem.spacing_above > 0]
        avg_spacing = np.mean(spacings) if spacings else 10
        large_spacing_threshold = avg_spacing * 1.5
        
        # Document type detection
        all_text = " ".join([elem.text.lower() for elem in elements[:100]])
        doc_type = self._detect_document_type(all_text)
        
        # Structural patterns analysis
        numbered_elements = len([e for e in elements if re.match(r'^\d+\.', e.text)])
        bullet_elements = len([e for e in elements if e.text.startswith('•') or e.text.startswith('-')])
        
        return {
            "body_size": body_size,
            "heading_sizes": heading_sizes,
            "avg_spacing": avg_spacing,
            "large_spacing_threshold": large_spacing_threshold,
            "doc_type": doc_type,
            "has_numbered_structure": numbered_elements > 5,
            "has_bullet_structure": bullet_elements > 10,
            "size_distribution": dict(size_counter),
            "total_elements": len(elements)
        }
    
    def _detect_document_type(self, text_sample: str) -> str:
        """Detect document type from text content"""
        type_indicators = {
            "academic": ["abstract", "methodology", "research", "study", "analysis", "citation"],
            "business": ["revenue", "profit", "financial", "quarterly", "executive", "strategy"],
            "manual": ["step", "procedure", "instruction", "guide", "how to", "section"],
            "report": ["summary", "overview", "findings", "recommendations", "conclusion"],
            "form": ["name:", "date:", "signature", "check", "fill", "complete"]
        }
        
        scores = {}
        for doc_type, keywords in type_indicators.items():
            score = sum(1 for keyword in keywords if keyword in text_sample)
            scores[doc_type] = score
        
        return max(scores.items(), key=lambda x: x[1])[0] if scores else "general"
    
    def _extract_title(self, elements: List[TextElement], doc_analysis: Dict) -> str:
        """Extract document title with improved heuristics"""
        if not elements:
            return ""
        
        # Look for title in first page, top 30%
        first_page_elements = [e for e in elements if e.page == 1]
        if not first_page_elements:
            return ""
        
        candidates = []
        
        for element in first_page_elements[:20]:  # Check first 20 elements
            score = 0
            
            # Font size factor
            if element.font_size > doc_analysis.get("body_size", 12) * 1.3:
                score += element.font_size
            
            # Bold factor
            if element.is_bold:
                score += 15
            
            # Position factor (higher = better for title)
            if element.bbox[1] < 200:  # Top of page
                score += 20
            
            # Length factor
            word_count = len(element.text.split())
            if 2 <= word_count <= 15:
                score += 10
            elif word_count > 20:
                score -= 10
            
            # Avoid numbered sections as titles
            if re.match(r'^\d+\.', element.text):
                score -= 20
            
            if score > 10:
                candidates.append((element, score))
        
        if candidates:
            best_candidate = max(candidates, key=lambda x: x[1])
            return best_candidate[0].text.strip()
        
        # Fallback to first substantial element
        for element in first_page_elements:
            if len(element.text.split()) >= 2:
                return element.text.strip()
        
        return ""
    
    def _identify_headings_adaptive(self, elements: List[TextElement], 
                                  doc_analysis: Dict) -> List[Tuple[TextElement, float, str]]:
        """Adaptive heading identification using multiple strategies"""
        heading_candidates = []
        
        for element in elements:
            score = 0
            level_hint = "H3"  # Default level
            
            # Strategy 1: Font size analysis
            size_ratio = element.font_size / doc_analysis.get("body_size", 12)
            if size_ratio > 1.4:
                score += 25
                level_hint = "H1"
            elif size_ratio > 1.2:
                score += 20
                level_hint = "H2"
            elif size_ratio > 1.1:
                score += 15
                level_hint = "H3"
            
            # Strategy 2: Formatting analysis
            if element.is_bold:
                score += 20
            if element.is_italic:
                score += 5
            
            # Strategy 3: Spacing analysis
            if element.spacing_above > doc_analysis.get("large_spacing_threshold", 15):
                score += 15
            
            # Strategy 4: Semantic pattern matching
            semantic_score, semantic_level = self._analyze_semantic_patterns(element.text, doc_analysis["doc_type"])
            score += semantic_score
            if semantic_level and semantic_score > 10:
                level_hint = semantic_level
            
            # Strategy 5: Structural analysis
            structure_score = self._analyze_structure_indicators(element, elements)
            score += structure_score
            
            # Strategy 6: Length and content analysis
            word_count = len(element.text.split())
            if word_count <= 8:
                score += 10
            elif word_count <= 15:
                score += 5
            elif word_count > 25:
                score -= 10
            
            # Title case bonus
            if element.text.istitle() and 2 <= word_count <= 10:
                score += 8
            
            # Position bonus
            if element.column == 0:  # First column
                score += 5
            
            if score >= self.min_heading_score * 100:  # Convert to 0-100 scale
                heading_candidates.append((element, score / 100, level_hint))
        
        return self._refine_heading_candidates(heading_candidates)
    
    def _analyze_semantic_patterns(self, text: str, doc_type: str) -> Tuple[float, str]:
        """Analyze text against semantic patterns"""
        text = text.strip()
        max_score = 0
        best_level = "H3"
        
        # Check all pattern categories
        for category, patterns in self.semantic_patterns.items():
            if category == doc_type or category in ["numbered", "structural"]:
                for pattern, weight, level in patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        score = weight * 20  # Scale to 0-20
                        if score > max_score:
                            max_score = score
                            best_level = level
        
        return max_score, best_level
    
    def _analyze_structure_indicators(self, element: TextElement, all_elements: List[TextElement]) -> float:
        """Analyze structural indicators for heading likelihood"""
        score = 0
        
        # Check if followed by indented content
        element_idx = all_elements.index(element)
        if element_idx < len(all_elements) - 1:
            next_element = all_elements[element_idx + 1]
            if (next_element.page == element.page and 
                next_element.bbox[0] > element.bbox[0] + 20):  # Indented
                score += 10
        
        # Check for bullet points following
        following_elements = all_elements[element_idx + 1:element_idx + 4]
        bullet_count = sum(1 for e in following_elements 
                          if e.text.startswith(('•', '-', '*')) or re.match(r'^\d+\.', e.text))
        if bullet_count >= 2:
            score += 15
        
        # Check isolation (spacing before and after)
        if element.spacing_above > 15:
            score += 5
        
        return score
    
    def _refine_heading_candidates(self, candidates: List[Tuple[TextElement, float, str]]) -> List[Tuple[TextElement, float, str]]:
        """Refine and deduplicate heading candidates"""
        if not candidates:
            return []
        
        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Remove very similar headings
        refined = []
        seen_texts = set()
        
        for element, score, level in candidates:
            text_normalized = re.sub(r'\s+', ' ', element.text.lower().strip())
            
            # Skip exact duplicates
            if text_normalized in seen_texts:
                continue
            
            # Skip very short or very long candidates
            if len(text_normalized) < 3 or len(element.text.split()) > 30:
                continue
            
            refined.append((element, score, level))
            seen_texts.add(text_normalized)
        
        return refined
    
    def _build_hierarchical_outline(self, headings: List[Tuple[TextElement, float, str]], 
                                  doc_analysis: Dict) -> List[Dict]:
        """Build hierarchical outline from heading candidates"""
        if not headings:
            return []
        
        # Sort by document order (page, then y-coordinate)
        headings.sort(key=lambda x: (x[0].page, x[0].bbox[1]))
        
        # Group by confidence and font size for level assignment
        size_groups = defaultdict(list)
        for element, score, suggested_level in headings:
            size_key = round(element.font_size, 1)
            size_groups[size_key].append((element, score, suggested_level))
        
        # Assign levels based on font size hierarchy
        sorted_sizes = sorted(size_groups.keys(), reverse=True)
        level_mapping = {}
        
        for i, size in enumerate(sorted_sizes[:3]):  # Only H1, H2, H3
            level = f"H{i + 1}"
            for element, score, suggested_level in size_groups[size]:
                key = (element.page, element.text)
                level_mapping[key] = level
        
        # Build final outline
        outline = []
        for element, score, suggested_level in headings:
            key = (element.page, element.text)
            final_level = level_mapping.get(key, suggested_level)
            
            outline.append({
                "level": final_level,
                "text": element.text.strip(),
                "page": element.page
            })
        
        return outline
