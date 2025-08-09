import os
import sys
import json
import logging
import time
import re
import fitz  # PyMuPDF
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================
# Round 1A: PDF Structure Extraction Submodule
# ==============================================
#version 2
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

# ==============================================
# Round 1B: Persona-Driven Document Intelligence
# ==============================================

class PersonaAnalyzer:
    """Advanced persona analysis with semantic understanding"""
    
    def __init__(self):
        self.domain_keywords = {
            "academic": ["research", "study", "analysis", "thesis", "publication", "scholar"],
            "business": ["business", "financial", "market", "strategy", "corporate", "executive"],
            "technical": ["engineering", "development", "system", "technical", "software"],
            "medical": ["medical", "healthcare", "clinical", "patient", "treatment"],
            "legal": ["legal", "law", "court", "contract", "compliance"],
            "education": ["education", "student", "teacher", "curriculum", "learning"]
        }
        
        self.role_patterns = {
            "researcher": r"(research|study|investigate|analyze)",
            "analyst": r"(analy[sz]e|evaluate|assess|report)",
            "manager": r"(manage|lead|oversee|direct|coordinate)",
            "consultant": r"(consult|advise|recommend|guide)",
            "student": r"(learn|study|understand|exam|assignment)"
        }
        

        # Initialize cross-encoder for semantic matching
        try:
            # Use a tiny model that won't require large downloads
            self.cross_encoder = CrossEncoder('/models/cross-encoder/stsb-TinyBERT-L-4')

            logging.info("Loaded TinyBERT model successfully")
        except Exception as e:
            logging.warning(f"Couldn't load cross-encoder: {e}")
            self.cross_encoder = None
    
    def analyze_persona(self, persona_text: str) -> Dict[str, Any]:
        """Comprehensive persona analysis"""
        if not persona_text:
            return {"type": "general", "domain": "general", "experience": "intermediate"}
        
        text_lower = persona_text.lower()
        
        # Domain detection
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                domain_scores[domain] = score
        
        primary_domain = max(domain_scores.items(), key=lambda x: x[1])[0] if domain_scores else "general"
        
        # Role detection
        primary_role = "general"
        for role, pattern in self.role_patterns.items():
            if re.search(pattern, text_lower):
                primary_role = role
                break
        
        # Experience level detection
        experience_indicators = {
            "senior": ["senior", "lead", "principal", "expert", "experienced", "veteran"],
            "junior": ["junior", "entry", "beginner", "new", "fresh", "trainee"],
            "intermediate": ["analyst", "specialist", "professional", "coordinator"]
        }
        
        experience_level = "intermediate"  # default
        for level, indicators in experience_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                experience_level = level
                break
        
        # Extract specific interests/expertise
        interests = self._extract_interests(text_lower)
        
        # Extract keywords
        keywords = self._extract_keywords(persona_text)
        
        return {
            "type": f"{primary_domain}_{primary_role}",
            "domain": primary_domain,
            "role": primary_role,
            "experience": experience_level,
            "interests": interests,
            "keywords": keywords,
            "raw_text": persona_text
        }
    
    def analyze_job(self, job_text: str) -> Dict[str, Any]:
        """Analyze job-to-be-done requirements"""
        if not job_text:
            return {"type": "general", "requirements": [], "complexity": "medium"}
        
        text_lower = job_text.lower()
        
        # Extract action verbs
        action_verbs = re.findall(r'\b(analyz|review|create|develop|design|implement|assess|evaluat|research|study|prepar|plan|organize|manag)\w*', text_lower)
        
        # Extract key requirements
        requirements = self._extract_requirements(job_text)
        
        # Assess complexity
        complexity = self._assess_job_complexity(job_text)
        
        # Detect job type
        job_type = self._detect_job_type(text_lower)
        
        # Extract keywords
        keywords = self._extract_keywords(job_text)
        
        return {
            "type": job_type,
            "requirements": requirements,
            "complexity": complexity,
            "actions": list(set(action_verbs)),
            "keywords": keywords,
            "raw_text": job_text
        }
    
    def _extract_interests(self, text: str) -> List[str]:
        """Extract specific interests from persona text"""
        interest_indicators = ["focus", "specialize", "expert", "experience", "work"]
        interests = []
        
        for indicator in interest_indicators:
            pattern = rf"{indicator}\s+(?:in|on|with)\s+([a-zA-Z\s]+?)(?:\.|,|$)"
            matches = re.findall(pattern, text)
            interests.extend([match.strip() for match in matches])
        
        return interests[:5]  # Top 5
    
    def _extract_requirements(self, job_text: str) -> List[str]:
        """Extract key requirements from job description"""
        # Split by common delimiters
        sentences = re.split(r'[.;!?]', job_text)
        requirements = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and any(verb in sentence.lower() for verb in ["analyze", "create", "develop", "prepare", "plan"]):
                requirements.append(sentence)
        
        return requirements[:10]  # Top 10
    
    def _assess_job_complexity(self, job_text: str) -> str:
        """Assess job complexity level"""
        complexity_indicators = {
            "high": ["comprehensive", "detailed", "complex", "advanced", "sophisticated"],
            "low": ["simple", "basic", "quick", "brief", "summary"]
        }
        
        text_lower = job_text.lower()
        word_count = len(job_text.split())
        
        # Check for complexity indicators
        for level, indicators in complexity_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                return level
        
        # Use word count as fallback
        if word_count > 100:
            return "high"
        elif word_count < 30:
            return "low"
        else:
            return "medium"
    
    def _detect_job_type(self, text: str) -> str:
        """Detect the type of job/task"""
        job_types = {
            "analysis": ["analyze", "analysis", "review", "assess", "evaluate"],
            "creation": ["create", "develop", "design", "build", "generate"],
            "research": ["research", "study", "investigate", "explore"],
            "planning": ["plan", "organize", "prepare", "coordinate"],
            "summary": ["summarize", "overview", "summary", "brief"]
        }
        
        for job_type, keywords in job_types.items():
            if any(keyword in text for keyword in keywords):
                return job_type
        
        return "general"
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Remove stopwords and punctuation
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        word_counts = Counter(words)
        
        # Get top 20 most frequent words
        keywords = [word for word, count in word_counts.most_common(20)]
        
        return keywords

class DocumentCollectionAnalyzer:
    """Main processor for Round 1B that coordinates all components"""
    
    def __init__(self):
        self.pdf_extractor = PDFStructureExtractor()
        self.persona_analyzer = PersonaAnalyzer()
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def process_collection(self, config: Dict) -> Dict:
        """Main processing pipeline for Round 1B"""
        try:
            # Analyze persona and job
            persona_analysis = self.persona_analyzer.analyze_persona(config["persona"]["role"])
            job_analysis = self.persona_analyzer.analyze_job(config["job_to_be_done"]["task"])
            
            logger.info(f"Persona analysis: {persona_analysis['type']}")
            logger.info(f"Job analysis: {job_analysis['type']}")
            
            # Process each document
            all_sections = []
            input_documents = []
            
            for doc_meta in config["documents"]:
                pdf_path = os.path.join("/app/input", doc_meta["filename"])
                if not os.path.exists(pdf_path):
                    logger.warning(f"Document not found: {pdf_path}")
                    continue
                
                logger.info(f"Processing document: {doc_meta['filename']}")
                
                # Extract document structure using Round 1A functionality
                outline = self.pdf_extractor.extract_outline(pdf_path)
                
                filename = doc_meta["filename"]
                # Extract content for each section
                with fitz.open(pdf_path) as pdf_doc:
                    sections = self._extract_section_content(pdf_doc, outline["outline"])
                    scored_sections = self._score_sections(sections, persona_analysis, job_analysis)
                    
                    # Add document name to each section
                    for section in scored_sections:
                        section["document"] = filename
                    
                    all_sections.extend(scored_sections)
                    input_documents.append(filename)
            
            logger.info(f"Total sections extracted: {len(all_sections)}")
            
            # Rank sections across all documents
            ranked_sections = self._rank_sections(all_sections, persona_analysis, job_analysis)
            
            # Generate final output
            return self._format_output(config, ranked_sections, persona_analysis, job_analysis)
            
        except Exception as e:
            logger.error(f"Error processing document collection: {e}" , exc_info=True)
            return {
                "metadata": {
                    "input_documents": [],
                    "persona": "",
                    "job_to_be_done": "",
                    "processing_timestamp": datetime.now().isoformat(),
                    "error": str(e)
                },
                "extracted_sections": [],
                "subsection_analysis": []
            }
    
    def _extract_section_content(self, doc: fitz.Document, outline: List[Dict]) -> List[Dict]:
        """Extract content for each outline section"""
        sections = []
        
        # Get all text with position info
        all_text_elements = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if block["type"] == 0 and "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        for span in line["spans"]:
                            line_text += span["text"]
                        
                        if line_text.strip():
                            all_text_elements.append({
                                "text": line_text.strip(),
                                "page": page_num + 1,
                                "bbox": line["bbox"]
                            })
        
        # Match outline headings to text positions and extract content
        for i, heading in enumerate(outline):
            section_content = self._extract_section_content_between_headings(
                all_text_elements, heading, 
                outline[i + 1] if i + 1 < len(outline) else None
            )
            
            if section_content:
                sections.append({
                    "title": heading["text"],
                    "level": heading["level"],
                    "page": heading["page"],
                    "content": section_content,
                    "word_count": len(section_content.split())
                })
        
        return sections
    
    def _extract_section_content_between_headings(self, all_elements: List[Dict], current_heading: Dict, 
                                                next_heading: Optional[Dict]) -> str:
        """Extract content between two headings"""
        content_parts = []
        current_page = current_heading["page"]
        
        # Find elements after current heading
        start_collecting = False
        
        for element in all_elements:
            # Start collecting after we find the heading
            if (element["page"] == current_page and 
                current_heading["text"].lower() in element["text"].lower()):
                start_collecting = True
                continue
            
            # Stop collecting when we reach next heading
            if (start_collecting and next_heading and 
                element["page"] >= next_heading["page"] and
                next_heading["text"].lower() in element["text"].lower()):
                break
            
            # Collect content
            if start_collecting:
                content_parts.append(element["text"])
                
                # Stop if we've collected enough or moved too far
                if len(content_parts) > 50:  # Limit section size
                    break
        
        return " ".join(content_parts).strip()
    
    def _score_sections(self, sections: List[Dict], persona_analysis: Dict, 
                       job_analysis: Dict) -> List[Dict]:
        """Score sections based on relevance to persona and job"""
        if not sections:
            return []
        
        # Prepare job requirements text for comparison
        job_text = " ".join(job_analysis.get("requirements", []) + 
                           [job_analysis.get("raw_text", "")])
        persona_text = " ".join(persona_analysis.get("interests", []) + 
                               [persona_analysis.get("raw_text", "")])
        
        # Combine for unified relevance scoring
        query_text = f"{job_text} {persona_text}"
        
        # Extract section texts for vectorization
        section_texts = [section["content"] for section in sections if section["content"]]
        
        if not section_texts or not query_text.strip():
            # Fallback scoring if no vectorization possible
            for section in sections:
                section["relevance_score"] = self._fallback_scoring(section, persona_analysis, job_analysis)
                section["ranking_factors"] = {"fallback": True}
            return sections
        
        try:
            # Calculate TF-IDF similarity
            all_texts = section_texts + [query_text]
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            
            # Calculate similarity scores
            query_vector = tfidf_matrix[-1]
            section_vectors = tfidf_matrix[:-1]
            similarities = cosine_similarity(section_vectors, query_vector).flatten()
            
            # Enhanced scoring with multiple factors
            for i, section in enumerate(sections):
                if i < len(similarities):
                    base_score = similarities[i]
                else:
                    base_score = 0.0
                
                # Apply various scoring factors
                factors = self._calculate_scoring_factors(section, persona_analysis, job_analysis)
                
                # Combine scores
                final_score = base_score * factors["content_relevance"] * factors["persona_match"] * factors["job_alignment"]
                
                section["relevance_score"] = round(final_score, 4)
                section["ranking_factors"] = factors
        
        except Exception as e:
            logger.warning(f"TF-IDF scoring failed: {e}. Using fallback scoring.")
            # Fallback to keyword-based scoring
            for section in sections:
                section["relevance_score"] = self._fallback_scoring(section, persona_analysis, job_analysis)
                section["ranking_factors"] = {"fallback": True}
        
        return sections
    
    def _calculate_scoring_factors(self, section: Dict, persona_analysis: Dict, 
                                 job_analysis: Dict) -> Dict[str, float]:
        """Calculate various scoring factors for a section"""
        factors = {
            "content_relevance": 1.0,
            "persona_match": 1.0,
            "job_alignment": 1.0,
            "section_quality": 1.0
        }
        
        content = section["content"].lower()
        
        # Content relevance (based on section title and content quality)
        if section["word_count"] > 50:
            factors["content_relevance"] *= 1.2
        elif section["word_count"] < 20:
            factors["content_relevance"] *= 0.8
        
        # Persona match
        persona_interests = persona_analysis.get("interests", [])
        persona_domain = persona_analysis.get("domain", "")
        persona_keywords = persona_analysis.get("keywords", [])
        
        interest_matches = sum(1 for interest in persona_interests 
                             if interest.lower() in content)
        if interest_matches > 0:
            factors["persona_match"] *= (1.0 + interest_matches * 0.1)
        
        if persona_domain and persona_domain in content:
            factors["persona_match"] *= 1.15
        
        # Keyword matching
        keyword_matches = sum(1 for kw in persona_keywords if kw in content)
        factors["persona_match"] *= (1.0 + keyword_matches * 0.05)
        
        # Job alignment
        job_actions = job_analysis.get("actions", [])
        job_type = job_analysis.get("type", "")
        job_keywords = job_analysis.get("keywords", [])
        
        action_matches = sum(1 for action in job_actions 
                           if action in content)
        if action_matches > 0:
            factors["job_alignment"] *= (1.0 + action_matches * 0.1)
        
        if job_type and job_type in content:
            factors["job_alignment"] *= 1.1
            
        # Job keyword matching
        job_keyword_matches = sum(1 for kw in job_keywords if kw in content)
        factors["job_alignment"] *= (1.0 + job_keyword_matches * 0.05)
        
        # Section quality (based on structure and level)
        if section["level"] == "H1":
            factors["section_quality"] *= 1.2
        elif section["level"] == "H2":
            factors["section_quality"] *= 1.1
        
        return factors
    
    def _fallback_scoring(self, section: Dict, persona_analysis: Dict, 
                         job_analysis: Dict) -> float:
        """Fallback scoring method using simple keyword matching"""
        content = section["content"].lower()
        score = 0.3  # Base score
        
        # Check persona interests
        interests = persona_analysis.get("interests", [])
        for interest in interests:
            if interest.lower() in content:
                score += 0.1
        
        # Check job requirements
        requirements = job_analysis.get("requirements", [])
        for req in requirements:
            if any(word in content for word in req.lower().split()):
                score += 0.1
        
        return min(score, 1.0)
    
    def _rank_sections(self, sections: List[Dict], persona_analysis: Dict, 
                      job_analysis: Dict) -> List[Dict]:
        """Rank sections with diversity across documents"""
        if not sections:
            return []
        
        # Sort by relevance score
        sorted_sections = sorted(sections, key=lambda x: x["relevance_score"], reverse=True)
        
        # Apply diversity selection to avoid too many sections from same document
        selected = []
        document_count = defaultdict(int)
        max_per_document = max(2, 25 // len(set(s["document"] for s in sorted_sections)))
        
        for section in sorted_sections:
            if len(selected) >= 25:  # Maximum sections
                break
            
            doc_name = section["document"]
            if document_count[doc_name] < max_per_document:
                selected.append(section)
                document_count[doc_name] += 1
        
        return selected
    
    def _format_output(self, config: Dict, sections: List[Dict], 
                      persona_analysis: Dict, job_analysis: Dict) -> Dict:
        """Generate final output in required format"""
        extracted_sections = []
        subsection_analysis = []
        
        for i, section in enumerate(sections):
            extracted_sections.append({
                "document": section["document"],
                "page_number": section["page"],
                "section_title": section["title"],
                "importance_rank": i + 1,
                "relevance_score": section["relevance_score"],
                "section_level": section["level"]
            })
            
            # Create subsection analysis
            content = section["content"]
            if len(content) > 300:
                # Split long content into smaller parts
                sentences = [s.strip() + '.' for s in content.split('.') if s.strip()]
                chunks = []
                current_chunk = []
                current_length = 0
                
                for sentence in sentences:
                    sentence_length = len(sentence.split())
                    if current_length + sentence_length > 50 and current_chunk:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = [sentence]
                        current_length = sentence_length
                    else:
                        current_chunk.append(sentence)
                        current_length += sentence_length
                
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                
                # Take top 2 chunks
                for j, chunk in enumerate(chunks[:2]):
                    subsection_analysis.append({
                        "document": section["document"],
                        "page_number": section["page"],
                        "refined_text": chunk,
                        "subsection_index": j + 1
                    })
            else:
                subsection_analysis.append({
                    "document": section["document"],
                    "page_number": section["page"],
                    "refined_text": content,
                    "subsection_index": 1
                })
        
        return {
            "metadata": {
                "input_documents": [doc["filename"] for doc in config["documents"]],
                "persona": config["persona"]["role"],
                "job_to_be_done": config["job_to_be_done"]["task"],
                "processing_timestamp": datetime.now().isoformat(),
                "persona_analysis": {
                    "type": persona_analysis["type"],
                    "domain": persona_analysis["domain"],
                    "experience": persona_analysis["experience"]
                },
                "job_analysis": {
                    "type": job_analysis["type"],
                    "complexity": job_analysis["complexity"]
                },
                "total_sections_found": len(sections)
            },
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis
        }

# ==============================================
# Main Processor
# ==============================================

def process_1b(input_dir: str, output_dir: str):
    """Process Round 1B input and generate output"""
    try:
        start_time = time.time()
        
        # Load config
        config_path = os.path.join(input_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        logger.info("Loaded configuration")
        
        # Process documents
        analyzer = DocumentCollectionAnalyzer()
        result = analyzer.process_collection(config)
        
        # Save output
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "output.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        processing_time = time.time() - start_time
        logger.info(f"Processing complete in {processing_time:.2f} seconds")
        logger.info(f"Results saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error in process_1b: {e}")
        raise

if __name__ == "__main__":
    # Determine which round to run based on input structure
    input_dir = "/app/input"
    output_dir = "/app/output"
    
    # Check if this is Round 1B (has config.json)
    if os.path.exists(os.path.join(input_dir, "config.json")):
        logger.info("Detected Round 1B input format")
        process_1b(input_dir, output_dir)
    else:
        logger.error("Invalid input format for Round 1B")
        sys.exit(1)