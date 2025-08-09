import fitz  # PyMuPDF
import os
import re
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder
from pdf_structure_extractor import PDFStructureExtractor

logger = logging.getLogger(__name__)



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
    
    def __init__(self ,outlines_dir: str = None):
        self.pdf_extractor = PDFStructureExtractor()
        self.persona_analyzer = PersonaAnalyzer()
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.outlines_dir = outlines_dir
    
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
                filename = doc_meta["filename"]
                
                if not os.path.exists(pdf_path):
                    logger.warning(f"Document not found: {pdf_path}")
                    continue
                
                logger.info(f"Processing document: {filename}")
                
                try:
                    # Extract document structure (Round 1A)
                    outline = self.pdf_extractor.extract_outline(pdf_path)
                    
                    # Save Round 1A output if outlines_dir is specified
                    if self.outlines_dir and outline:
                        os.makedirs(self.outlines_dir, exist_ok=True)
                        outline_path = os.path.join(
                            self.outlines_dir, 
                            f"{os.path.splitext(filename)[0]}.json"
                        )
                        with open(outline_path, 'w', encoding='utf-8') as f:
                            json.dump(outline, f, indent=2, ensure_ascii=False)
                    
                    # Extract content for each section (Round 1B)
                    with fitz.open(pdf_path) as pdf_doc:
                        if not outline or "outline" not in outline:
                            logger.warning(f"No outline extracted from {filename}")
                            continue
                            
                        sections = self._extract_section_content(pdf_doc, outline["outline"])
                        scored_sections = self._score_sections(sections, persona_analysis, job_analysis)
                        
                        # Add document name to each section
                        for section in scored_sections:
                            section["document"] = filename
                        
                        all_sections.extend(scored_sections)
                        input_documents.append(filename)
                        
                except Exception as e:
                    logger.error(f"Error processing {filename}: {str(e)}")
                    continue
            
            logger.info(f"Total sections extracted: {len(all_sections)}")
            
            # Rank sections across all documents
            ranked_sections = self._rank_sections(all_sections, persona_analysis, job_analysis)
            
            # Generate final output
            return self._format_output(config, ranked_sections, persona_analysis, job_analysis)


        ####    
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
