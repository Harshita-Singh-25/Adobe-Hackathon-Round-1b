


# 🎯 Round 1B Adobe India Hackathon  
A persona-driven document analysis tool that extracts structured insights from PDFs based on a role-specific task. Designed to run securely in a containerized environment, producing rich JSON output for evaluation.

---

## 🚀 Solution Overview

This solution reads multiple PDF documents and processes them according to a persona’s role and their task objectives, providing structured output with metadata, key sections, and refined content analysis.

### Key Capabilities  
- **Persona-Task Contextualization**: Adapts document parsing based on provided role and task  
- **Semantic Section Extraction**: Identifies important sections based on relevance to the task  
- **Refined Text Analysis**: Highlights contextual content from deep document layers  
- **Robust Output Formatting**: Structured JSON capturing hierarchy, ranking, and timestamps  
- **Offline Secure Execution**: Runs in isolated Docker container with no network access  

---

## 🏗️ Architecture

### Components  
**Input Interface**  
- `config.json`: Defines the persona and goal-oriented task  
- `*.pdf` files: 3–10 documents for semantic parsing  

**Processing Engine**  
- **Metadata Collector**: Identifies and timestamps the document processing  
- **Section Extractor**: Maps document sections and ranks their task-relevance  
- **Subsection Analyzer**: Refines contextual content to aid understanding and planning  

**Output Generator**  
- Produces `output.json` with:
  - Source metadata  
  - Extracted section hierarchy  
  - Task-aligned content highlights  

---

## 🔧 Technical Workflow

### 1. Build the Docker Image  
```bash
docker build --platform linux/amd64 -t adobe-solution .
```

### 2. Prepare Test Case  
Create the following folder structure in your working directory:  
```
test_case/
├── input/
│   ├── config.json
│   ├── document1.pdf
│   └── document2.pdf
└── output/
```

**Sample config.json**
```json
{
  "documents": [
    { "filename": "document1.pdf" },
    { "filename": "document2.pdf" }
  ],
  "persona": {
    "role": "Travel Planner"
  },
  "job_to_be_done": {
    "task": "Plan a 4-day itinerary"
  }
}
```

---

### 3. Run the Container  
```bash
docker run --rm \
  -v "$(pwd)/test_case/input:/app/input" \
  -v "$(pwd)/test_case/output:/app/output" \
  --network none \
  adobe-solution
```

### Container Specs  
- **Architecture**: linux/amd64  
- **Dependencies**: Python + PDF parsing libraries  
- **Runtime**: Isolated, offline, secure  

---

## 📊 Output Format

After execution, results are stored in:  
```
/output/output.json
```

### Sample output:
```json
{
  "metadata": {
    "input_documents": ["document1.pdf"],
    "persona": "Travel Planner",
    "job_to_be_done": "Plan 4-day trip",
    "processing_timestamp": "2025-07-30T12:00:00"
  },
  "extracted_sections": [
    {
      "document": "document1.pdf",
      "section_title": "Top Attractions",
      "importance_rank": 1,
      "page_number": 3
    }
  ],
  "subsection_analysis": [
    {
      "document": "document1.pdf",
      "refined_text": "The Eiffel Tower offers...",
      "page_number": 3
    }
  ]
}
```

---

## 📁 Project Structure

```
├── Dockerfile           # Container configuration
├── main.py              # PDF processing pipeline
├── config.json          # Persona-task definition file
├── requirements.txt     # Package dependencies
└── README.md            # This guide
```

---

## ⚙️ Usage Summary

1. Place your PDFs and `config.json` in `input/`  
2. Run the container—automated pipeline will trigger  
3. Collect `output.json` results from `output/`  

✅ No manual parsing needed  
✅ Role-based insight extraction  
✅ Evaluation-ready formatting  

---
