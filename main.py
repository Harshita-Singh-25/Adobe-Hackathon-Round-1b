import os
import sys
import json
import logging
import time
from datetime import datetime
from persona_analyzer import DocumentCollectionAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        
        # Create subdirectory for 1A outputs
        outlines_dir = os.path.join(output_dir, "outlines")
        os.makedirs(outlines_dir, exist_ok=True)
        
        # Process documents
        analyzer = DocumentCollectionAnalyzer(outlines_dir=outlines_dir)
        result = analyzer.process_collection(config)
        
        # Save Round 1B output
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
    input_dir = "/app/input"
    output_dir = "/app/output"
    
    # We'll only process Round 1B in this setup
    logger.info("Starting Round 1B processing")
    process_1b(input_dir, output_dir)