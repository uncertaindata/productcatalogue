"""
Product Hierarchy Classifier - Solution Template
=================================================
This template provides a starting structure for the assignment.
Feel free to modify, extend, or completely rewrite as needed.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import re
from collections import defaultdict
import logging
from modified_code import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main execution function"""
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description='Product Hierarchy Classifier')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--sample', type=int, help='Process only N samples for testing')
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=42)
        logger.info(f"Using {len(df)} samples for testing")
    
    # Initialize classifier
    classifier = ProductHierarchyClassifier()
    
    # Process dataset
    start_time = time.time()
    classifier.process_dataset(df)
    processing_time = time.time() - start_time
    
    classifier.stats['processing_time_seconds'] = round(processing_time, 2)
    
    # Export results
    classifier.export_results(args.output)
    
    # Print summary
    print("\n" + "="*50)
    print("PROCESSING COMPLETE")
    print("="*50)
    print(f"Total Products: {classifier.stats['total_products']}")
    print(f"Product Groups Created: {classifier.stats['total_groups']}")
    print(f"Variants Created: {classifier.stats['total_variants']}")
    print(f"Products Assigned: {classifier.stats['products_assigned']}")
    print(f"Average Confidence: {classifier.stats.get('average_confidence', 0):.2f}")
    print(f"Processing Time: {processing_time:.2f} seconds")
    print("="*50)


if __name__ == '__main__':
    main()