#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the vocabulary builder.
This script demonstrates how to use the VocabularyBuilder class to build a vocabulary
from the MS MARCO dataset with a small sample size for testing purposes.
"""

import logging
from vocabulary import VocabularyBuilder

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to test the vocabulary builder.
    """
    logger.info("Starting vocabulary builder test")
    
    # Create vocabulary builder with a small sample size for testing
    builder = VocabularyBuilder(
        min_ngram=1,
        max_ngram=3,
        max_skip_distance=5,
        min_frequency=2,  # Lower threshold for testing
        max_vocab_size=10000,  # Limit vocabulary size for testing
        bert_model_name="sentence-transformers/msmarco-distilbert-dot-v5",
        output_dir="vocabulary_test",
        batch_size=16,  # Smaller batch size for testing
    )
    
    # Build vocabulary with a small sample size
    builder.build_vocabulary(max_samples=100)  # Process only 100 samples for testing
    
    logger.info("Vocabulary builder test completed")

if __name__ == "__main__":
    main()
