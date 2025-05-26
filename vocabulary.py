#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Vocabulary builder for BERT approximation using MS MARCO dataset.
This script implements a pipeline to build a hybrid vocabulary consisting of
subword tokens (from SentencePiece) and multi-word phrases (selected based on
frequency, PMI, and BERT-based cohesion scores).
"""

import os
import re
import json
import math
import argparse
import logging
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from collections import Counter

import nltk
import sentencepiece as spm
import torch
from transformers import BertTokenizer, BertForMaskedLM
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download NLTK data if needed
nltk.download('punkt_tab')

try:
    nltk.data.find('tokenizers/punkt')
except (LookupError, FileNotFoundError):
    logger.info("Downloading NLTK punkt tokenizer models...")
    nltk.download('punkt')
    nltk.download('punkt_tab')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords


class VocabularyBuilder:
    """
    Builds a hybrid vocabulary from the MS MARCO dataset.
    The process involves:
    1. Data Preparation: Normalizing text and segmenting into sentences.
    2. Training a Unigram Tokenizer: Using SentencePiece for a base subword vocabulary.
    3. Extracting N-grams and Skip-grams: Identifying frequent phrases.
    4. Computing PMI: Scoring phrases based on statistical association.
    5. BERT-Based Phrase Cohesion Scoring: Scoring phrases based on semantic cohesion.
    6. Selecting Top Phrases: Merging phrases from PMI and BERT scores.
    7. Building a Trie-Based Tokenizer: For efficient tokenization with the hybrid vocabulary.
    8. Final Vocabulary Output: Saving the token-to-ID mapping.
    """

    def __init__(
        self,
        msmarco_path: str,
        output_dir: str = "vocabulary_new",
        sp_vocab_size: int = 20000,
        min_freq_ngram: int = 5,
        top_n_pmi: int = 5000,
        top_n_bert: int = 5000,
        bert_model_name: str = 'bert-base-uncased',
        embedding_model_name: str = 'sentence-transformers/msmarco-distilbert-dot-v5',
        embedding_pooling_mode: str = 'mean',
        embedding_batch_size: int = 64,
        device: Optional[str] = None,
    ):
        self.msmarco_path = msmarco_path
        self.output_dir = output_dir
        self.sp_vocab_size = sp_vocab_size
        self.min_freq_ngram = min_freq_ngram
        self.top_n_pmi = top_n_pmi
        self.top_n_bert = top_n_bert
        self.bert_model_name = bert_model_name
        self.embedding_model_name = embedding_model_name
        self.embedding_pooling_mode = embedding_pooling_mode
        self.embedding_batch_size = embedding_batch_size

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        logger.info(f"Using device: {self.device}")

        os.makedirs(self.output_dir, exist_ok=True)

        self.normalized_sentences_file = os.path.join(self.output_dir, "msmarco_sentences.txt")
        self.sp_model_prefix = os.path.join(self.output_dir, "bertvocab_unigram")
        self.sp_model_file = f"{self.sp_model_prefix}.model"
        self.sp_vocab_file = f"{self.sp_model_prefix}.vocab"
        self.final_vocab_file = os.path.join(self.output_dir, "vocab_final.txt")
        self.embeddings_file = os.path.join(self.output_dir, "vocab_embeddings.npy")

        self.word_counter = Counter()
        self.bigram_counter = Counter()
        self.trigram_counter = Counter()
        self.skipgram_counter = Counter()
        
        self.bert_tokenizer = None
        self.bert_model = None
        self.embedding_tokenizer = None
        self.embedding_model = None

    def _normalize_text(self, text: str) -> str:
        """Normalizes text: lowercase, remove punctuation, normalize whitespace."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]+", " ", text)  # Keep alphanumeric and space
        text = re.sub(r"\s+", " ", text).strip() # Collapse multiple spaces
        return text

    def step1_prepare_data(self):
        """
        Loads MS MARCO, normalizes text, segments into sentences,
        and saves to msmarco_sentences.txt.
        """
        logger.info("Step 1: Preparing data...")
        if os.path.exists(self.normalized_sentences_file):
            logger.info(f"{self.normalized_sentences_file} already exists. Skipping data preparation.")
            return

        # This is a placeholder for loading MS MARCO.
        # Users should adapt this to their specific MS MARCO data format and location.
        # Assuming msmarco_path is a text file with one passage per line.
        # For a real MS MARCO tsv/json, parsing would be needed.
        # Example: load_dataset("ms_marco", "v1.1", split="train")
        
        raw_texts = []
        # Assuming self.msmarco_path is a file where each line is a document/passage
        # For demonstration, let's assume it's a simple text file.
        # In a real scenario, this would involve parsing the MS MARCO collection.
        # For example, if it's a .tsv file:
        # with open(self.msmarco_path, 'r', encoding='utf-8') as f:
        #     for line in f:
        #         parts = line.strip().split('\t')
        #         if len(parts) >= 2: # Assuming passage is in one of the columns
        #             raw_texts.append(parts[1]) # Adjust index as per actual file structure

        # For this example, let's simulate reading from a file.
        # Replace this with actual MS MARCO loading logic.
        if not os.path.exists(self.msmarco_path):
            logger.error(f"MS MARCO data file not found at {self.msmarco_path}. Please provide a valid path.")
            logger.error("For demonstration, creating a dummy msmarco.txt. Please replace with actual data.")
            with open(self.msmarco_path, "w", encoding="utf-8") as f_dummy:
                f_dummy.write("Barack Obama (born August 4, 1961) was the 44th President of the United States.\n")
                f_dummy.write("The Eiffel Tower is located in Paris. It is a famous landmark.\n")
                f_dummy.write("Artificial intelligence and machine learning are closely related fields. What is AI?\n")
        
        logger.info(f"Loading MS MARCO data from: {self.msmarco_path}")
        line_count = 0
        with open(self.msmarco_path, 'r', encoding='utf-8') as f_marco:
            for line in f_marco:
                # Assuming each line is a passage or document.
                # If MSMARCO is in a specific format (e.g. tsv with id, passage), parse accordingly.
                # For simplicity, let's assume the relevant text is the whole line or a specific column.
                # Example for TREC Collection (passages):
                # fields = line.strip().split('\t')
                # passage_text = fields[1] # if passage is the second field
                passage_text = line.strip() # Simplistic assumption
                if passage_text:
                    raw_texts.append(passage_text)
                line_count +=1
                if line_count % 100000 == 0:
                    logger.info(f"Read {line_count} lines from MS MARCO source.")

        if not raw_texts:
            logger.error("No text data loaded from MS MARCO file. Please check the file and loading logic.")
            raise ValueError("Failed to load MS MARCO data.")

        logger.info(f"Loaded {len(raw_texts)} passages/documents from MS MARCO.")

        processed_sentences_count = 0
        with open(self.normalized_sentences_file, "w", encoding="utf-8") as fout:
            for i, text in enumerate(tqdm(raw_texts, desc="Normalizing and segmenting")):
                # Normalize the whole passage first
                normalized_passage = self._normalize_text(text)
                if not normalized_passage:
                    continue
                # Then segment into sentences
                # NLTK's sent_tokenize expects original punctuation for better accuracy.
                # So, we tokenize sentences from original text, then normalize each sentence.
                # However, Task1.txt suggests normalizing then segmenting. Let's follow Task1.txt.
                # "After normalization, your text should be a continuous stream..."
                # "Split the text into individual sentences."
                # This implies normalization happens on the larger text block first.
                
                # The instruction "Split the text into individual sentences" after normalization
                # can be tricky if all punctuation is removed.
                # Let's refine: normalize but keep sentence boundaries for sent_tokenize.
                
                # Refined normalization for sentence splitting:
                temp_text = text.lower()
                # Keep sentence-ending punctuation for sent_tokenize
                temp_text = re.sub(r"[^a-z0-9\s.?!]+", " ", temp_text) 
                temp_text = re.sub(r"\s+", " ", temp_text).strip()

                sentences = sent_tokenize(temp_text)
                
                for sent in sentences:
                    # Now, fully normalize each sentence (remove the .?! used for splitting)
                    final_sent = re.sub(r"[.?!]", " ", sent) # Replace sentence enders with space
                    final_sent = re.sub(r"\s+", " ", final_sent).strip() # Normalize spaces again
                    if final_sent:
                        fout.write(final_sent + "\n")
                        processed_sentences_count += 1
                if (i + 1) % 10000 == 0:
                    logger.info(f"Processed {i+1}/{len(raw_texts)} passages.")

        logger.info(f"Step 1 finished. Normalized sentences saved to {self.normalized_sentences_file}. Total sentences: {processed_sentences_count}")

    def step2_train_sentencepiece_tokenizer(self):
        """Trains a Unigram LM tokenizer using SentencePiece."""
        logger.info("Step 2: Training SentencePiece Unigram tokenizer...")
        if os.path.exists(self.sp_model_file) and os.path.exists(self.sp_vocab_file):
            logger.info("SentencePiece model and vocab already exist. Skipping training.")
            return

        if not os.path.exists(self.normalized_sentences_file):
            logger.error(f"{self.normalized_sentences_file} not found. Run data preparation first.")
            raise FileNotFoundError(f"{self.normalized_sentences_file} not found.")

        spm.SentencePieceTrainer.train(
            input=self.normalized_sentences_file,
            model_prefix=self.sp_model_prefix,
            vocab_size=self.sp_vocab_size,
            model_type='unigram',
            character_coverage=1.0, # For English, 0.9995 is often default
            # num_threads=os.cpu_count(), # Use available cores
            # max_sentence_length=10000 # Default is 4192, might need adjustment
        )
        logger.info(f"Step 2 finished. SentencePiece model saved to {self.sp_model_file} and vocab to {self.sp_vocab_file}")

    def step3_extract_ngrams_and_skipgrams(self):
        """Extracts frequent n-grams and skip-grams from the corpus."""
        logger.info("Step 3: Extracting n-grams and skip-grams...")

        if not os.path.exists(self.normalized_sentences_file):
            logger.error(f"{self.normalized_sentences_file} not found. Run data preparation first.")
            raise FileNotFoundError(f"{self.normalized_sentences_file} not found.")

        line_count = 0
        with open(self.normalized_sentences_file, "r", encoding="utf-8") as fin:
            for line in tqdm(fin, desc="Counting n-grams"):
                words = line.strip().split()
                if not words:
                    continue
                
                self.word_counter.update(words)
                
                for i in range(len(words) - 1):
                    self.bigram_counter[f"{words[i]} {words[i+1]}"] += 1
                
                for i in range(len(words) - 2):
                    self.trigram_counter[f"{words[i]} {words[i+1]} {words[i+2]}"] += 1
                    self.skipgram_counter[f"{words[i]} {words[i+2]}"] += 1 # skip-1 bigram
                line_count +=1
        logger.info(f"Initial counts: Unigrams={len(self.word_counter)}, Bigrams={len(self.bigram_counter)}, Trigrams={len(self.trigram_counter)}, Skipgrams={len(self.skipgram_counter)}")

        # Filter by frequency
        self.frequent_bigrams = {bg: c for bg, c in self.bigram_counter.items() if c >= self.min_freq_ngram}
        self.frequent_trigrams = {tg: c for tg, c in self.trigram_counter.items() if c >= self.min_freq_ngram}
        self.frequent_skipgrams = {sg: c for sg, c in self.skipgram_counter.items() if c >= self.min_freq_ngram}
        logger.info(f"After frequency filtering (min_freq={self.min_freq_ngram}): Bigrams={len(self.frequent_bigrams)}, Trigrams={len(self.frequent_trigrams)}, Skipgrams={len(self.frequent_skipgrams)}")

        # Filter out stopword-only phrases
        stop_words = set(stopwords.words('english'))
        def is_meaningful(phrase_str: str) -> bool:
            return any(w not in stop_words for w in phrase_str.split())

        self.frequent_bigrams = {bg: c for bg, c in self.frequent_bigrams.items() if is_meaningful(bg)}
        self.frequent_trigrams = {tg: c for tg, c in self.frequent_trigrams.items() if is_meaningful(tg)}
        # Skipgrams are already 2 words, less likely to be pure stopwords if meaningful
        self.frequent_skipgrams = {sg: c for sg, c in self.frequent_skipgrams.items() if is_meaningful(sg)}
        logger.info(f"After stopword filtering: Bigrams={len(self.frequent_bigrams)}, Trigrams={len(self.frequent_trigrams)}, Skipgrams={len(self.frequent_skipgrams)}")
        
        # Combine candidate lists
        self.candidate_phrases_with_counts = {}
        self.candidate_phrases_with_counts.update(self.frequent_bigrams)
        self.candidate_phrases_with_counts.update(self.frequent_trigrams)
        # For skipgrams, if they are already present as bigrams/trigrams, their counts might differ.
        # Task1.txt: "Also take skip-gram pairs (as bigram phrases) that are not already in the bigram list."
        # This implies skipgrams are treated as bigrams of their outer words.
        # Let's add skipgrams if the phrase itself (as a bigram) isn't already a candidate from bigram_counter.
        for sg, count in self.frequent_skipgrams.items():
            if sg not in self.candidate_phrases_with_counts: # sg is already a "word1 word2" string
                 self.candidate_phrases_with_counts[sg] = count
        
        logger.info(f"Step 3 finished. Total candidate phrases for PMI: {len(self.candidate_phrases_with_counts)}")

    def _compute_pmi(self, phrase_str: str, N_words: int) -> float:
        words = phrase_str.split()
        k = len(words)
        
        phrase_count = self.candidate_phrases_with_counts.get(phrase_str)
        # Fallback for skipgrams if their count was not used directly
        if phrase_count is None and phrase_str in self.skipgram_counter:
             phrase_count = self.skipgram_counter[phrase_str]

        if not phrase_count or phrase_count == 0:
            return -float('inf')

        # P(phrase) = count(phrase) / N_words (approx)
        # P(wi) = count(wi) / N_words
        # PMI = log [ P(phrase) / (P(w1)*P(w2)*...*P(wk)) ]
        # PMI = log [ (count(phrase)/N) / ( (count(w1)/N) * ... * (count(wk)/N) ) ]
        # PMI = log [ count(phrase) * N^(k-1) / (count(w1)*...*count(wk)) ]
        
        numerator = phrase_count * (N_words ** (k - 1))
        denominator = 1
        for w in words:
            count_w = self.word_counter.get(w)
            if not count_w or count_w == 0: return -float('inf') # Should not happen if phrase exists
            denominator *= count_w
        
        if denominator == 0: return -float('inf')
        
        pmi_val = math.log(numerator / denominator)
        # Task1.txt uses average PMI: pmi_val / k
        return pmi_val / k


    def step4_compute_pmi_for_candidates(self):
        """Computes PMI for candidate phrases and selects top N."""
        logger.info("Step 4: Computing PMI for candidate phrases...")
        if not hasattr(self, 'candidate_phrases_with_counts') or not self.candidate_phrases_with_counts:
            logger.error("Candidate phrases not extracted. Run step 3 first.")
            raise ValueError("Candidate phrases not available.")

        N_words = sum(self.word_counter.values())
        if N_words == 0:
            logger.error("Total word count is zero. Cannot compute PMI.")
            raise ValueError("Word counter is empty.")

        pmi_scores = {}
        for phrase_str in tqdm(self.candidate_phrases_with_counts.keys(), desc="Calculating PMI"):
            pmi_scores[phrase_str] = self._compute_pmi(phrase_str, N_words)

        # Filter PMI > 0 and sort
        self.pmi_selected_phrases = {p: s for p, s in pmi_scores.items() if s > 0}
        sorted_by_pmi = sorted(self.pmi_selected_phrases.items(), key=lambda x: x[1], reverse=True)
        
        self.top_pmi_phrases = [phrase for phrase, score in sorted_by_pmi[:self.top_n_pmi]]
        logger.info(f"Step 4 finished. Selected {len(self.top_pmi_phrases)} phrases based on PMI (top {self.top_n_pmi}).")

    def _load_bert_for_cohesion(self):
        if self.bert_tokenizer is None or self.bert_model is None:
            logger.info(f"Loading BERT model for cohesion scoring: {self.bert_model_name}")
            self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)
            self.bert_model = BertForMaskedLM.from_pretrained(self.bert_model_name)
            self.bert_model.to(self.device)
            self.bert_model.eval()

    def _bert_phrase_cohesion(self, phrase_str: str) -> float:
        self._load_bert_for_cohesion()
        words = phrase_str.split()
        if not words: return -float('inf')

        log_prob_sum = 0.0
        num_valid_words = 0

        for i, target_word in enumerate(words):
            masked_words = words[:]
            masked_words[i] = self.bert_tokenizer.mask_token
            masked_text = " ".join(masked_words)

            inputs = self.bert_tokenizer(masked_text, return_tensors='pt', add_special_tokens=True).to(self.device)
            
            # Find the index of the MASK token
            try:
                mask_token_indices = torch.where(inputs['input_ids'][0] == self.bert_tokenizer.mask_token_id)[0]
                if mask_token_indices.nelement() == 0: continue # Should not happen
                mask_index_in_sequence = mask_token_indices[0] # Take the first if multiple (e.g. word itself is [MASK])
            except:
                logger.warning(f"Could not find MASK token for phrase '{phrase_str}', word '{target_word}'. Skipping word.")
                continue


            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            
            logits = outputs.logits[0, mask_index_in_sequence, :] # Logits for the [MASK] position
            probs = torch.softmax(logits, dim=-1)
            
            target_word_ids = self.bert_tokenizer.encode(target_word, add_special_tokens=False)
            if not target_word_ids: 
                logger.warning(f"BERT tokenizer could not encode target word '{target_word}' in phrase '{phrase_str}'. Skipping word.")
                continue
            
            # If target_word is tokenized into multiple subwords by BERT, this scoring is tricky.
            # Task1.txt: "target_id = tokenizer.convert_tokens_to_ids(target_word)"
            # This assumes target_word is a single token in BERT's vocab.
            # If not, we might need a more complex way (e.g. average prob of subwords, or prob of first subword)
            # For simplicity, let's use the ID of the first subword if it's multi-subword.
            target_id = target_word_ids[0]
            
            prob_target_word = probs[target_id].item()

            if prob_target_word < 1e-12: prob_target_word = 1e-12 # Avoid log(0)
            log_prob_sum += math.log(prob_target_word)
            num_valid_words +=1
        
        if num_valid_words == 0: return -float('inf')
        return log_prob_sum / num_valid_words


    def step5_bert_cohesion_scoring(self):
        """Scores candidate phrases using BERT-based cohesion."""
        logger.info("Step 5: Scoring phrases with BERT-based cohesion...")
        
        # We score phrases that passed PMI filtering or a broader set of candidates.
        # Task1.txt: "Take the list of candidate phrases filtered by PMI ... and compute ... BERT cohesion score"
        # Let's use self.pmi_selected_phrases keys if available, otherwise all candidates.
        phrases_to_score_bert = []
        if hasattr(self, 'pmi_selected_phrases') and self.pmi_selected_phrases:
            phrases_to_score_bert = list(self.pmi_selected_phrases.keys())
            logger.info(f"Scoring {len(phrases_to_score_bert)} phrases that passed PMI filtering.")
        elif hasattr(self, 'candidate_phrases_with_counts') and self.candidate_phrases_with_counts:
            phrases_to_score_bert = list(self.candidate_phrases_with_counts.keys()) # This could be very large
            logger.warning(f"PMI selected phrases not found. Scoring all {len(phrases_to_score_bert)} candidate phrases with BERT. This might be slow.")
            # Optionally, limit this list if it's too big, e.g., by taking top N by frequency first.
            # phrases_to_score_bert = sorted(self.candidate_phrases_with_counts, key=self.candidate_phrases_with_counts.get, reverse=True)[:2*self.top_n_bert] # Example limit
        else:
            logger.error("No candidate phrases available for BERT cohesion scoring.")
            raise ValueError("No candidate phrases.")

        bert_scores = {}
        # Consider batching for BERT scoring if performance is an issue.
        # The provided _bert_phrase_cohesion is one-by-one.
        for phrase_str in tqdm(phrases_to_score_bert, desc="BERT Cohesion Scoring"):
            bert_scores[phrase_str] = self._bert_phrase_cohesion(phrase_str)
        
        sorted_by_bert = sorted(bert_scores.items(), key=lambda x: x[1], reverse=True)
        self.top_bert_phrases = [phrase for phrase, score in sorted_by_bert[:self.top_n_bert]]
        logger.info(f"Step 5 finished. Selected {len(self.top_bert_phrases)} phrases based on BERT cohesion (top {self.top_n_bert}).")

    def step6_merge_selected_phrases(self):
        """Merges top phrases from PMI and BERT scoring."""
        logger.info("Step 6: Merging selected phrases...")
        if not hasattr(self, 'top_pmi_phrases') or not hasattr(self, 'top_bert_phrases'):
            logger.error("Top PMI or Top BERT phrases not available. Run steps 4 and 5 first.")
            raise ValueError("Missing PMI/BERT phrase lists.")

        self.selected_phrases = list(set(self.top_pmi_phrases) | set(self.top_bert_phrases))
        # Optional: sort for deterministic output
        self.selected_phrases.sort() 
        logger.info(f"Step 6 finished. Total unique selected phrases: {len(self.selected_phrases)}")

    def step7_build_trie_tokenizer(self):
        """Builds a trie for the hybrid vocabulary (conceptual step)."""
        # The actual trie construction and tokenization logic will be part of the tokenizer
        # that uses the final vocabulary. Here, we just acknowledge the concept.
        # The `vocab_final.txt` produced in step 8 is the key output for such a tokenizer.
        logger.info("Step 7: Trie-based tokenizer concept.")
        logger.info("The final vocabulary file can be used to build a trie for longest-match tokenization.")
        # Implementation of Trie and tokenizer class would go here or in a separate module.
        # For this script, the main output is vocab_final.txt.
        pass # Actual trie data structure not stored in this builder class for now.

    def step8_output_final_vocabulary(self):
        """Outputs the final vocabulary to vocab_final.txt."""
        logger.info("Step 8: Outputting final vocabulary...")
        if not os.path.exists(self.sp_vocab_file):
            logger.error(f"SentencePiece vocab {self.sp_vocab_file} not found. Run step 2.")
            raise FileNotFoundError(f"{self.sp_vocab_file} not found.")
        if not hasattr(self, 'selected_phrases'):
            logger.error("Selected phrases not available. Run step 6.")
            raise ValueError("Selected phrases list is missing.")

        final_vocab_list = []
        
        # 1. Special tokens
        # Task1.txt: [PAD], [UNK], [CLS], [SEP], [MASK]
        # BERT default: [PAD]=0, [UNK]=100, [CLS]=101, [SEP]=102, [MASK]=103
        # We can define our own order.
        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        final_vocab_list.extend(special_tokens)

        # 2. Subword tokens from SentencePiece
        # SentencePiece .vocab format: <token>\t<score>
        # <unk> is usually 0th, <s> 1st, </s> 2nd if not specified otherwise.
        # Our SP training didn't specify <s>, </s>. It has <unk>.
        # We replace SP's <unk> with our [UNK].
        sp_subwords = []
        with open(self.sp_vocab_file, "r", encoding="utf-8") as f_spv:
            for line in f_spv:
                token, score = line.strip().split('\t')
                if token == "<unk>": # Skip SP's <unk>, we use our [UNK]
                    continue
                # SentencePiece uses U+2581 (LOWER ONE EIGHTH BLOCK) for word boundary.
                # Task1.txt: "token_str = token.replace(' ', ' ')" - convert SP marker to actual space
                token_str = token.replace('\u2581', ' ') 
                sp_subwords.append(token_str)
        final_vocab_list.extend(sp_subwords)
        logger.info(f"Added {len(special_tokens)} special tokens and {len(sp_subwords)} subword tokens.")
        
        # 3. Phrase tokens
        # Task1.txt: "final_vocab.append(" " + phrase)" - ensure leading space for phrases
        num_added_phrases = 0
        for phrase_str in self.selected_phrases:
            # Ensure phrase is not already covered by subwords (e.g. if a "phrase" is a single word already in SP vocab)
            # This check can be complex. For now, assume selected_phrases are multi-word or distinct.
            token_to_add = " " + phrase_str # Add leading space as per Task1.txt
            if token_to_add not in final_vocab_list: # Avoid duplicates if a phrase somehow matches a subword string
                final_vocab_list.append(token_to_add)
                num_added_phrases += 1
        logger.info(f"Added {num_added_phrases} unique phrase tokens.")

        with open(self.final_vocab_file, "w", encoding="utf-8") as fout:
            for token in final_vocab_list:
                fout.write(token + "\n")
        
        logger.info(f"Step 8 finished. Final vocabulary saved to {self.final_vocab_file}. Total tokens: {len(final_vocab_list)}")

    def _load_embedding_model(self):
        """Loads the model and tokenizer for embedding generation."""
        if self.embedding_model is None or self.embedding_tokenizer is None:
            logger.info(f"Loading embedding model and tokenizer: {self.embedding_model_name}")
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
            self.embedding_model = AutoModel.from_pretrained(self.embedding_model_name)
            self.embedding_model.to(self.device)
            self.embedding_model.eval()

    def _mean_pooling(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Performs mean pooling on the last hidden state, taking into account the attention mask.
        Special tokens [CLS] and [SEP] are excluded from the count for the denominator,
        as suggested by Task2.md.
        """
        token_embeddings = model_output.last_hidden_state  # Corrected: remove [0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum embeddings where attention_mask is 1
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        
        # Sum of mask elements to get count of non-padding tokens for each sequence
        sum_mask = attention_mask.sum(1)
        
        # Subtract 2 for [CLS] and [SEP] tokens, clamp to avoid division by zero
        # as per Task2.md description for msmarco-distilbert-dot-v5 mean pooling.
        # This assumes [CLS] and [SEP] are always present and part of the attention_mask sum.
        num_content_tokens = torch.clamp(sum_mask - 2, min=1e-9)
        
        mean_embeddings = sum_embeddings / num_content_tokens.unsqueeze(-1)
        return mean_embeddings

    def build_vocabulary(self):
        """Runs the full vocabulary building pipeline."""
        logger.info("Starting vocabulary building pipeline...")
        self.step1_prepare_data()
        self.step2_train_sentencepiece_tokenizer()
        self.step3_extract_ngrams_and_skipgrams()
        self.step4_compute_pmi_for_candidates()
        self.step5_bert_cohesion_scoring()
        self.step6_merge_selected_phrases()
        self.step7_build_trie_tokenizer() # Conceptual step
        self.step8_output_final_vocabulary()
        self.step9_generate_bert_embeddings()
        logger.info("Vocabulary building pipeline completed successfully.")

    def step9_generate_bert_embeddings(self):
        """
        Generates BERT embeddings for each token in the final vocabulary using batching
        and specified pooling, then saves them as a .npy file.
        """
        logger.info("Step 9: Generating BERT embeddings for the final vocabulary...")

        if not os.path.exists(self.final_vocab_file):
            logger.error(f"Final vocabulary file {self.final_vocab_file} not found. Run step 8 first.")
            raise FileNotFoundError(f"Final vocabulary file {self.final_vocab_file} not found.")

        self._load_embedding_model() # Load the specified embedding model and tokenizer

        vocab_list = []
        with open(self.final_vocab_file, "r", encoding="utf-8") as f:
            vocab_list = [line.strip() for line in f]
        
        if not vocab_list:
            logger.error("Vocabulary list is empty. Cannot generate embeddings.")
            raise ValueError("Vocabulary list is empty.")

        logger.info(f"Loaded {len(vocab_list)} tokens from {self.final_vocab_file} for embedding generation.")

        hidden_size = self.embedding_model.config.hidden_size
        all_embeddings_np = np.zeros((len(vocab_list), hidden_size), dtype=np.float32)
        
        # Find index of [PAD] token if it exists, to handle its embedding separately
        pad_token_str = "[PAD]"
        pad_token_idx = -1
        if pad_token_str in vocab_list:
            pad_token_idx = vocab_list.index(pad_token_str)
            # Embedding for [PAD] is already zeros by np.zeros initialization

        logger.info(f"Processing vocabulary in batches of {self.embedding_batch_size}...")
        with torch.no_grad():
            for i in tqdm(range(0, len(vocab_list), self.embedding_batch_size), desc="Generating Embeddings"):
                batch_token_strings = vocab_list[i : i + self.embedding_batch_size]
                current_batch_indices = list(range(i, min(i + self.embedding_batch_size, len(vocab_list))))

                # Filter out the [PAD] token string from model processing, its embedding is already zero
                # Keep track of original positions to place embeddings correctly
                tokens_to_process_in_model = []
                original_indices_for_model_batch = []

                for k, token_str in enumerate(batch_token_strings):
                    original_idx = current_batch_indices[k]
                    if original_idx == pad_token_idx: # Already handled by np.zeros
                        continue
                    tokens_to_process_in_model.append(token_str)
                    original_indices_for_model_batch.append(original_idx)
                
                if not tokens_to_process_in_model: # Batch might only contain [PAD] or be empty
                    continue

                inputs = self.embedding_tokenizer(
                    tokens_to_process_in_model,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    add_special_tokens=True, # Ensure [CLS] and [SEP] for pooling logic
                    max_length=self.embedding_tokenizer.model_max_length
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                model_output = self.embedding_model(**inputs)
                
                if self.embedding_pooling_mode == 'cls':
                    batch_embeddings = model_output.last_hidden_state[:, 0, :]
                elif self.embedding_pooling_mode == 'mean':
                    batch_embeddings = self._mean_pooling(model_output, inputs['attention_mask'])
                else:
                    logger.error(f"Unknown pooling mode: {self.embedding_pooling_mode}")
                    raise ValueError(f"Unknown pooling mode: {self.embedding_pooling_mode}")
                
                # Move to CPU and convert to NumPy
                batch_embeddings_np = batch_embeddings.cpu().numpy().astype(np.float32)
                
                # Place these embeddings into the correct rows in all_embeddings_np
                for j, original_idx in enumerate(original_indices_for_model_batch):
                    all_embeddings_np[original_idx] = batch_embeddings_np[j]

        np.save(self.embeddings_file, all_embeddings_np)
        logger.info(f"Step 9 finished. Vocabulary embeddings saved to {self.embeddings_file}. Shape: {all_embeddings_np.shape}")


def main():
    parser = argparse.ArgumentParser(description="Build a hybrid vocabulary for BERT approximation.")
    parser.add_argument(
        "--msmarco_path", 
        type=str, 
        default="./datasets/ms_marco_tiny/collection_tiny.tsv", 
        help="Path to MS MARCO data file (e.g., collection.tsv or a preprocessed text file)."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="vocabulary_debug", 
        help="Directory to save vocabulary files."
    )
    parser.add_argument(
        "--sp_vocab_size", 
        type=int, 
        default=20000, 
        help="Target vocabulary size for SentencePiece Unigram model."
    )
    parser.add_argument(
        "--min_freq_ngram", 
        type=int, 
        default=5, 
        help="Minimum frequency for n-grams/skip-grams to be considered."
    )
    parser.add_argument(
        "--top_n_pmi", 
        type=int, 
        default=5000, 
        help="Number of top phrases to select based on PMI."
    )
    parser.add_argument(
        "--top_n_bert", 
        type=int, 
        default=5000, 
        help="Number of top phrases to select based on BERT cohesion score."
    )
    parser.add_argument(
        "--bert_model_name", 
        type=str, 
        default='bert-base-uncased', 
        help="Pre-trained BERT model to use for cohesion scoring."
    )
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default='sentence-transformers/msmarco-distilbert-dot-v5',
        help="Pre-trained model to use for generating final vocabulary embeddings."
    )
    parser.add_argument(
        "--embedding_pooling_mode",
        type=str,
        default='mean',
        choices=['mean', 'cls'],
        help="Pooling mode for final vocabulary embeddings ('mean' or 'cls')."
    )
    parser.add_argument(
        "--embedding_batch_size",
        type=int,
        default=64, # Adjusted default
        help="Batch size for generating final vocabulary embeddings."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda", 
        help="Device for PyTorch operations (e.g., 'cuda', 'cpu'). Autodetects if None and no default set here."
    )
    
    args = parser.parse_args()

    builder = VocabularyBuilder(
        msmarco_path=args.msmarco_path,
        output_dir=args.output_dir,
        sp_vocab_size=args.sp_vocab_size,
        min_freq_ngram=args.min_freq_ngram,
        top_n_pmi=args.top_n_pmi,
        top_n_bert=args.top_n_bert,
        bert_model_name=args.bert_model_name,
        embedding_model_name=args.embedding_model_name,
        embedding_pooling_mode=args.embedding_pooling_mode,
        embedding_batch_size=args.embedding_batch_size,
        device=args.device
    )

    builder.build_vocabulary()

if __name__ == "__main__":
    main()
