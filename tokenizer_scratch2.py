"""
simple_hardcoded_tokenizer.py

Implements a simple word tokenizer from scratch:
- Reads corpus files efficiently for 1GB corpus
- Tokenizes sentences into words using basic string operations
- Builds a simple vocabulary from training data only
- Dynamically splits concatenated words using corpus-based dictionary
- Saves vocabulary to use for tokenizing train/test/val sets later
"""

import os
import re
from collections import Counter
import mmap

# -----------------------------
# Config
# -----------------------------
TRAIN_CORPUS = "corpora/splits/train.txt"
TEST_CORPUS = "corpora/splits/test.txt"  # for later use
VAL_CORPUS = "corpora/splits/val.txt"    # for later use
MIN_FREQ = 2
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
OUTPUT_VOCAB = "tokenizer_simple/vocab.txt"
OUTPUT_DICT = "tokenizer_simple/dictionary.txt"

os.makedirs(os.path.dirname(OUTPUT_VOCAB), exist_ok=True)

# -----------------------------
# Step 1: Efficient corpus reading for large files
# -----------------------------
def read_corpus_efficient(file_path):
    """
    Efficiently reads large corpus files using memory mapping
    """
    word_freq = Counter()
    
    if not os.path.exists(file_path):
        print(f"Warning: Corpus file {file_path} not found")
        return word_freq
        
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                buffer = ""
                chunk_size = 1024 * 1024  # 1MB chunks
                
                while True:
                    chunk = mmapped_file.read(chunk_size).decode('utf-8', errors='ignore')
                    if not chunk:
                        break
                    
                    # Process complete lines to avoid breaking words
                    lines = (buffer + chunk).split('\n')
                    buffer = lines[-1]  # Save incomplete line for next chunk
                    
                    for line in lines[:-1]:
                        line = line.strip()
                        if line:
                            tokens = simple_tokenize(line)
                            word_freq.update(tokens)
                
                # Process remaining buffer
                if buffer.strip():
                    tokens = simple_tokenize(buffer.strip())
                    word_freq.update(tokens)
                    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return word_freq

# -----------------------------
# Step 2: Simple tokenizer
# -----------------------------
def simple_tokenize(text):
    """
    Tokenizes a string into words:
    - Lowercases everything
    - Splits by spaces and punctuation
    - Handles apostrophes in contractions
    """
    text = text.lower().strip()
    if not text:
        return []
    
    # Enhanced tokenization that handles contractions better
    tokens = []
    word = ""
    
    for i, char in enumerate(text):
        if char.isalnum() or char == "'":
            word += char
        else:
            if word:
                # Handle common contractions
                if word.endswith("'s") or word.endswith("'t") or word.endswith("'d") or word.endswith("'ll") or word.endswith("'re") or word.endswith("'ve"):
                    tokens.append(word)
                else:
                    tokens.append(word)
                word = ""
            if char.strip():  # non-whitespace punctuation
                tokens.append(char)
    
    if word:
        tokens.append(word)
    
    return tokens

# -----------------------------
# Step 3: Build vocabulary and dictionary from training data only
# -----------------------------
def build_vocab_from_train(train_files, min_freq=2):
    """
    Build vocabulary ONLY from training data
    """
    print("Reading training corpus to build vocabulary...")
    word_freq = Counter()
    
    for file_path in train_files:
        if os.path.exists(file_path):
            print(f"Processing {file_path}...")
            file_freq = read_corpus_efficient(file_path)
            word_freq.update(file_freq)
            print(f"  Added {len(file_freq)} unique words from {file_path}")
    
    print(f"Total unique words in training corpus: {len(word_freq)}")
    
    # Build vocabulary (including special tokens)
    vocab = set(SPECIAL_TOKENS)
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab.add(word)
    
    # Build dictionary for splitting (only reasonably sized words)
    dictionary = set()
    for word, freq in word_freq.items():
        if freq >= min_freq and 2 <= len(word) <= 15:  # Reasonable word lengths
            dictionary.add(word)
    
    # Add special tokens to dictionary for completeness
    for token in SPECIAL_TOKENS:
        dictionary.add(token)
    
    # Create word to index mapping
    vocab_list = sorted(vocab)
    vocab_dict = {word: idx for idx, word in enumerate(vocab_list)}
    
    return vocab_dict, dictionary, word_freq

# -----------------------------
# Step 4: Improved concatenated word splitting
# -----------------------------
def split_concatenated_word(word, dictionary, max_word_length=15):
    """
    Improved greedy longest-match approach to split concatenated words
    with backtracking for better results
    """
    word = word.lower()
    
    # If it's already in dictionary or too short, return as is
    if word in dictionary or len(word) <= 3:
        return [word] if word in dictionary else ['[UNK]']
    
    # Try to split the word
    def backtrack_split(remaining, current_split):
        if not remaining:
            return current_split
        
        # Try all possible splits from longest to shortest
        for length in range(min(max_word_length, len(remaining)), 0, -1):
            segment = remaining[:length]
            if segment in dictionary:
                result = backtrack_split(remaining[length:], current_split + [segment])
                if result:
                    return result
        
        return None
    
    # Try backtracking first
    result = backtrack_split(word, [])
    if result:
        return result
    
    # Fallback: greedy approach
    tokens = []
    start = 0
    while start < len(word):
        found = False
        # Try from longest to shortest substring
        for end in range(min(len(word), start + max_word_length), start, -1):
            subword = word[start:end]
            if subword in dictionary:
                tokens.append(subword)
                start = end
                found = True
                break
        
        if not found:
            # If no dictionary word found, take reasonable chunks
            chunk_size = min(5, len(word) - start)
            tokens.append('[UNK]')  # or tokens.append(word[start:start+chunk_size])
            start += chunk_size
    
    return tokens

# -----------------------------
# Step 5: Tokenizer class for consistent usage
# -----------------------------
class SimpleTokenizer:
    def __init__(self, vocab=None, dictionary=None):
        self.vocab = vocab or {}
        self.dictionary = dictionary or set()
        self.inverse_vocab = {v: k for k, v in vocab.items()} if vocab else {}
    
    def tokenize(self, text):
        """Tokenize text using the learned vocabulary"""
        basic_tokens = simple_tokenize(text)
        final_tokens = []
        
        for token in basic_tokens:
            if token in self.vocab:
                final_tokens.append(token)
            else:
                # Try to split concatenated words
                split_tokens = split_concatenated_word(token, self.dictionary)
                for split_token in split_tokens:
                    if split_token in self.vocab:
                        final_tokens.append(split_token)
                    else:
                        final_tokens.append('[UNK]')
        
        return final_tokens
    
    def convert_tokens_to_ids(self, tokens):
        """Convert tokens to their corresponding IDs"""
        return [self.vocab.get(token, self.vocab['[UNK]']) for token in tokens]
    
    def convert_ids_to_tokens(self, ids):
        """Convert IDs back to tokens"""
        return [self.inverse_vocab.get(id, '[UNK]') for id in ids]
    
    def save(self, vocab_path, dict_path):
        """Save vocabulary and dictionary"""
        with open(vocab_path, 'w', encoding='utf-8') as f:
            for word, idx in sorted(self.vocab.items(), key=lambda x: x[1]):
                f.write(f"{word}\t{idx}\n")
        
        with open(dict_path, 'w', encoding='utf-8') as f:
            for word in sorted(self.dictionary):
                f.write(f"{word}\n")
    
    @classmethod
    def load(cls, vocab_path, dict_path):
        """Load vocabulary and dictionary"""
        vocab = {}
        dictionary = set()
        
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    vocab[parts[0]] = int(parts[1])
        
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                dictionary.add(line.strip())
        
        return cls(vocab, dictionary)

# -----------------------------
# Step 6: Main processing - Build from training data only
# -----------------------------
def main():
    print("=== Building Tokenizer from Training Data ===")
    
    # Step 1: Build vocabulary from TRAINING data only
    train_files = [TRAIN_CORPUS]
    vocab, dictionary, word_freq = build_vocab_from_train(train_files, MIN_FREQ)
    
    print(f"Final vocabulary size: {len(vocab)}")
    print(f"Dictionary size for splitting: {len(dictionary)}")
    
    # Step 2: Create tokenizer
    tokenizer = SimpleTokenizer(vocab, dictionary)
    
    # Step 3: Save vocabulary and dictionary for later use
    tokenizer.save(OUTPUT_VOCAB, OUTPUT_DICT)
    print(f"Saved vocabulary to: {OUTPUT_VOCAB}")
    print(f"Saved dictionary to: {OUTPUT_DICT}")
    
    # Step 4: Demonstrate usage
    print("\n=== Testing Tokenizer ===")
    test_sentences = [
        "shewastryingtodo her homework",
        "whattimeisit now",
        "hello world! this is a test.",
        "they're going to the park, aren't they?"
    ]
    
    for sentence in test_sentences:
        tokens = tokenizer.tokenize(sentence)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        print(f"Text: {sentence}")
        print(f"Tokens: {tokens}")
        print(f"IDs: {ids}")
        print()
    
    # Step 5: Show vocabulary statistics
    print("=== Vocabulary Statistics ===")
    print(f"Total vocabulary size: {len(vocab)}")
    print(f"Special tokens: {SPECIAL_TOKENS}")
    
    # Show most common words
    common_words = [(word, freq) for word, freq in word_freq.items() if word in vocab]
    common_words.sort(key=lambda x: x[1], reverse=True)
    print(f"\nTop 10 most common words:")
    for word, freq in common_words[:10]:
        print(f"  {word}: {freq}")

# -----------------------------
# Step 7: Functions for later use with test/val sets
# -----------------------------
def tokenize_dataset(file_path, tokenizer, output_path):
    """Tokenize a complete dataset file and save results"""
    print(f"Tokenizing {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"File {file_path} not found")
        return
    
    tokenized_lines = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                tokens = tokenizer.tokenize(line)
                tokenized_lines.append(" ".join(tokens))
            
            if line_num % 10000 == 0:
                print(f"  Processed {line_num} lines...")
    
    # Save tokenized output
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in tokenized_lines:
            f.write(line + '\n')
    
    print(f"Saved tokenized data to {output_path}")

def tokenize_all_datasets():
    """Tokenize train, test, and validation sets using the saved tokenizer"""
    print("=== Tokenizing All Datasets ===")
    
    # Load the tokenizer we built from training data
    tokenizer = SimpleTokenizer.load(OUTPUT_VOCAB, OUTPUT_DICT)
    print(f"Loaded tokenizer with vocabulary size: {len(tokenizer.vocab)}")
    
    # Tokenize each dataset
    datasets = [
        (TRAIN_CORPUS, "tokenizer_simple/train_tokenized.txt"),
        (TEST_CORPUS, "tokenizer_simple/test_tokenized.txt"), 
        (VAL_CORPUS, "tokenizer_simple/val_tokenized.txt")
    ]
    
    for input_file, output_file in datasets:
        if os.path.exists(input_file):
            tokenize_dataset(input_file, tokenizer, output_file)
        else:
            print(f"Skipping {input_file} - file not found")

if __name__ == "__main__":
    # First: Build vocabulary from training data only
    #main()
    
    # Later: Use this to tokenize all datasets
    tokenize_all_datasets()