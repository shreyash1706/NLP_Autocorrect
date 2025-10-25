"""
simple_tokenizer_model.py

Complete tokenizer model that can be saved/loaded as a single unit
- Contains both vocabulary and dictionary
- Can tokenize any new sentence
- Easy to use in pipelines
"""

import os
import pickle
from collections import Counter

class SimpleTokenizerModel:
    """
    Complete tokenizer model that handles everything
    """
    def __init__(self, vocab=None, dictionary=None, special_tokens=None):
        self.vocab = vocab or {}
        self.dictionary = dictionary or set()
        self.special_tokens = special_tokens or ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
    def tokenize(self, text):
        """Tokenize any text into tokens"""
        basic_tokens = self._simple_tokenize(text)
        final_tokens = []
        
        for token in basic_tokens:
            if token in self.vocab:
                final_tokens.append(token)
            else:
                # Try to split concatenated words
                split_tokens = self._split_concatenated_word(token)
                for split_token in split_tokens:
                    if split_token in self.vocab:
                        final_tokens.append(split_token)
                    else:
                        final_tokens.append('[UNK]')
        
        return final_tokens
    
    def encode(self, text):
        """Convert text directly to numerical IDs"""
        tokens = self.tokenize(text)
        return self.convert_tokens_to_ids(tokens)
    
    def decode(self, ids):
        """Convert numerical IDs back to text"""
        tokens = self.convert_ids_to_tokens(ids)
        return " ".join(tokens)
    
    def convert_tokens_to_ids(self, tokens):
        """Convert tokens to numerical IDs"""
        return [self.vocab.get(token, self.vocab['[UNK]']) for token in tokens]
    
    def convert_ids_to_tokens(self, ids):
        """Convert IDs back to tokens"""
        return [self.inverse_vocab.get(id, '[UNK]') for id in ids]
    
    def _simple_tokenize(self, text):
        """Internal tokenization method"""
        text = text.lower().strip()
        if not text:
            return []
        
        tokens = []
        word = ""
        
        for char in text:
            if char.isalnum() or char == "'":
                word += char
            else:
                if word:
                    tokens.append(word)
                    word = ""
                if char.strip():
                    tokens.append(char)
        
        if word:
            tokens.append(word)
        
        return tokens
    
    def _split_concatenated_word(self, word, max_word_length=15):
        """Split glued words using the dictionary"""
        word = word.lower()
        
        if word in self.dictionary or len(word) <= 3:
            return [word] if word in self.dictionary else ['[UNK]']
        
        # Backtracking split
        def backtrack_split(remaining, current_split):
            if not remaining:
                return current_split
            
            for length in range(min(max_word_length, len(remaining)), 0, -1):
                segment = remaining[:length]
                if segment in self.dictionary:
                    result = backtrack_split(remaining[length:], current_split + [segment])
                    if result:
                        return result
            return None
        
        result = backtrack_split(word, [])
        if result:
            return result
        
        # Greedy fallback
        tokens = []
        start = 0
        while start < len(word):
            found = False
            for end in range(min(len(word), start + max_word_length), start, -1):
                subword = word[start:end]
                if subword in self.dictionary:
                    tokens.append(subword)
                    start = end
                    found = True
                    break
            
            if not found:
                tokens.append('[UNK]')
                start += min(5, len(word) - start)
        
        return tokens
    
    def save_model(self, model_path):
        """Save complete tokenizer model as a single file"""
        model_data = {
            'vocab': self.vocab,
            'dictionary': self.dictionary,
            'special_tokens': self.special_tokens
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Saved tokenizer model to {model_path}")
    
    @classmethod
    def load_model(cls, model_path):
        """Load complete tokenizer model from file"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        return cls(
            vocab=model_data['vocab'],
            dictionary=model_data['dictionary'],
            special_tokens=model_data['special_tokens']
        )
    
    def get_vocab_size(self):
        """Get vocabulary size"""
        return len(self.vocab)
    
    def get_special_tokens(self):
        """Get special tokens"""
        return self.special_tokens

# -----------------------------
# Usage Examples
# -----------------------------

def create_and_save_tokenizer():
    """Create tokenizer from your existing files"""
    # Load from your existing vocab and dictionary files
    vocab = {}
    dictionary = set()
    
    # Load vocab (word -> ID)
    with open("tokenizer_simple/vocab.txt", 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                vocab[parts[0]] = int(parts[1])
    
    # Load dictionary (words for splitting)
    with open("tokenizer_simple/dictionary.txt", 'r', encoding='utf-8') as f:
        for line in f:
            dictionary.add(line.strip())
    
    # Create the model
    tokenizer = SimpleTokenizerModel(vocab, dictionary)
    
    # Save as single model file
    tokenizer.save_model("tokenizer_model.pkl")
    
    return tokenizer

def demonstrate_tokenizer_usage():
    """Show how to use the tokenizer in pipelines"""
    
    # Load the model (after you've created it)
    tokenizer = SimpleTokenizerModel.load_model("tokenizer_model.pkl")
    
    print("=== Tokenizer Model Demo ===")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Test sentences
    test_sentences = [
        "shewastryingtodo her homework",
        "whattimeisit now?",
        "hello world! this is a test.",
        "they're going to the store."
    ]
    
    for sentence in test_sentences:
        print(f"\nInput: '{sentence}'")
        
        # Different ways to tokenize
        tokens = tokenizer.tokenize(sentence)
        ids = tokenizer.encode(sentence)
        decoded = tokenizer.decode(ids)
        
        print(f"Tokens: {tokens}")
        print(f"IDs:    {ids}")
        print(f"Decoded: '{decoded}'")

def tokenize_datasets_with_model():
    """Tokenize all datasets using the model"""
    tokenizer = SimpleTokenizerModel.load_model("tokenizer_model.pkl")
    
    datasets = [
        ("corpora/splits/train.txt", "tokenizer_simple/train_ids.txt"),
        ("corpora/splits/test.txt", "tokenizer_simple/test_ids.txt"),
        ("corpora/splits/val.txt", "tokenizer_simple/val_ids.txt")
    ]
    
    for input_file, output_file in datasets:
        if os.path.exists(input_file):
            print(f"Tokenizing {input_file}...")
            
            id_lines = []
            with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        ids = tokenizer.encode(line)
                        id_lines.append(" ".join(map(str, ids)))
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for line in id_lines:
                    f.write(line + '\n')
            
            print(f"Saved to {output_file}")

if __name__ == "__main__":
    # Step 1: Create the model from your existing files
    tokenizer = create_and_save_tokenizer()
    
    # Step 2: Demonstrate usage
    demonstrate_tokenizer_usage()
    
    # Step 3: Tokenize datasets (optional)
    # tokenize_datasets_with_model()