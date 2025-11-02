"""
test_wordpiece_tokenizer.py
Test the WordPiece tokenizer - MUST HAVE THE SAME CLASS DEFINITION
"""

import pickle
import re

class FastWordPieceTokenizer:
    def __init__(self, vocab=None):
        self.vocab = vocab or {}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.unk_token_id = self.vocab.get("[unk]", 1)
        
        # Precompute common patterns for speed
        self.common_prefixes = {'un', 're', 'pre', 'dis', 'mis', 'non', 'over', 'under'}
        self.common_suffixes = {'ing', 'ed', 'ly', 'es', 's', 'er', 'est', 'ment', 'ness'}
    
    def _fast_split_word(self, word):
        """FAST WordPiece-like splitting with common patterns"""
        word = word.lower()
        
        # Check if whole word is in vocabulary (fastest path)
        if word in self.vocab:
            return [word]
        
        # Try common prefixes first (most common case)
        for prefix in self.common_prefixes:
            if word.startswith(prefix) and len(word) > len(prefix) + 2:
                suffix = word[len(prefix):]
                if suffix in self.vocab:
                    return [prefix, suffix]
        
        # Try common suffixes
        for suffix in self.common_suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                prefix = word[:-len(suffix)]
                if prefix in self.vocab:
                    return [prefix, suffix]
        
        # Greedy longest-match-first splitting
        tokens = []
        start = 0
        max_attempts = len(word) * 2  # Prevent infinite loops
        
        while start < len(word) and max_attempts > 0:
            max_attempts -= 1
            
            # Find longest substring that's in vocabulary
            end = len(word)
            found = False
            
            while end > start:
                substr = word[start:end]
                if substr in self.vocab:
                    tokens.append(substr)
                    start = end
                    found = True
                    break
                end -= 1
            
            if not found:
                # Can't split this part, use characters or unk
                if len(word) - start <= 3:
                    tokens.extend(list(word[start:]))
                else:
                    tokens.append('[unk]')
                break
        
        return tokens if tokens else ['[unk]']
    
    def tokenize(self, text):
        """FAST tokenization with WordPiece splitting"""
        if not text or not text.strip():
            return []
        
        text = text.strip()
        
        # First, split into words and punctuation using regex
        basic_tokens = re.findall(r'\w+|[^\w\s]', text)
        
        # Apply WordPiece splitting to each word token
        final_tokens = []
        for token in basic_tokens:
            if token.isalnum() and len(token) > 1:
                # Split using WordPiece
                sub_tokens = self._fast_split_word(token)
                final_tokens.extend(sub_tokens)
            else:
                # Keep punctuation and single characters as-is
                final_tokens.append(token.lower())
        
        return final_tokens
    
    def encode(self, text):
        """Convert text to token IDs"""
        tokens = self.tokenize(text)
        token_ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
        return tokens, token_ids
    
    def encode_ids_only(self, text):
        """Convert text to token IDs only (fastest)"""
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]
    
    def decode(self, token_ids):
        """Convert token IDs back to text"""
        tokens = [self.inverse_vocab.get(token_id, '[unk]') for token_id in token_ids]
        return " ".join(tokens)
    
    def get_token_mapping(self, text):
        """Get detailed mapping of text to tokens and their IDs"""
        tokens, token_ids = self.encode(text)
        mapping = []
        for token, token_id in zip(tokens, token_ids):
            mapping.append({
                'token': token,
                'token_id': token_id,
                'status': 'KNOWN' if token in self.vocab else 'UNKNOWN'
            })
        return mapping

# Now load and test
def main():
    print("🚀 Loading WordPiece tokenizer...")
    
    # Load the tokenizer
    with open("wikitext2_tokenizer_wordpiece.pkl", 'rb') as f:
        tokenizer = pickle.load(f)
    
    print("✅ WordPiece Tokenizer loaded successfully!")
    print(f"📊 Vocabulary size: {len(tokenizer.vocab):,} tokens")
    
    # Test sentences - specifically chosen to show WordPiece splitting
    test_sentences = [
        # Basic sentences
        "Hello world! This is a test.",
        "The quick brown fox jumps over the lazy dog.",
        
        # WordPiece splitting examples
        "unhappiness misunderstanding rediscovery",
        "antidisestablishmentarianism",
        "unbelievable preprocessing",
        
        # Contractions and punctuation
        "I can't believe it's working!",
        "The company's revenue increased by 15%",
        
        # Mixed case to test case insensitivity
        "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
        "Hello WORLD! This IS a TEST.",
        
        # Special characters and URLs
        "Email: test@example.com - visit https://site.org",
        "Temperature: 25°C, price: $19.99",
        
        # Edge cases
        "abcdefghijklmnopqrstuvwxyz",  # Long unknown word
        "a i",  # Single letters
        "!!!",  # Only punctuation
    ]
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n{'='*80}")
        print(f"📝 TEST {i}: '{sentence}'")
        print(f"{'='*80}")
        
        # Get tokens and token IDs
        tokens, token_ids = tokenizer.encode(sentence)
        print(f"🔤 Tokens: {tokens}")
        print(f"🔢 Token IDs: {token_ids}")
        
        # Show detailed mapping
        mapping = tokenizer.get_token_mapping(sentence)
        print(f"📊 Token Mapping:")
        for item in mapping:
            status_icon = "✅" if item['status'] == 'KNOWN' else "❌"
            print(f"   {status_icon} '{item['token']:15}' -> {item['token_id']:5} [{item['status']}]")
        
        # Decode back
        decoded = tokenizer.decode(token_ids)
        print(f"🔄 Decoded: '{decoded}'")
        
        # Show WordPiece statistics
        subword_tokens = [token for token in tokens if len(token) < 4 and token.isalnum()]
        if subword_tokens:
            print(f"🎯 WordPiece splits: {subword_tokens}")
        
        unknown_tokens = [token for token in tokens if token not in tokenizer.vocab]
        if unknown_tokens:
            print(f"⚠️  Unknown tokens: {unknown_tokens}")

    # Additional statistics
    print(f"\n{'='*80}")
    print("📈 TOKENIZER STATISTICS")
    print(f"{'='*80}")
    print(f"Vocabulary size: {len(tokenizer.vocab):,}")
    print(f"UNK token ID: {tokenizer.unk_token_id}")
    
    # Test encode_ids_only for speed
    print(f"\n⚡ Testing encode_ids_only (fast method):")
    test_text = "The quick brown fox"
    token_ids_fast = tokenizer.encode_ids_only(test_text)
    print(f"Input: '{test_text}'")
    print(f"Token IDs (fast): {token_ids_fast}")

if __name__ == "__main__":
    main()