"""
test_tokenizer.py
Test the tokenizer - MUST HAVE THE SAME CLASS DEFINITION
"""

import pickle
import re

class WikiTextTokenizer:
    def __init__(self, vocab=None, dictionary=None):
        self.vocab = vocab or {}
        self.dictionary = dictionary or set()
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        if '[unk]' not in self.vocab:
            self.vocab['[unk]'] = max(self.vocab.values()) + 1 if self.vocab else 1
            self.inverse_vocab[self.vocab['[unk]']] = '[unk]'
        
        self.unk_token_id = self.vocab['[unk]']
    
    def split_into_tokens(self, text):
        text = text.lower().strip()
        if not text:
            return []
        
        tokens = re.findall(r"""
            \w+          # words (letters, numbers, underscores)
            | [^\w\s]    # punctuation and special characters
        """, text, re.VERBOSE)
        
        return tokens
    
    def tokenize(self, text):
        return self.split_into_tokens(text)
    
    def encode(self, text):
        tokens = self.split_into_tokens(text)
        token_ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
        return tokens, token_ids
    
    def encode_ids_only(self, text):
        tokens, token_ids = self.encode(text)
        return token_ids
    
    def decode(self, token_ids):
        tokens = [self.inverse_vocab.get(id, '[unk]') for id in token_ids]
        return " ".join(tokens)
    
    def get_token_mapping(self, text):
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
    print("Loading tokenizer...")
    
    # Load the tokenizer
    with open("wikitext_tokenizer_final.pkl", 'rb') as f:
        tokenizer = pickle.load(f)
    
    print("✅ Tokenizer loaded successfully!")
    
    # Test sentences
    test_sentences = [
        "Hello world! This is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "I can't believe it's working!",
        "Machine learning and artificial intelligence",
        "Email: user@example.com",
        "Temperature: 25°C, price: $19.99"
    ]
    
    for sentence in test_sentences:
        print(f"\n📝 Input: '{sentence}'")
        
        # Get tokens and token IDs
        tokens, token_ids = tokenizer.encode(sentence)
        print(f"🔤 Tokens: {tokens}")
        print(f"🔢 Token IDs: {token_ids}")
        
        # Show detailed mapping
        mapping = tokenizer.get_token_mapping(sentence)
        print(f"📊 Token Mapping:")
        for item in mapping:
            print(f"   '{item['token']}' -> {item['token_id']} [{item['status']}]")
        
        # Decode back
        decoded = tokenizer.decode(token_ids)
        print(f"🔄 Decoded: '{decoded}'")
        print("-" * 60)

if __name__ == "__main__":
    main()