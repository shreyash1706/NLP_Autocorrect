"""
use_bktree_simple.py

Simple script to use the BK-Tree candidate generator
"""

import pickle

def load_candidate_generator():
    """Load the candidate generator"""
    try:
        with open("wikitext_candidate_generator.pkl", 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        print("Error: Candidate generator file not found!")
        print("Please run create_bktree_standalone.py first")
        return None

def spell_check_word(word, candidate_gen):
    """Spell check a single word"""
    from bktree import CandidateGenerator  # Import the class
    
    # Create a temporary generator for this function
    temp_gen = CandidateGenerator()
    temp_gen.dictionary = set(candidate_gen['dictionary'])
    temp_gen.word_frequencies = candidate_gen['word_frequencies']
    temp_gen.bk_tree.build_tree(candidate_gen['dictionary'])
    
    candidates = temp_gen.generate_candidates(word, max_candidates=5)
    return candidates

def main():
    # Load candidate generator
    candidate_gen = load_candidate_generator()
    if candidate_gen is None:
        return
    
    print("=== WikiText Spell Checker ===")
    print("Type words to check spelling (or 'quit' to exit)")
    print("-" * 50)
    
    while True:
        word = input("\nEnter word: ").strip().lower()
        
        if word in ['quit', 'exit', 'q']:
            break
        
        if not word:
            continue
        
        # Spell check the word
        candidates = spell_check_word(word, candidate_gen)
        
        if candidates:
            print(f"Suggestions for '{word}':")
            for i, (candidate, distance) in enumerate(candidates, 1):
                print(f"  {i}. {candidate} (edit distance: {distance})")
        else:
            print(f"✓ '{word}' appears to be correct!")

if __name__ == "__main__":
    main()