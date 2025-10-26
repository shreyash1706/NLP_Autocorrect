"""
use_bktree_correct_params.py - With correct parameter names
"""

import pickle
from bktree import CandidateGenerator

def main():
    try:
        # Load the saved data
        with open("wikitext_candidate_generator.pkl", 'rb') as f:
            saved_data = pickle.load(f)
        
        # Create new candidate generator
        candidate_gen = CandidateGenerator()
        
        # Restore the data
        if isinstance(saved_data, dict):
            dictionary_set = saved_data['dictionary']
            dictionary_list = list(dictionary_set)
            
            candidate_gen.dictionary = dictionary_set
            candidate_gen.word_frequencies = saved_data['word_frequencies']
            candidate_gen.bk_tree.build_tree(dictionary_list)
        else:
            candidate_gen = saved_data
        
        # ✅ CONFIGURABLE PARAMETERS - CHANGE THESE AS NEEDED
        MAX_EDIT_DISTANCE = 3    # Changed from default 2 to 3
        TOP_K_CANDIDATES = 8     # Changed from default 10 to 8
        
        print("=== WikiText Spell Checker ===")
        print(f"Settings: Max Edit Distance = {MAX_EDIT_DISTANCE}, Top-K = {TOP_K_CANDIDATES}")
        print("Type words to check spelling (or 'quit' to exit)")
        print("-" * 50)
        
        while True:
            word = input("\nEnter word: ").strip().lower()
            
            if word in ['quit', 'exit', 'q']:
                break
            
            if not word:
                continue
            
            # ✅ CORRECT PARAMETER NAMES
            candidates = candidate_gen.generate_candidates(
                word, 
                max_edit_distance=MAX_EDIT_DISTANCE,  # ✅ Correct: max_edit_distance (not max_distance)
                max_candidates=TOP_K_CANDIDATES       # ✅ Correct: max_candidates
            )
            
            if candidates:
                print(f"Suggestions for '{word}':")
                for i, (candidate, distance) in enumerate(candidates, 1):
                    freq = candidate_gen.word_frequencies.get(candidate, 0)
                    print(f"  {i}. {candidate} (edit distance: {distance}, freq: {freq})")
            else:
                print(f"✓ '{word}' appears to be correct!")
                
    except FileNotFoundError:
        print("Error: wikitext_candidate_generator.pkl not found!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()