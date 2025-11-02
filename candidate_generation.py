import pickle
import os
from testbktree import BKTree,CandidateGenerator,BKNode


with open("spell_checker_object.pkl", 'rb') as f:
            spell_checker = pickle.load(f)
    

def generate_candidates(word, max_candidates=25, max_edit_distance=2):
    """Generate candidate corrections for a given word."""
    candidates = spell_checker.generate_candidates(
        word,
        max_candidates=max_candidates,
        max_edit_distance=max_edit_distance
    )
    return candidates

if __name__ == "__main__":
    # Example usage
    test_word = "talkeds"
    candidates = generate_candidates(test_word, max_candidates=10, max_edit_distance=3)
    print(f"Candidates for '{test_word}':")
    for candidate, distance in candidates:
        print(f"  {candidate} (edit distance: {distance})")