"""
scoring.py

Handles detection of misspelled last word and candidate generation using BK-tree.
No scoring or UI suggestions.
"""

import pickle
import os


class LastWordCandidateGenerator:
    """
    Generates candidate corrections for misspelled last words in sentences
    using a BK-tree.
    """

    def __init__(self, bk_tree=None, dictionary=None):
        """
        Initialize the generator with BK-tree and dictionary.
        """
        self.bk_tree = bk_tree
        self.dictionary = set(dictionary or [])

    def is_misspelled(self, word):
        """
        Check if a word is not in the dictionary.
        """
        return word.lower() not in self.dictionary

    def get_candidates(self, word, max_edit_distance=2, max_candidates=10):
        """
        Get candidate words from BK-tree within a max edit distance.
        """
        if not self.bk_tree or not getattr(self.bk_tree, 'root', None):
            return []
        seen = set()
        results = []
        for cand, dist in self.bk_tree.search(word.lower(), max_edit_distance):
            if cand not in seen:
                seen.add(cand)
                results.append((cand, dist))
                if len(results) >= max_candidates:
                    break
        return results

    def get_candidates_for_last_word(self, sentence, max_edit_distance=2, top_k=10):
        """
        If the last word in the sentence is misspelled,
        return its BK-tree candidates.
        Otherwise, return an empty list.
        """
        if not sentence or not sentence.strip():
            return []

        words = sentence.strip().split()
        if not words:
            return []

        last_word = words[-1]
        if not last_word.isalpha():
            return []

        if not self.is_misspelled(last_word):
            return []

        candidates = self.get_candidates(last_word, max_edit_distance, top_k)
        return candidates


def load_bk_tree(bk_tree_path='spell_checker_object.pkl'):
    """
    Load BK-tree and dictionary from saved object.
    """
    bk_tree = None
    dictionary = set()
    if os.path.exists(bk_tree_path):
        try:
            with open(bk_tree_path, 'rb') as f:
                obj = pickle.load(f)
                bk_tree = getattr(obj, 'bk_tree', None)
                dictionary = set(getattr(obj, 'dictionary', []) or [])
            print(f"✓ Loaded BK-tree from {bk_tree_path}")
        except Exception as e:
            print(f"⚠ Error loading BK-tree: {e}")
    else:
        print(f"⚠ {bk_tree_path} not found")
    return bk_tree, dictionary


def demo():
    """
    Demo for candidate generation (no scoring).
    """
    print("=" * 60)
    print("Candidate Generation Demo")
    print("=" * 60)

    bk_tree, dictionary = load_bk_tree()

    if not bk_tree:
        print("❌ BK-tree not loaded.")
        return

    generator = LastWordCandidateGenerator(bk_tree, dictionary)

    test_sentences = [
        "I love to reat books",        # 'reat' -> 'read'
        "The cat is sleeping on the mat",  # Correct
        "She went to the stor to buy food",  # 'stor' -> 'store'
        "We are going to the bech today",   # 'bech' -> 'beach'
    ]

    for sentence in test_sentences:
        print(f"\n📝 Sentence: {sentence}")
        candidates = generator.get_candidates_for_last_word(sentence)
        if candidates:
            print(f"❌ Last word '{sentence.split()[-1]}' is misspelled")
            print("📊 Candidates:")
            for i, (cand, dist) in enumerate(candidates, 1):
                print(f"  {i}. {cand} (edit distance: {dist})")
        else:
            print("✓ Last word is correct")

    print("\nDemo complete.")
    print("=" * 60)


if __name__ == "__main__":
    demo()


import streamlit as st

# Disable browser's spellcheck completely
st.markdown("""
<style>
textarea, input, [contenteditable="true"] {
  spellcheck: false !important;
  -webkit-text-security: none !important;
  -moz-text-decoration-style: none !important;
  text-decoration: none !important;
  caret-color: auto !important;
}
::spelling-error {
  text-decoration: none !important;
}
</style>
<script>
document.addEventListener("DOMContentLoaded", function() {
  // Force disable spellcheck on all textareas
  document.querySelectorAll("textarea").forEach(el => el.setAttribute("spellcheck", "false"));
});
</script>
""", unsafe_allow_html=True)
