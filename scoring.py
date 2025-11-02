import pickle
import os
# from candidate_generation import generate_candidates
import re
from testbktree import BKTree,CandidateGenerator,BKNode
import dill

# Load spell checker with dill to handle __main__ pickles
with open("spell_checker_object.pkl", 'rb') as f:
    spell_checker = dill.load(f)

def generate_candidates(spell_checker, word, max_candidates=25, max_edit_distance=2):
    """Use the prebuilt BK-tree spell_checker to produce candidates."""
    return spell_checker.generate_candidates(
        word,
        max_candidates=max_candidates,
        max_edit_distance=max_edit_distance,
    )
with open('unigram_small.pkl', 'rb') as f:
    unigram = pickle.load(f)
    
with open("trigram_small.pkl", 'rb') as f:
    trigram = pickle.load(f)

def score_candidate(candidate, sentence, unigram_freqs, trigram_freqs):
    """Compute simple unigram+trigram scores for a candidate in the given sentence context."""
    total_unigram_freq = 16  # adjust to your corpus totals if you have them
    total_trigram_freq = 11

    words = re.findall(r"\b[\w']+\b", sentence.lower())
    if len(words) >= 3:
        w1, w2 = words[-3], words[-2]
    elif len(words) == 2:
        w1, w2 = "" , words[-1]
    elif len(words) == 1:
        w1, w2 = "", ""
    else:
        w1, w2 = "", ""

    unigram_f = unigram_freqs.get(candidate, 0)
    trigram_f = trigram_freqs.get((w1, w2, candidate), 0)

    eps_u, eps_t = 1e-5, 1e-6
    unigram_score = (unigram_f + eps_u) / (total_unigram_freq + eps_u * len(unigram_freqs))
    trigram_score = (trigram_f + eps_t) / (total_trigram_freq + eps_t * len(trigram_freqs))
    return unigram_score, trigram_score

def final_score(candidate, distance, word, unigram_score, trigram_score):
    """Combine edit-distance, unigram, trigram into a single score."""
    W_ed, W_u, W_t = 0.55, 0.10, 0.35
    return (W_ed * (1 / (1 + distance))) + (W_u * unigram_score) + (W_t * trigram_score)

def get_final_candidates(spell_checker, word, sentence, unigram_freqs, trigram_freqs,
                         max_candidates=25, max_edit_distance=3):
    """Generate, score, and rank candidates for the given word and sentence."""
    candidates = generate_candidates(spell_checker, word, max_candidates, max_edit_distance)
    scored = []
    for candidate, distance in candidates:
        ug, tg = score_candidate(candidate, sentence, unigram, trigram)
        scored.append((candidate, final_score(candidate, distance, word, ug, tg)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

if __name__ == "__main__":
    test_sentence = "Einstein's theory of realtivity"
    test_word = "realtivity"
    print(f"Candidates for '{test_word}':")
    if test_word in spell_checker.dictionary:
        print(f"  '{test_word}' is found in the dictionary. No correction needed.")
    final_candidates = get_final_candidates(spell_checker, test_word, test_sentence, unigram_freqs=unigram, trigram_freqs=trigram)
    for candidate, final in final_candidates:
        print(f"  {candidate} final score: {final:.6f}")
