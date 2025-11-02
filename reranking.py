import pickle
import os
import re
import torch
import onnxruntime as ort
import onnx
from test_tokenizer import FastWordPieceTokenizer
# from candidate_generation import generate_candidates
import re
from testbktree import BKTree,CandidateGenerator,BKNode
from scoring import  get_final_candidates
from gru_inference import predict_next_word

# Remove unnecessary pickle load of spell_checker (avoid __main__ pickle issues)
with open("spell_checker_object.pkl", 'rb') as f:
    spell_checker = pickle.load(f)

with open('unigram_small.pkl', 'rb') as f:        
    unigram = pickle.load(f)
    
with open("trigram_small.pkl", 'rb') as f:
    trigram = pickle.load(f)

def get_gru_prob(context, candidate, session, tokenizer, top_k=50):
    """Estimate likelihood of candidate being the next word."""
    try:
        preds = predict_next_word(context, top_k=top_k, session=session, tokenizer=tokenizer)
        for word, prob in preds:
            if candidate.lower() == word.lower() or candidate.lower().startswith(word.lower()):
                return prob
        return 1e-6
    except Exception:
        return 1e-6

def gru_rerank(final_candidates, sentence, word, session, tokenizer, alpha=0.7, top_k=50):
    """Blend your scores with GRU next-word probabilities."""
    context = sentence.replace(word, "").strip()
    reranked = []
    for candidate, score in final_candidates:
        gru_prob = get_gru_prob(context, candidate, session, tokenizer, top_k=top_k)
        new_score = alpha * score + (1 - alpha) * gru_prob
        reranked.append((candidate, new_score))
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked

if __name__ == "__main__":
    with open("wikitext2_tokenizer_wordpiece.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    session = ort.InferenceSession("gru_model.onnx")
    test_sentence = "Einstein's theory of realtivity"
    test_word = "realtivity"
    final_candidates = get_final_candidates(spell_checker, test_word, test_sentence, unigram_freqs=unigram, trigram_freqs=trigram)
    print("\nBefore GRU reranking:")
    for c, s in final_candidates[:5]:
        print(f"{c:15s}  score={s:.6f}")
    reranked = gru_rerank(final_candidates, test_sentence, test_word, session, tokenizer, alpha=0.7)
    print("\nAfter GRU reranking:")
    for c, s in reranked[:5]:
        print(f"{c:15s}  score={s:.6f}")