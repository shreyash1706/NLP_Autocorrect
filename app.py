import json
import random
from testbktree import BKTree, CandidateGenerator, BKNode
from test_tokenizer import FastWordPieceTokenizer
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
from reranking import gru_rerank
import onnxruntime as ort
import pickle
from scoring import get_final_candidates

@st.cache_resource
def load_resources():
    with open("spell_checker_object.pkl", "rb") as f:
        spell_checker = pickle.load(f)

    with open("unigram_small.pkl", "rb") as f:
        unigram = pickle.load(f)

    with open("trigram_small.pkl", "rb") as f:
        trigram = pickle.load(f)

    # with open("wikitext2_tokenizer_wordpiece.pkl", "rb") as f:
    #     tokenizer = pickle.load(f)
    
    return spell_checker, unigram, trigram



# Load everything once
spell_checker, unigram, trigram = load_resources()

st.set_page_config(page_title="Context Aware Auto-Corrector", layout="centered")
st.title("🪄 Context-Aware Auto-Corrector")

component_path = str(Path(__file__).parent / "autocorrect_component")
autocorrect = components.declare_component("autocorrect", path=component_path)

# Initialize persistent state
if "suggestions" not in st.session_state:
    st.session_state.suggestions = []

# Call the component ONCE, with current suggestions
result = autocorrect(suggestions=st.session_state.suggestions, key="autocorrect")

if result is not None:
    text = result.get("text", "") or ""
    last_word = result.get("lastWord", "") or ""

    # Only recompute if user actually typed a new last_word
    if last_word and (st.session_state.get("last_processed") != last_word):
        st.session_state.last_processed = last_word
        if last_word.lower() in spell_checker.dictionary:
            st.session_state.suggestions = []
        else:
            candidates = get_final_candidates(
                spell_checker,
                last_word,
                text,
                unigram,
                trigram,
                max_candidates=25,
                max_edit_distance=3,
            )

            # Optionally use GRU rerank
            # reranked = gru_rerank(candidates, text, last_word, session, tokenizer, alpha=0.7, top_k=50)
            # st.session_state.suggestions = [w for w, _ in reranked[:5]]

            st.session_state.suggestions = [w for w, _ in candidates[:10]]

        # Update component with new suggestions
        st.rerun()

    # For debugging
    # st.write("Suggestions:", st.session_state.suggestions)
    st.write("Text:", text)
    st.write("Last Word:", last_word)
