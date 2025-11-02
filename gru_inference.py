import onnx
import onnxruntime as ort
import re
import pickle
import torch
import torch.nn as nn
import numpy as np

from test_tokenizer import FastWordPieceTokenizer  # must exist and match original name!

with open("wikitext2_tokenizer_wordpiece.pkl", "rb") as f:
    tokenizer = pickle.load(f)

session = ort.InferenceSession("gru_model.onnx")

def predict_next_word(sentence, session=session, tokenizer=tokenizer, top_k=5):
    # 1️⃣ Tokenize (get token ids)
    tokens, token_ids = tokenizer.encode(sentence)
    input_ids = torch.tensor([token_ids], dtype=torch.long)

    # 2️⃣ Prepare ONNX inputs
    
    # --- FIX 2: Match NUM_LAYERS and HIDDEN_DIM from export ---
    # Shape is (num_layers, batch_size, hidden_dim)
    # Using 3 and 512 from our export script
    hidden_in = np.zeros((3, 1, 512), dtype=np.float32)  
    
    # --- FIX 1: Use 'input_tokens' to match the exported name ---
    ort_inputs = {"input_tokens": input_ids.numpy(), "hidden_in": hidden_in}

    # 3️⃣ Run ONNX inference
    logits, hidden_out = session.run(None, ort_inputs)

    # 4️⃣ Get last-step logits → probabilities
    last_logits = torch.tensor(logits)[:, -1, :]
    probs = torch.softmax(last_logits, dim=-1)

    # 5️⃣ Get top-k predictions
    top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)
    top_probs = top_probs.squeeze().tolist()
    top_indices = top_indices.squeeze().tolist()

    # 6️⃣ Convert ids → tokens
    results = []
    for i, idx in enumerate(top_indices):
        word = tokenizer.inverse_vocab.get(idx, "[unk]")
        results.append((word, float(top_probs[i])))

    return results

if __name__ == "__main__":
    sentence = "Einstein's theory of "
    predictions = predict_next_word(sentence, top_k=25)
    for w, p in predictions:
        print(f"{w:15s}  {p:.4f}")