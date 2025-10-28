import streamlit as st
import random
import json
import streamlit.components.v1 as components

st.set_page_config(page_title="Context Aware Auto-Corrector", layout="centered")

st.title("🪄 Context-Aware Auto-Corrector")

# Placeholder suggestions (simulate model output)
sample_suggestions = ["apple", "banana", "grape", "orange", "mango"]

# Send them to JS
suggestions_json = json.dumps(sample_suggestions)


#get the current text from html element

#track last word and sentence both , 

#using the text generate bk for last word if not in dictionary


# Custom HTML + JS
components.html(f"""
<!DOCTYPE html>
<html>
<head>
<style>
textarea {{
  width: 100%;
  height: 200px;
  font-size: 16px;
  padding: 10px;
  border-radius: 8px;
  border: 1px solid #ccc;
  resize: none;
}}

#numSuggestions {{
  margin-bottom: 10px;
  padding: 5px 10px;
  font-size: 15px;
  border-radius: 6px;
  border: 1px solid #ccc;
  display: block;
}}
.suggestions {{
  display: flex;
  gap: 8px;
  margin-top: 10px;
}}

.suggestion {{
  background-color: #f0f0f0;
  border-radius: 5px;
  padding: 5px 10px;
  cursor: pointer;
  transition: background 0.2s;
}}

.suggestion:hover {{
  background-color: #e0e0e0;
}}
</style>
</head>
<body>

<textarea id="textArea" placeholder="Type here..."></textarea>

<label for="numSuggestions">Number of Suggestions:</label>
  <select id="numSuggestions">
    <option value="2">2</option>
    <option value="3" selected>3</option>
    <option value="4">4</option>
    <option value="5">5</option>
  </select>


<div id="suggestions" class="suggestions"></div>

<script>
const suggestions = {suggestions_json};
const numSelect = document.getElementById("numSuggestions");
const textArea = document.getElementById("textArea");
const suggestionsDiv = document.getElementById("suggestions");

textArea.addEventListener("keydown", function(e) {{
  if (e.key === " ") {{
    // Delay to get updated value
    setTimeout(() => {{
      const text = textArea.value.trim();
      const words = text.split(/\\s+/);
      const lastWord = words[words.length - 1];

      // Simulate: if last word is short, show suggestions
      if (lastWord.length > 0 && lastWord.length < 5) {{
        showSuggestions();
      }} else {{
        suggestionsDiv.innerHTML = "";
      }}
    }}, 50);
  }}
}});

function showSuggestions() {{
  suggestionsDiv.innerHTML = "";
  const randomSubset = suggestions.sort(() => 0.5 - Math.random()).slice(0, numSelect.value);
  randomSubset.forEach(s => {{
    const btn = document.createElement("div");
    btn.className = "suggestion";
    btn.textContent = s;
    btn.onclick = () => replaceLastWord(s);
    suggestionsDiv.appendChild(btn);
  }});
}}

function replaceLastWord(newWord) {{
  const text = textArea.value.trimEnd();
  const words = text.split(/\\s+/);
  words[words.length - 1] = newWord;
  textArea.value = words.join(" ") + " ";
  suggestionsDiv.innerHTML = "";
}}
</script>
</body>
</html>
""", height=350)
