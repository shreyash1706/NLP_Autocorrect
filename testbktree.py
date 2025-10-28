"""
test_bktree_clean.py
Clean version without frequency and test cases
"""

import pickle

# ✅ COMPLETE CLASS DEFINITIONS
class BKNode:
    def __init__(self, word):
        self.word = word
        self.children = {}

class BKTree:
    def __init__(self):
        self.root = None
        self.dictionary = set()
    
    def build_tree(self, words):
        if not words:
            return
        self.root = BKNode(words[0])
        self.dictionary.add(words[0])
        for word in words[1:]:
            self._insert(self.root, word)
            self.dictionary.add(word)
    
    def _insert(self, node, word):
        distance = self._edit_distance(node.word, word)
        if distance == 0:
            return
        if distance in node.children:
            self._insert(node.children[distance], word)
        else:
            node.children[distance] = BKNode(word)
    
    def _edit_distance(self, word1, word2):
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + 1)
        return dp[m][n]
    
    def search(self, word, max_distance=2):
        if not self.root:
            return []
        candidates = []
        self._search_recursive(self.root, word, max_distance, candidates)
        return candidates
    
    def _search_recursive(self, node, word, max_distance, candidates):
        if not node:
            return
        distance = self._edit_distance(node.word, word)
        if distance <= max_distance:
            candidates.append((node.word, distance))
        for d in range(max(1, distance - max_distance), distance + max_distance + 1):
            if d in node.children:
                self._search_recursive(node.children[d], word, max_distance, candidates)

class CandidateGenerator:
    def __init__(self, dictionary_path=None):
        self.bk_tree = BKTree()
        self.dictionary = set()
    
    def generate_candidates(self, word, max_candidates=10, max_edit_distance=2):
        """
        Generate spelling correction candidates for a word
        """
        word = word.lower()
        
        # Direct BK-Tree search
        bk_candidates = self.bk_tree.search(word, max_edit_distance)
        
        # Common misspellings
        common_candidates = self._common_misspellings(word)
        
        # Combine all candidates
        all_candidates = self._combine_candidates(bk_candidates, common_candidates)
        
        # Rank and return top candidates
        ranked_candidates = self._rank_candidates(word, all_candidates, max_candidates)
        
        return ranked_candidates
    
    def _common_misspellings(self, word):
        """Check for common misspellings"""
        common_misspellings = {
            'teh': 'the', 'adn': 'and', 'waht': 'what', 'tahnk': 'thank',
            'recieve': 'receive', 'seperate': 'separate', 'definately': 'definitely',
            'becuase': 'because', 'accomodate': 'accommodate', 'arguement': 'argument',
            'embarass': 'embarrass', 'existance': 'existence', 'guage': 'gauge',
            'harrass': 'harass', 'judgement': 'judgment', 'liason': 'liaison',
            'maintenence': 'maintenance', 'neccessary': 'necessary', 'privelege': 'privilege',
            'rythm': 'rhythm', 'tommorow': 'tomorrow'
        }
        
        candidates = []
        if word in common_misspellings:
            correction = common_misspellings[word]
            if correction in self.dictionary:
                candidates.append((correction, 1))
        
        return candidates
    
    def _combine_candidates(self, *candidate_lists):
        """Combine candidates from different strategies"""
        all_candidates = {}
        
        for candidate_list in candidate_lists:
            for word, distance in candidate_list:
                if word not in all_candidates or distance < all_candidates[word]:
                    all_candidates[word] = distance
        
        return [(word, dist) for word, dist in all_candidates.items()]
    
    def _rank_candidates(self, original_word, candidates, max_candidates):
        """Rank candidates by multiple factors"""
        if not candidates:
            return []
        
        scored_candidates = []
        
        for candidate, edit_distance in candidates:
            score = 0
            
            # Factor 1: Edit distance (lower is better)
            score += (3 - edit_distance) * 10
            
            # Factor 2: Length similarity
            length_diff = abs(len(original_word) - len(candidate))
            score -= length_diff * 2
            
            # Factor 3: Same starting letter bonus
            if original_word and candidate and original_word[0] == candidate[0]:
                score += 3
            
            scored_candidates.append((candidate, score, edit_distance))
        
        # Sort by score and return top candidates
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return [(word, edit_dist) for word, score, edit_dist in scored_candidates[:max_candidates]]

def get_user_settings():
    """Get user preferences for search parameters"""
    print("\n⚙️  Configure Search Parameters:")
    print("-" * 40)
    
    try:
        max_edit_distance = int(input("Max edit distance (1-5, default 2): ") or "2")
        max_edit_distance = max(1, min(5, max_edit_distance))
        
        max_candidates = int(input("Max candidates to show (1-50, default 10): ") or "10")
        max_candidates = max(1, min(50, max_candidates))
        
        return max_edit_distance, max_candidates
    except ValueError:
        print("⚠️  Using default values (edit distance: 2, candidates: 10)")
        return 2, 10

def interactive_mode(spell_checker, max_edit_distance=2, max_candidates=10):
    """Interactive mode for user input"""
    print(f"\n🎮 Interactive Mode (Edit Distance: {max_edit_distance}, Max Candidates: {max_candidates})")
    print("Type 'settings' to change parameters, 'quit' to exit")
    print("-" * 60)
    
    while True:
        user_input = input("\n🔍 Enter word to check: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        elif user_input.lower() in ['settings', 'config']:
            max_edit_distance, max_candidates = get_user_settings()
            print(f"✅ Updated: Edit Distance={max_edit_distance}, Max Candidates={max_candidates}")
            continue
        elif not user_input:
            continue
        
        # Generate candidates
        candidates = spell_checker.generate_candidates(
            user_input, 
            max_candidates=max_candidates,
            max_edit_distance=max_edit_distance
        )
        
        if candidates:
            print(f"\n📝 Suggestions for '{user_input}':")
            print("-" * 40)
            for i, (candidate, distance) in enumerate(candidates, 1):
                print(f"  {i:2d}. {candidate:15} (edit distance: {distance})")
            
            # Show best recommendation
            best_candidate = candidates[0][0]
            best_distance = candidates[0][1]
            print(f"\n✨ Best correction: '{user_input}' → '{best_candidate}' (edit distance: {best_distance})")
        else:
            print(f"✓ '{user_input}' appears to be correct or no suggestions found")

def main():
    try:
        # Load the spell checker
        with open("spell_checker_object.pkl", 'rb') as f:
            spell_checker = pickle.load(f)
        
        print("✅ Spell checker loaded successfully!")
        print(f"📊 Dictionary size: {len(spell_checker.dictionary):,} words")
        
        # Get user settings
        max_edit_distance, max_candidates = get_user_settings()
        
        # Enter interactive mode directly
        interactive_mode(spell_checker, max_edit_distance, max_candidates)
        
        print("\n🎉 Thank you for using the BK-Tree Spell Checker!")
        
    except FileNotFoundError:
        print("❌ Error: 'spell_checker_object.pkl' not found!")
        print("Please run create_bktree_standalone_fixed.py first")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()