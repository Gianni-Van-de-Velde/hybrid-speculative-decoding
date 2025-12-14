from typing import List, Tuple


class PrefixTreeNode:
    def __init__(self):
        self.children = {}
        self.starts = set()

class PrefixTree:
    def __init__(self, max_match_length: int = 10, max_continuation_length: int = 10):
        self.root = PrefixTreeNode()
        self.max_match_length = max_match_length
        self.max_continuation_length = max_continuation_length
        self.context = []
    
    def insert_incremental(self, new_context: List[int]):
        # We expect the context to be added part by part.
        start_idx = len(self.context)
        self.context.extend(new_context)
            
        for i in range(start_idx, len(self.context)):
            node = self.root
            # insert reversed suffix up to continuation_length
            start_position = max(i-self.max_match_length+1, 0)
            for t in reversed(self.context[start_position:i+1]):
                if t not in node.children:
                    node.children[t] = PrefixTreeNode()
                node = node.children[t]
                # +1 makes it land on the first token after the match
                node.starts.add(i+1)

    def top_k_unique_matches(self, tokens: List[int]) -> Tuple[List[int], List[int]]:
        node = self.root
        best_matches = {}  # pos -> length
        # search reversed to align rightmost token
        for i, t in enumerate(reversed(tokens[-self.max_match_length:])):
            if t in node.children:
                node = node.children[t]
                for pos in node.starts:
                    # keep only the longest match per position
                    if pos not in best_matches or i + 1 > best_matches[pos]:
                        best_matches[pos] = i + 1
            else:
                break
            
        # Use self.context to convert positions to tokens
        spec_tokens, parents = [], []
        for pos in best_matches.keys():
            start_idx = pos
            for j in range(self.max_continuation_length):
                if start_idx + j >= len(self.context):
                    break
                spec_tokens.append(self.context[start_idx + j])
                if j == 0:
                    parents.append(-1)
                else:
                    parents.append(len(spec_tokens) - 2)

        return spec_tokens, parents


