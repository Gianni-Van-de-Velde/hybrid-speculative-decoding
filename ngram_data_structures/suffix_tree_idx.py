from typing import List, Tuple


class SuffixTreeNode:
    def __init__(self):
        self.children = {}
        self.starts = set()

class SuffixTree:
    def __init__(self, max_match_length: int = 10, max_continuation_length: int = 10):
        self.root = SuffixTreeNode()
        self.max_match_length = max_match_length
        self.max_continuation_length = max_continuation_length
        self.context = []

    def insert_incremental(self, new_context: List[int]):
        # We expect the context to be added part by part.
        start_idx = len(self.context)
        self.context.extend(new_context)
            
        for i in range(start_idx, len(self.context)):
            node = self.root
            # insert suffix up to max_match_length
            start_position = max(i-self.max_match_length+1, 0)
            for j, t in enumerate(self.context[start_position:i+1]):
                if t not in node.children:
                    node.children[t] = SuffixTreeNode()
                node = node.children[t]
                # +1 makes it land on the first token after the match
                node.starts.add(start_position + j + 1)

    def top_k_unique_matches(self, tokens: List[int]) -> Tuple[List[int], List[int]]:
        best_matches = {}  # pos -> length
        for i in range(1, self.max_match_length + 1):
            node = self.root
            for t in tokens[-i:]:
                if t in node.children:
                    node = node.children[t]
                else:
                    break
            else:
                for pos in node.starts:
                    best_matches[pos] = i
            
        # Use self.context to convert positions to tokens
        spec_tokens, parents = [], []
        for pos in best_matches.keys():
            start_idx = pos
            for j in range(self.max_continuation_length):
                spec_tokens.append(self.context[start_idx + j])
                if j == 0:
                    parents.append(-1)
                else:
                    parents.append(len(spec_tokens) - 2)

        return spec_tokens, parents


