from typing import List, Tuple


class SuffixTreeNode:
    def __init__(self):
        self.children = {}

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
            start_position = max(i-self.max_match_length-self.max_continuation_length+1, 0)
            for t in self.context[start_position:i+1]:
                if t not in node.children:
                    node.children[t] = SuffixTreeNode()
                node = node.children[t]

    def top_k_unique_matches(self, tokens: List[int]) -> Tuple[List[int], List[int]]:
        for i in reversed(range(1, self.max_match_length + 1)):
            node = self.root
            for t in tokens[-i:]:
                if t in node.children:
                    node = node.children[t]
                else:
                    break
            else:
                spec_tokens, parents = self.speculate_tree(node, [], [], -1, 0)
                return spec_tokens, parents

        return [], []
    
    # This does some pointer magic to edit lists in place
    def speculate_tree(self, node: SuffixTreeNode, tokens: list, parents: list, parent: int, depth: int) -> Tuple[List[int], List[int]]:
        if depth >= self.max_continuation_length:
            return tokens, parents
        
        for child_token, child_node in node.children.items():
            tokens.append(child_token)
            parents.append(parent)
            tokens, parents = self.speculate_tree(child_node, tokens, parents, parent=len(tokens)-1, depth=depth+1)
        return tokens, parents