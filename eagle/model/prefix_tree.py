class PrefixTreeNode:
    def __init__(self):
        self.children = {}
        self.starts = []

class PrefixTree:
    def __init__(self, max_match_length: int = 10):
        self.root = PrefixTreeNode()
        self.max_match_length = max_match_length
        self.context = []

    def insert_incremental(self, new_context):
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
                node.starts.append(i+1)

    def top_k_unique_matches(self, tokens, nr_continuations: int = 3, minimal_match_length: int = 1):
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
        # take top-k longest matches
        sorted_matches = sorted(best_matches.items(), key=lambda x: x[1], reverse=True)
        filtered_matches = [(pos, length) for pos, length in sorted_matches if length >= minimal_match_length]
        top_k_matches = filtered_matches[:nr_continuations]
        return top_k_matches


