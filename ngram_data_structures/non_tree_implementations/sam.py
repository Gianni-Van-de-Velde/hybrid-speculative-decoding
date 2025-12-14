import os
import json
import torch
from dataclasses import dataclass, field
from typing import Optional, Union, List, Literal, Dict, Any
from enum import Enum


@dataclass
class SamdConfig:
    max_predicts: int = field(default=60)
    alpha: float = field(default=4.0)
    K: int = field(default=8)
    len_bias: int = field(default=5)
    cache_type: Literal["dynamic", "static"] = field(
        default="static"
    )

class ForwardType(str, Enum):
    prefill = "prefill"
    seq_decode = "seq_decode"
    tree_decode = "tree_decode"


class ForwardState:

    def __init__(self, forward_type: ForwardType | None) -> None:
        self.forward_type = forward_type


class MaskState:

    def __init__(self, mask: Optional[torch.Tensor]) -> None:
        self.mask = mask

    def set_state(self, mask: Optional[torch.Tensor]) -> None:
        self.mask = mask


def load_token_recycle(tree_path: Optional[str] = None):
    if tree_path is None:
        tree_path = "token_recycle.json"
    samd_path = os.path.dirname(__file__)
    with open(os.path.join(samd_path, "config", tree_path), "r") as f:
        tree_adj: dict = json.load(f)["tree_adj"]
    num_node = len(tree_adj)
    tree: List[List[int]] = []
    for i in range(num_node):
        tree.append(tree_adj[str(i)])
    print("tree_path:", tree_path)
    print("len_tree:", len(tree))
    return tree


def load_eagle(tree_model_path: str, tree_path: Optional[str] = None):
    if tree_path is None:
        tree_path = "eagle.json"
    samd_path = os.path.dirname(__file__)
    with open(os.path.join(samd_path, "config", tree_path), "r") as f:
        tree = json.load(f)["tree_choices"]
    with open(os.path.join(tree_model_path, "config.json")) as f:
        tree_config = json.load(f)
    return tree, tree_config


def load_eagle2(tree_model_path: str):
    with open(os.path.join(tree_model_path, "config.json")) as f:
        tree_config = json.load(f)
    return tree_config

import torch
from typing import List, Tuple, Dict
from dataclasses import dataclass
from copy import deepcopy
from collections import deque
from tqdm import tqdm

def pad_path(path, length, pad_value=-1):
    return path + [pad_value] * (length - len(path))

class DynSAM:
   
    @dataclass
    class SAMState:
        next: dict[int, int]
        link: int
        length: int
        min_endpos: int

    def __init__(self, 
        max_predicts: int = 40, 
        alpha: float = 4.0, 
        device: str = "cuda"
    ):
        self.max_predicts = max_predicts
        self.alpha = alpha
        self.states: List[DynSAM.SAMState] = [DynSAM.SAMState(next={}, link=-1, length=0, min_endpos=0)]
        self.input_ids: List[int] = [-1]
        self.last = 0
        self.max_length = 0
        self.device = device
        
        # params needed to be reset for each query
        self.cur_index = 0
        self.cur_length = 0
    
    def reset(self):
        self.states: List[DynSAM.SAMState] = [DynSAM.SAMState(next={}, link=-1, length=0, min_endpos=0)]
        self.input_ids: List[int] = [-1]
        self.last = 0
        self.max_length = 0
        self.cur_index = 0
        self.cur_length = 0
    
    def expand_state(self, state: SAMState):
        new_index = len(self.states)
        self.states.append(state)
        return new_index

    def add_state(self, token: int):
        self.max_length += 1
        cur = self.expand_state(
            DynSAM.SAMState(
                next={}, link=-1, 
                length=self.max_length, 
                min_endpos=self.max_length
            )
        )
        p = self.last
        while p != -1 and token not in self.states[p].next:
            self.states[p].next[token] = cur
            p = self.states[p].link
        if p == -1:
            self.states[cur].link = 0
        else:
            q = self.states[p].next[token]
            if self.states[p].length + 1 == self.states[q].length:
                self.states[cur].link = q
            else:
                clone = self.expand_state(deepcopy(self.states[q]))
                self.states[clone].length = self.states[p].length + 1
                while p != -1 and self.states[p].next[token] == q:
                    self.states[p].next[token] = clone
                    p = self.states[p].link
                self.states[q].link = self.states[cur].link = clone
        self.last = cur
           
    def transfer_state(self, index: int, length: int, token: int):
        while index != 0 and token not in self.states[index].next:
            index = self.states[index].link
            length = self.states[index].length
        if token in self.states[index].next:
            index = self.states[index].next[token]
            length += 1
        else:
            index = length = 0
        return index, length
    
    def transfer_cur_state(self, token: int):
        self.cur_index, self.cur_length = \
            self.transfer_state(self.cur_index, self.cur_length, token)
    
    def to_anc(self, index: int, length: int):
        length_to_end = self.max_length - self.states[index].min_endpos
        while index != 0 and self.max_predicts > length_to_end:
            index = self.states[index].link
            length = self.states[index].length
            length_to_end = self.max_length - self.states[index].min_endpos
        return index, length
    
    def add_tokens(self, tokens: List[int]):
        for token in tokens:
            self.transfer_cur_state(token)
            self.add_state(token)
        self.input_ids.extend(tokens)
    
    def transfer_tokens(self, tokens: List[int]):
        for token in tokens:
            self.transfer_cur_state(token)

    def lookup(self, token: int):
        index, length = \
            self.transfer_state(self.cur_index, self.cur_length, token)
        return index, length

    def gen_draft(self, index: int, match_length: int, start_token: int):
        n = min(self.max_predicts, 1 + int(match_length * self.alpha))
        endpos = self.states[index].min_endpos
        seq = [start_token] + self.input_ids[endpos + 1:endpos + n]
        seq_position_ids = torch.arange(0, len(seq), dtype=torch.long, device=self.device).unsqueeze(0)
        return seq, {"seq_position_ids": seq_position_ids}

    def gen_buffers(self, anc_tree: List[int]):
        n = len(anc_tree)
        is_leaf = [True] * n
        tree_position_ids = [0] * n
        for i in range(1, n):
            is_leaf[anc_tree[i]] = False
            tree_position_ids[i] = tree_position_ids[anc_tree[i]] + 1
        tree_position_ids = torch.tensor([tree_position_ids], dtype=torch.long, device=self.device)
        
        tree_attn_mask = torch.zeros((n, n), dtype=torch.bool)
        for i in range(n):
            j = i
            while j != -1:
                tree_attn_mask[i, j] = True
                j = anc_tree[j]
        tree_attn_mask = tree_attn_mask.view(1, 1, n, n).to(self.device)
        
        retrieve_indices_nest = []
        for i in range(n):
            if not is_leaf[i]:
                continue
            retrieve_indices = [i]
            while retrieve_indices[-1] != 0:
                retrieve_indices.append(anc_tree[retrieve_indices[-1]])
            retrieve_indices_nest.append(list(reversed(retrieve_indices)))
        max_depth = max(len(x) for x in retrieve_indices_nest)
        retrieve_indices_nest = [pad_path(x, max_depth) for x in retrieve_indices_nest]
        tree_retrieve_indices = torch.tensor(retrieve_indices_nest, dtype=torch.long, device=self.device)
        return {
            "tree_attn_mask": tree_attn_mask,
            "tree_position_ids": tree_position_ids,
            "tree_retrieve_indices": tree_retrieve_indices,
        }
    
    def gen_tree_draft(self, index: int, match_length: int, start_token: int):
        n = min(self.max_predicts, 1 + int(match_length * self.alpha))
        h: List[Tuple[int, int, int]] = []
        tree = []
        anc_tree = []
        h.append((index, -1, start_token))
        while len(tree) != n and len(h) != len(tree):
            cur_tree_index = len(tree)
            cur_index, anc_tree_index, cur_token = h[cur_tree_index]
            tree.append(cur_token)
            anc_tree.append(anc_tree_index)
            if len(tree) == n:
                break
            for n_token, n_index in self.states[cur_index].next.items():
                h.append((n_index, cur_tree_index, n_token))
        return tree, self.gen_buffers(anc_tree)

import torch
from typing import List, Tuple, Dict
from dataclasses import dataclass
from copy import deepcopy
from collections import deque
from tqdm import tqdm
from dataclasses import dataclass, field
import heapq

def pad_path(path, length, pad_value=-1):
    return path + [pad_value] * (length - len(path))

@dataclass(order=True)
class SearchItem:
    prob: float
    token: int = field(compare=False)
    index: int = field(compare=False)
    anc_tree_index: int = field(compare=False)
    depth: int = field(compare=False)


class StaticSAM:
   
    @dataclass
    class SAMState:
        next: dict[int, int]
        link: int
        length: int
        cnt_endpos: int

    @staticmethod
    def build(
        batch_tokens: List[List[int]], 
        eos_token: int,
        verbose: bool =True
    ):
        sam = StaticSAM()
        sam.add_batch_tokens(batch_tokens, eos_token, verbose)
        sam.init_topk_next()
        return sam

    def __init__(self, 
        max_predicts: int = 40, 
        alpha: float = 4.0, 
        K: int = 8,
        device: str = "cuda"
    ):
        self.states: List[StaticSAM.SAMState] = [StaticSAM.SAMState(next={}, link=-1, length=0, cnt_endpos=0)]
        self.states_topk_next = None
        self.last = 0
        self.max_length = 0

        self.max_predicts = max_predicts
        self.alpha = alpha
        self.device = device
        self.K = K

        # params needed to be reset for each query
        self.cur_index = 0
        self.cur_length = 0
        
    def expand_state(self, state: SAMState):
        new_index = len(self.states)
        self.states.append(state)
        return new_index
    
    def add_state(self, token: int):
        self.max_length += 1
        cur = self.expand_state(
            StaticSAM.SAMState(
                next={}, link=-1, 
                length=self.max_length, 
                cnt_endpos=0,
            )
        )
        p = self.last
        while p != -1 and token not in self.states[p].next:
            self.states[p].next[token] = cur
            p = self.states[p].link
        if p == -1:
            self.states[cur].link = 0
        else:
            q = self.states[p].next[token]
            if self.states[p].length + 1 == self.states[q].length:
                self.states[cur].link = q
            else:
                clone = self.expand_state(deepcopy(self.states[q]))
                self.states[clone].length = self.states[p].length + 1
                while p != -1 and self.states[p].next[token] == q:
                    self.states[p].next[token] = clone
                    p = self.states[p].link
                self.states[q].link = self.states[cur].link = clone
        self.last = cur
        while cur != 0:
            self.states[cur].cnt_endpos += 1
            cur = self.states[cur].link

    def transfer_state(self, index: int, length: int, token: int):
        while index != 0 and token not in self.states[index].next:
            index = self.states[index].link
            length = self.states[index].length
        if token in self.states[index].next:
            index = self.states[index].next[token]
            length += 1
        else:
            index = length = 0
        return index, length
    
    def transfer_cur_state(self, token: int):
        self.cur_index, self.cur_length = \
            self.transfer_state(self.cur_index, self.cur_length, token)
    
    def add_tokens(self, tokens: List[int]):
        for token in tokens:
            self.transfer_cur_state(token)
            self.add_state(token)
    
    def transfer_tokens(self, tokens: List[int]):
        for token in tokens:
            self.transfer_cur_state(token)

    def lookup(self, token: int):
        index, length = \
            self.transfer_state(self.cur_index, self.cur_length, token)
        return index, length

    def reset(self):
        self.cur_index = 0
        self.cur_length = 0

    def add_batch_tokens(self, batch_tokens: List[List[int]], eos_token: int, verbose: bool):
        for tokens in tqdm(batch_tokens, desc="build sam...", disable=not verbose):
            self.add_tokens(tokens)
            if tokens[-1] != eos_token:
                self.add_tokens([eos_token])

    def init_topk_next(self, k: int = 8):
        self.states_topk_next = [None] * len(self.states)
        for index in tqdm(range(len(self.states)), "init top-k next"):            
            all_next = list(self.states[index].next.items())
            topk_next = sorted(
                all_next,
                key=lambda item: self.states[item[1]].cnt_endpos,
                reverse=True
            )[:k]
            self.states_topk_next[index] = topk_next

    def gen_buffers(self, anc_tree: List[int]):
        n = len(anc_tree)
        is_leaf = [True] * n
        tree_position_ids = [0] * n
        for i in range(1, n):
            is_leaf[anc_tree[i]] = False
            tree_position_ids[i] = tree_position_ids[anc_tree[i]] + 1
        tree_position_ids = torch.tensor([tree_position_ids], dtype=torch.long, device=self.device)
        
        tree_attn_mask = torch.zeros((n, n), dtype=torch.bool)
        for i in range(n):
            j = i
            while j != -1:
                tree_attn_mask[i, j] = True
                j = anc_tree[j]
        tree_attn_mask = tree_attn_mask.view(1, 1, n, n).to(self.device)
        
        retrieve_indices_nest = []
        for i in range(n):
            if not is_leaf[i]:
                continue
            retrieve_indices = [i]
            while retrieve_indices[-1] != 0:
                retrieve_indices.append(anc_tree[retrieve_indices[-1]])
            retrieve_indices_nest.append(list(reversed(retrieve_indices)))
        max_depth = max(len(x) for x in retrieve_indices_nest)
        retrieve_indices_nest = [pad_path(x, max_depth) for x in retrieve_indices_nest]
        tree_retrieve_indices = torch.tensor(retrieve_indices_nest, dtype=torch.long, device=self.device)
        return {
            "tree_attn_mask": tree_attn_mask,
            "tree_position_ids": tree_position_ids,
            "tree_retrieve_indices": tree_retrieve_indices,
        }
    
    def gen_draft(self, index: int, match_length: int, start_token: int):
        n = min(self.max_predicts, 1 + int(match_length * self.alpha))
        h = []
        tree = []
        anc_tree = []
        dep_cnt = {}
        heapq.heappush(h, SearchItem(prob=-1.0, token=start_token, index=index, anc_tree_index=-1, depth=0))
        while len(tree) != n and len(h) != 0:
            item: SearchItem = heapq.heappop(h)
            if item.depth not in dep_cnt:
                dep_cnt[item.depth] = 0
            if dep_cnt[item.depth] + 1 > self.K:
                continue
            dep_cnt[item.depth] += 1
            cur_tree_index = len(tree)
            tree.append(item.token)
            anc_tree.append(item.anc_tree_index)
            if len(tree) == n:
                break
            cnt_sum = self.states[item.index].cnt_endpos
            next_states = self.states_topk_next[item.index][:self.K]
            for n_token, n_index in next_states:
                n_prob = self.states[n_index].cnt_endpos / cnt_sum
                heapq.heappush(
                    h, 
                    SearchItem(
                        prob=item.prob * n_prob,
                        token=n_token,
                        index=n_index,
                        anc_tree_index=cur_tree_index,
                        depth=item.depth + 1
                    )
                )
        return tree, self.gen_buffers(anc_tree)

    # def gen_draft(self, index: int, match_length: int, start_token: int):
    #     n = min(self.max_predicts, 1 + int(match_length * self.alpha))
    #     seq_index = [index]
    #     seq = [start_token]
    #     while len(seq) != n:
    #         index = seq_index[-1]
    #         if len(self.states_topk_next[index]) == 0:
    #             break
    #         n_token, n_index = self.states_topk_next[index][0]
    #         seq_index.append(n_index)
    #         seq.append(n_token)
    #     seq_position_ids = torch.arange(0, len(seq), dtype=torch.long, device=self.device).unsqueeze(0)
    #     return seq, {"seq_position_ids": seq_position_ids}

import torch
from typing import List, Tuple, Dict, Optional
from enum import Enum
from collections import namedtuple


from transformers import LlamaConfig, LlamaForCausalLM

# from transformers import LlamaTokenizer
# tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained('/data/models/vicuna-7b-v1.3')

class CandidateType(str, Enum):
    sequence = "sequence"
    tree = "tree"

Candidates = namedtuple('Candidates', ['type', 'tokens', 'candidate_tokens', 'buffers_kwargs'])

TOPK = 8

class DraftModel(torch.nn.Module):
    
    def __init__(self,
        config: SamdConfig,
        sam_dyn: DynSAM = None,
        sam_static: StaticSAM = None,
        lm: LlamaForCausalLM = None,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.config = config
        self.sam_dyn = sam_dyn if sam_dyn is not None else DynSAM(config.max_predicts, config.alpha, device)
        self.sam_static = sam_static if sam_static is not None else StaticSAM(config.max_predicts, config.alpha, device)
        self.sam_dyn.max_predicts = config.max_predicts
        self.sam_dyn.alpha = config.alpha
        self.sam_static.max_predicts = config.max_predicts
        self.sam_static.alpha = config.alpha
        self.sam_static.K = config.K
        self.sam_static.device = device
        self.len_bias = config.len_bias

    def reset(self):
        self.sam_dyn.reset()
        self.sam_static.reset()

    def lookup(self, start_token: int):
        index_dyn, match_dyn = self.sam_dyn.lookup(start_token)
        index_static, match_static = self.sam_static.lookup(start_token)
        match_static -= self.len_bias
        if match_dyn >= match_static:
            seq, buffers_kwargs = self.sam_dyn.gen_draft(index_dyn, match_dyn, start_token)
            return (CandidateType.sequence, seq, buffers_kwargs)
        else:
            tree, buffers_kwargs = self.sam_static.gen_draft(index_static, match_static, start_token)
            return (CandidateType.tree, tree, buffers_kwargs)
    
    def update(self,
        tokens: Optional[torch.Tensor] = None,
    ):
        tokens_list = tokens.tolist()
        self.sam_dyn.add_tokens(tokens_list)
        self.sam_static.transfer_tokens(tokens_list)

    def prefill_update(self, 
        tokens: Optional[torch.Tensor] = None,
    ):
        self.update(tokens)
