import argparse
import time
from ngram_data_structures.prefix_tree_cpp import PrefixTree
from ngram_data_structures.prefix_tree_speedy_cpp import PrefixTreeSpeedy
from ngram_data_structures.suffix_tree_idx_cpp import SuffixTree as SuffixTreeIdx
from ngram_data_structures.suffix_tree_partial_result_cpp import SuffixTree as SuffixTreePartial
from ngram_data_structures.suffix_tree_full_cpp import SuffixTree as SuffixTreeFull
from ngram_data_structures.prefix_tree import PrefixTree as PrefixTreePy
from ngram_data_structures.non_tree_implementations.hashmap import HashMapNgram
from ngram_data_structures.non_tree_implementations.hashmap_cpp import HashMapNgram as HashmapCpp

from ngram_data_structures.non_tree_implementations.sam import DraftModel, SamdConfig
from ngram_data_structures.non_tree_implementations.pld import find_candidate_pred_tokens
from transformers import AutoTokenizer
import torch
import time
import os
import json

# This is a massive badly written script to test all implementations of ngram matching

# Load the ngram test data from cuad
prompts = []
with open("eagle/data/cuad/question.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        prompts.append(item["turns"][0])
            
answers = []
with open("cuad/CASD-ext-EAGLE3-hypertuned_Llama3.1-temperature-0.0-continuation_length-15-nr_continuations-2-minimal_match_length-3.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        for choice in item["choices"]:
            answers.append(choice["turns"][0])

base_model_path = "../Llama-3.1-8B-Instruct/"
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# Base function to test ngram implementations
def test_ngram_speed(ngram_init, ngram_insert_incremental, ngram_query, as_tensor = False, device = "cpu", output_file: str = None):
    total_insert_times = []
    total_query_times = []
    
    for _ in range(10):
        for prompt, answer in zip(prompts, answers):
            
            tokens_prompt = tokenizer(prompt, add_special_tokens=False)
            tokens_answer = tokenizer(answer, add_special_tokens=False)
        
            prompt = tokens_prompt['input_ids']
            query = tokens_answer['input_ids']

            input_ids = torch.tensor(prompt + query, dtype=torch.long, device=device).unsqueeze(0)
            prompt_len = len(prompt)

            tim = time.time()
            ngram = ngram_init()
            if as_tensor:
                ngram_insert_incremental(ngram, input_ids[:prompt_len])
            else:
                ngram_insert_incremental(ngram, prompt)
            time_elapsed = time.time() - tim
            total_insert_times.append(time_elapsed)
            
            tim = time.time()
            if as_tensor:
                for i in range(len(query)):
                    ngram_query(ngram, input_ids[:, :prompt_len+i+1])
                    ngram_insert_incremental(ngram, input_ids[:, prompt_len+i:prompt_len+i+1])
            else:
                for i in range(len(query)):
                    ngram_query(ngram, query[:i+1])
                    ngram_insert_incremental(ngram, query[i:i+1])
            time_elapsed = time.time() - tim
            total_query_times.append(time_elapsed)

    if output_file is not None:
        with open(output_file, "w") as f:
            json.dump(total_insert_times, f)

        with open(output_file.replace(".json", "_query.json"), "w") as f:
            json.dump(total_query_times, f)



# Implementation for tree based approach
def test_prefix_tree_speed(Tree, output_file: str = None, max_match_length=10, max_continuation_length=10):
    def ngram_init():
        return Tree(max_match_length=max_match_length, max_continuation_length=max_continuation_length)

    def ngram_insert_incremental(tree, tokens):
        tree.insert_incremental(tokens)
    
    def ngram_query(tree, tokens):
        return tree.top_k_unique_matches(tokens)

    test_ngram_speed(ngram_init, ngram_insert_incremental, ngram_query, output_file=output_file)

# Implementation for SAM approach
samd_config = SamdConfig(
    max_predicts=40,
    alpha=4,
    len_bias=5,
)

def test_sam_speed_cpu(output_file: str = None):
    def ngram_init():
        return DraftModel(samd_config, device="cpu")

    def ngram_insert_incremental(model, tokens):
        model.update(tokens.squeeze(0))
    
    def ngram_query(model, tokens):
        return model.lookup(tokens)

    test_ngram_speed(ngram_init, ngram_insert_incremental, ngram_query, as_tensor = True, output_file=output_file)

def test_sam_speed_gpu(output_file: str = None):
    def ngram_init():
        return DraftModel(samd_config, device="cuda")

    def ngram_insert_incremental(model, tokens):
        model.update(tokens.squeeze(0))
    
    def ngram_query(model, tokens):
        return model.lookup(tokens)

    test_ngram_speed(ngram_init, ngram_insert_incremental, ngram_query, as_tensor = True, device = "cuda", output_file=output_file)

def test_hash_map_speed(Hashmap, output_file: str = None):
    def ngram_init():
        return Hashmap()

    def ngram_insert_incremental(model, tokens):
        model.preprocess_prompt(tokens)
    
    def ngram_query(model, tokens):
        return model.find_text_position(tokens)

    test_ngram_speed(ngram_init, ngram_insert_incremental, ngram_query, output_file=output_file)

if __name__ == "__main__":
    # Get args
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_match_length", type=int, default=10)
    parser.add_argument("--max_continuation_length", type=int, default=10)
    parser.add_argument("--file_name_suffix", type=str, default="")
    parser.add_argument("--tree_only", type=bool, default=False)
    args = parser.parse_args()

    os.makedirs("ngram_stats", exist_ok=True)

    if not args.tree_only:
        # Test SAM
        test_sam_speed_cpu(output_file="ngram_stats/sam_cpu_times.json")
        test_sam_speed_gpu(output_file="ngram_stats/sam_gpu_times.json")

        # Test PLD
        test_ngram_speed(lambda : None, lambda _, __: None, lambda _, tokens: find_candidate_pred_tokens(tokens), as_tensor = True, device = "cpu", output_file="ngram_stats/pld_cpu_times.json")
        test_ngram_speed(lambda : None, lambda _, __: None, lambda _, tokens: find_candidate_pred_tokens(tokens), as_tensor = True, device = "cuda", output_file="ngram_stats/pld_gpu_times.json")

    test_hash_map_speed(HashMapNgram, output_file=f"ngram_stats/hash_map_times{args.file_name_suffix}.json")
    test_hash_map_speed(HashmapCpp, output_file=f"ngram_stats/hash_map_cpp_times{args.file_name_suffix}.json")

    # Test Prefix Tree
    matches = test_prefix_tree_speed(PrefixTreePy, output_file=f"ngram_stats/prefix_tree_py_times{args.file_name_suffix}.json", max_match_length=args.max_match_length, max_continuation_length=args.max_continuation_length)

    # Warm up for fair comparison
    test_prefix_tree_speed(PrefixTree, max_match_length=args.max_match_length, max_continuation_length=args.max_continuation_length)

    matches_new = test_prefix_tree_speed(PrefixTree, output_file=f"ngram_stats/prefix_tree_times{args.file_name_suffix}.json", max_match_length=args.max_match_length, max_continuation_length=args.max_continuation_length)
    matches_new = test_prefix_tree_speed(PrefixTreeSpeedy, output_file=f"ngram_stats/prefix_tree_speedy_times{args.file_name_suffix}.json", max_match_length=args.max_match_length, max_continuation_length=args.max_continuation_length)
    matches_new = test_prefix_tree_speed(SuffixTreeIdx, output_file=f"ngram_stats/suffix_tree_idx_times{args.file_name_suffix}.json", max_match_length=args.max_match_length, max_continuation_length=args.max_continuation_length)
    matches_new = test_prefix_tree_speed(SuffixTreeFull, output_file=f"ngram_stats/suffix_tree_full_times{args.file_name_suffix}.json", max_match_length=args.max_match_length, max_continuation_length=args.max_continuation_length)
    matches_new = test_prefix_tree_speed(SuffixTreePartial, output_file=f"ngram_stats/suffix_tree_partial_times{args.file_name_suffix}.json", max_match_length=args.max_match_length, max_continuation_length=args.max_continuation_length)

