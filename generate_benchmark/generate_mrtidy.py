import argparse
from datasets import load_dataset
import random
import json
import os
import random
from transformers import AutoTokenizer


def preprocess(language):
    target = f"eagle/data/mrtydi-{language}/question.jsonl"
    if os.path.exists(target):
        print(f"File {target} already exists, skipping...")
        return

    ds = load_dataset("castorini/mr-tydi", language)

    random_80 = ds['train'].shuffle(seed=42).select(range(80))

    # Load prompts from JSON file
    with open("generate_benchmark/metadata/google_translated_prompt_per_language.json", "r", encoding="utf-8") as f:
        google_translated_prompt_per_language = json.load(f)

    os.makedirs(f"eagle/data/mrtydi-{language}", exist_ok=True)

    random.seed(42)
    # TODO should be the commented version
    #tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("../Llama-3.1-8B-Instruct/")
    with open(target, "w") as f:
        for i, row in enumerate(random_80):
            # Usually, we run with 4096 cache size and use max 1024 generated tokens. Subtract 300 margin for speculated tokens, question and just to be sure -> 2800 tokens
            selected_passages = row['positive_passages']
            cumulative_token_count = len(tokenizer(selected_passages[0]['text'])['input_ids'])

            if cumulative_token_count > 2800:
                print(f"Warning: Positive passage alone exceeds token limit for question {i} in {language}.")
            
            # Add negative passages one by one until token limit is reached
            for neg_passage in row['negative_passages']:
                
                cumulative_token_count += len(tokenizer(neg_passage['text'])['input_ids'])
                if cumulative_token_count > 2800:
                    break
                selected_passages.append(neg_passage)

            if len(selected_passages) < 2:
                print(f"Warning: Only {len(selected_passages)} passages selected for question {i} in {language}.")
            
            random.shuffle(selected_passages)
            passages_texts = [p['text'] for p in selected_passages]
            passages_str = "\n".join(passages_texts)
            prompt = google_translated_prompt_per_language[language].format(passages=passages_str, query=row['query'])
            out = {
                "question_id": i,
                "category": row['query'],
                "turns": [prompt],
            }
            f.write(json.dumps(out) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", nargs="+", type=str, help="List of languages")
    args = parser.parse_args()
    print(args.language)

    available_langs = { 
        'arabic', 
        'bengali', 
        #'combined', 
        'english', 
        'finnish', 
        'indonesian', 
        'japanese', 
        'korean', 
        'russian', 
        'swahili', 
        'telugu', 
        'thai',
    }

    for lang in args.language:
        if lang.lower() in available_langs:
            preprocess(lang)