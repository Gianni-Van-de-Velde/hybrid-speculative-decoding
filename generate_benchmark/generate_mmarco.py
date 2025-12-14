import argparse
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict
import random
import json
import os
import random

def preprocess(language):
    target = f"eagle/data/mmarco-{language}/question.jsonl"
    if os.path.exists(target):
        print(f"File {target} already exists, skipping...")
        return

    ds = load_dataset("unicamp-dl/mmarco", language, trust_remote_code=True)

    # This dataset only has contrastive pairs, but to be conform with MS-MARCO, we need 10 passages, one of which is the positive. This is what we work towards below.
    random_80 = ds['train'].shuffle(seed=42).select(range(80))
    queries_set = set(random_80["query"])
    # gather positives + negatives in one pass
    grouped = defaultdict(lambda: {"positives": set(), "negatives": set()})

    for ex in tqdm(ds['train']):
        if ex["query"] in queries_set:
            grouped[ex["query"]]["positives"].add(ex["positive"])
            grouped[ex["query"]]["negatives"].add(ex["negative"])

    seed = 42
    random.seed(seed)

    merged = []
    for q, vals in grouped.items():
        passages = list(vals["negatives"])[:9]
        pos = list(vals["positives"])[0]
        insert_idx = random.randrange(len(passages)+1)
        passages.insert(insert_idx, pos)
        merged.append({"query": q, "passages": passages})

    # Load prompts from JSON file
    with open("generate_benchmark/metadata/google_translated_prompt_per_language.json", "r", encoding="utf-8") as f:
        google_translated_prompt_per_language = json.load(f)

    os.makedirs(f"eagle/data/mmarco-{language}", exist_ok=True)

    with open(target, "w") as f:
        for i, row in enumerate(merged):
            passages_str = "\n".join(row["passages"])
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
        'english', 
        'chinese', 
        'french', 
        'german', 
        'indonesian', 
        'italian', 
        'portuguese', 
        'russian', 
        'spanish', 
        'arabic', 
        'dutch', 
        'hindi', 
        'japanese', 
        'vietnamese' 
    }

    for lang in args.language:
        if lang.lower() in available_langs:
            preprocess(lang)