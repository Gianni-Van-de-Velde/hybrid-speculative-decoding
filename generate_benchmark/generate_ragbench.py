from datasets import load_dataset
import json
import os
from transformers import AutoTokenizer

if __name__ == "__main__":
    # TODO should be the commented version
    #tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("../Llama-3.1-8B-Instruct/")

    for dataset in ['covidqa', 'cuad', 'delucionqa', 'emanual', 'expertqa', 'finqa', 'hagrid', 'hotpotqa', 'msmarco', 'pubmedqa', 'tatqa', 'techqa']:
        ds = load_dataset("galileo-ai/ragbench", dataset)
        os.makedirs(f"eagle/data/{dataset}", exist_ok=True)
        
        random_80_questions = ds['test'].shuffle(seed=42).select(range(80))
        with open(f"eagle/data/{dataset}/question.jsonl", "w") as f:
            for i, row in enumerate(random_80_questions):
                documents_str = "\n".join(row["documents"])
                documents_tokens = tokenizer(documents_str)['input_ids']
                # Usually, we run with 4096 cache size and use max 1024 generated tokens. Subtract 300 margin for speculated tokens, question and just to be sure -> 2800 tokens
                max_document_tokens = 2800
                if len(documents_tokens) > max_document_tokens:
                    documents_str = tokenizer.decode(documents_tokens[:max_document_tokens], skip_special_tokens=True)
                prompt = f"""Use the following pieces of context to answer the question.
        {documents_str}
        Question: {row['question']}"""
                out = {
                    "question_id": i,
                    "category": row["question"],
                    "turns": [prompt]
                }
                f.write(json.dumps(out) + "\n")