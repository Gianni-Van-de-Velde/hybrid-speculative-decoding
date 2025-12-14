from datasets import load_dataset
import os
import json
import pandas as pd

if __name__ == "__main__":
    ds_hotpot = load_dataset("mteb/HotpotQA-NL", 'corpus')
    ds_hotpot_qrels = load_dataset("mteb/HotpotQA-NL", 'qrels')
    ds_hotpot_queries = load_dataset("mteb/HotpotQA-NL", 'queries')

    # Load splits
    corpus = ds_hotpot['test']
    qrels = ds_hotpot_qrels['test']
    queries = ds_hotpot_queries['test']

    # Sample 80 queries
    queries_sample = queries.shuffle(seed=42).select(range(80))
    query_ids = set(queries_sample["_id"])

    # Filter qrels to those queries
    qrels_small = qrels.filter(lambda ex: ex["query-id"] in query_ids)

    # Filter corpus to only relevant passages
    corpus_ids = set(qrels_small["corpus-id"])
    corpus_small = corpus.filter(lambda ex: ex["_id"] in corpus_ids)

    # Convert reduced sets to pandas
    df_queries = pd.DataFrame(queries_sample)
    df_qrels = pd.DataFrame(qrels_small)
    df_corpus = pd.DataFrame(corpus_small)

    # Merge qrels with corpus
    qrels_with_text = df_qrels.merge(df_corpus, left_on="corpus-id", right_on="_id", how="left", suffixes=("_query", "_passage"))

    # Group passages by query-id
    grouped = qrels_with_text.groupby("query-id").agg({
        "text": list,
        "score": list
    }).reset_index()

    # Join with queries
    merged = grouped.rename(columns={"text": "passages"}).merge(df_queries.rename(columns={"text": "query"}), left_on="query-id", right_on="_id", how="left")

    # Load prompts from JSON file
    with open("generate_benchmark/metadata/google_translated_prompt_per_language.json", "r", encoding="utf-8") as f:
        google_translated_prompt_per_language = json.load(f)

    language = 'dutch'
    target = f"eagle/data/hotpot-{language}/question.jsonl"

    os.makedirs(f"eagle/data/hotpot-{language}", exist_ok=True)

    with open(target, "w") as f:
        for i, row in merged.iterrows():
            passages_str = "\n".join(row["passages"])
            prompt = google_translated_prompt_per_language[language].format(passages=passages_str, query=row['query'])
            out = {
                "question_id": i,
                "category": row['query'],
                "turns": [prompt],
            }
            f.write(json.dumps(out) + "\n")