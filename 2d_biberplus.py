import json
import sys

from biberplus.tagger import calculate_tag_frequencies, load_config, load_pipeline


def get_biber_features(text, pipeline, config):
    if not text or not text.strip():
        return {}
    df = calculate_tag_frequencies(text, pipeline, config)
    # df has features as index/columns; convert to a plain dict of {feature: rate}
    # BiberPlus returns a DataFrame — flatten to dict
    if df is None or df.empty:
        return {}
    # Result is typically a single-row DataFrame with feature names as columns
    return df.iloc[0].to_dict()


def tag_entries(entries, pipeline, config):
    for entry in entries:
        text = entry.get("text", "")
        entry["biberplus"] = get_biber_features(text, pipeline, config)
    return entries


def process_record(record, pipeline, config):
    annotation = record.get("llm_register_annotation")
    if annotation is None:
        return record

    for key in ("merged", "merged_core"):
        entries = annotation.get(key)
        if entries:
            annotation[key] = tag_entries(entries, pipeline, config)

    return record


def main(path):
    print("Loading BiberPlus pipeline...", file=sys.stderr)
    config = load_config()
    config.update(
        {
            "use_gpu": True,
            "biber": True,
            "function_words": False,
        }
    )
    pipeline = load_pipeline(config)
    print("Pipeline loaded.", file=sys.stderr)

    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record = process_record(record, pipeline, config)
            print(json.dumps(record, ensure_ascii=False))
            if i % 100 == 0:
                print(f"Processed {i} records...", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <file.jsonl>")
        sys.exit(1)
    main(sys.argv[1])
