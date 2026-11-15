"""Create the frozen inputs used by the two recipe-recommender notebooks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from importlib.metadata import version

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from remy.model.embedding import EMBEDDING_DIR
from remy.splits import SPLIT_DIR
from remy.checkpoints import atomic_json, file_sha256


def generate_minilm(recipes, directory, device="cpu", batch_size=64, chunk_size=4096, encoder=None):
    """Resume at the last flushed chunk; completed arrays are atomically published."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    recipes = recipes.sort_values("id")
    ids = recipes.id.to_numpy()
    fields = dict(name=recipes.name.fillna(""), ingredients=recipes.ingredients.map(", ".join))
    fields["vectors"] = "Name: " + fields["name"] + "\nIngredients: " + fields["ingredients"]
    digest = hashlib.sha256(ids.tobytes())
    for field, texts in fields.items():
        digest.update(field.encode())
        digest.update("\0".join(texts).encode())
    identity = {"input_sha256": digest.hexdigest(), "model": "sentence-transformers/all-MiniLM-L6-v2",
                "max_seq_length": 256, "rows": len(ids), "dtype": "float16",
                "sentence_transformers": version("sentence-transformers"), "transformers": version("transformers")}
    progress_path = directory / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress["identity"] != identity:
            raise ValueError("Embedding inputs or library versions changed; use a new output directory")
    else:
        if any((directory / f"{field}.npy").exists() for field in fields):
            raise ValueError("Existing arrays have no resumable manifest; use another output directory")
        progress = {"identity": identity, "fields": {field: {"rows": 0} for field in fields}, "complete": False}
        atomic_json(progress_path, progress)
    id_path = directory / "id.npy"
    if id_path.exists():
        if not np.array_equal(np.load(id_path), ids):
            raise ValueError("Saved embedding recipe IDs do not match the input catalog")
    else:
        with (directory / "id.npy.tmp").open("wb") as target:
            np.save(target, ids)
            target.flush()
            os.fsync(target.fileno())
        (directory / "id.npy.tmp").replace(id_path)

    for field, texts in fields.items():
        state = progress["fields"][field]
        target = directory / f"{field}.npy"
        partial = directory / f"{field}.partial.npy"
        if target.exists():
            values = np.load(target, mmap_mode="r")
            if state["rows"] != len(ids) or values.shape[0] != len(ids) or values.dtype != np.float16:
                raise ValueError(f"Invalid completed embedding array: {target}")
            checksum = file_sha256(target)
            if state.get("sha256", checksum) != checksum:
                raise ValueError(f"Embedding checksum mismatch: {target}")
            state["sha256"] = checksum
            atomic_json(progress_path, progress)
            print(f"Embeddings {field}: already complete ({len(ids):,} recipes)", flush=True)
            continue
        if encoder is None:
            import torch
            from sentence_transformers import SentenceTransformer
            torch.set_num_threads(4)
            encoder = SentenceTransformer(identity["model"], device=device)
            encoder.max_seq_length = 256
        shape = (len(ids), encoder.get_sentence_embedding_dimension())
        if partial.exists():
            values = np.lib.format.open_memmap(partial, mode="r+")
            if values.shape != shape or values.dtype != np.float16:
                raise ValueError(f"Invalid partial embedding array: {partial}")
        else:
            if state["rows"]:
                raise ValueError(f"Missing partial embeddings for {field}")
            values = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16, shape=shape)
        for start in range(state["rows"], len(ids), chunk_size):
            end = min(len(ids), start + chunk_size)
            chunk = texts.iloc[start:end].tolist()
            if any(len(encoder.tokenizer.encode(text, truncation=False)) > 256 for text in chunk):
                raise ValueError(f"{field} exceeds MiniLM context in rows {start}:{end}")
            if field == "vectors":
                chunk = [encoder.tokenizer.decode(encoder.tokenizer.encode(text, add_special_tokens=False),
                         clean_up_tokenization_spaces=False) for text in chunk]
            encoded = encoder.encode(chunk, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
            if encoded.shape != (end-start, shape[1]) or not np.isfinite(encoded).all():
                raise ValueError(f"Invalid encoder output for {field}, rows {start}:{end}")
            values[start:end] = encoded.astype(np.float16)
            values.flush()
            with partial.open("rb") as source:
                os.fsync(source.fileno())
            state["rows"] = end
            progress["current_field"] = field
            atomic_json(progress_path, progress)
            print(f"Embeddings {field}: {end:,}/{len(ids):,}", flush=True)
        del values
        partial.replace(target)
        state["sha256"] = file_sha256(target)
        atomic_json(progress_path, progress)
    progress["complete"] = True
    atomic_json(progress_path, progress)
    atomic_json(directory / "metadata.json", {**identity, "fields": ["name", "ingredients", "joint"], "normalize_embeddings": True})
    print("All recipe embeddings complete", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind',choices=['minilm','tfidf'],default='minilm')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--chunk-size',type=int,default=4096)
    parser.add_argument('--split-dir',type=Path,default=SPLIT_DIR)
    parser.add_argument('--output-dir',type=Path)
    args = parser.parse_args()

    if args.batch_size < 1 or args.chunk_size < 1:
        parser.error('batch-size and chunk-size must be positive')
    recipes = pd.read_parquet(args.split_dir/'recipes.parquet').sort_values('id')
    directory = args.output_dir or (EMBEDDING_DIR if args.kind == 'minilm' else EMBEDDING_DIR.parent/'tfidf')
    directory.mkdir(parents=True,exist_ok=True)

    if args.kind == 'tfidf':
        np.save(directory/'id.npy',recipes.id.to_numpy())
        cutoff = pd.read_parquet(args.split_dir/'windows.parquet').start.min()
        texts = recipes.name.fillna('') + ' ' + recipes.ingredients.map(' '.join)
        encoder = TfidfVectorizer(stop_words='english',ngram_range=(1,2),max_features=30000,sublinear_tf=True,dtype=np.float32)
        encoder.fit(texts[recipes.submitted < cutoff])
        sparse.save_npz(directory/'vectors.npz',encoder.transform(texts))
        np.save(directory/'idf.npy',encoder.idf_)
        (directory/'vocabulary.json').write_text(json.dumps({k:int(v) for k,v in encoder.vocabulary_.items()}))
        (directory/'metadata.json').write_text(json.dumps(dict(fit_before=str(cutoff.date()),
            features=len(encoder.vocabulary_),fields=['name','ingredients']),indent=2))
        return
    
    generate_minilm(recipes, directory, args.device, args.batch_size, args.chunk_size)


if __name__ == '__main__':
    main()
