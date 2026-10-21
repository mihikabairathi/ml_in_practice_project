"""Generate Qwen recipe representations once; reruns reuse complete feature files."""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from remy.model.embedding import EMBEDDING_DIR
from remy.paths import PROJECT_ROOT
from remy.splits import SPLIT_DIR

MODEL_NAME = 'Qwen/Qwen3-Embedding-0.6B'
DESCRIPTION_WORDS = 64
MODEL_REVISION = '97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3'
HEADER_TAGS = {
    'preparation', 'time-to-make', 'course', 'main-ingredient', 'dietary', 'occasion',
    'cuisine', 'low-in-something', 'equipment', 'number-of-servings', 'taste-mood',
    'free-of-something',
}


def recipe_texts(recipes):
    return {
        'name': recipes.name.fillna('').str.strip(),
        'ingredients': recipes.ingredients.map(', '.join),
        'tags': recipes.tags.map(lambda tags: ', '.join(t.replace('-', ' ') for t in tags if t not in HEADER_TAGS)),
        'description': recipes.description.fillna('').map(lambda t: ' '.join(t.split()[:DESCRIPTION_WORDS])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    os.environ.setdefault('HF_HOME', str(PROJECT_ROOT / 'scratchpad/huggingface'))
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(2)
    recipes = pd.read_parquet(SPLIT_DIR / 'recipes.parquet').sort_values('id').reset_index(drop=True)
    model = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION, device=args.device,
                               model_kwargs={'torch_dtype': torch.float32 if args.device == 'cpu' else torch.float16})
    model.tokenizer.padding_side = 'left'
    model[0].auto_model.config.use_cache = False
    texts = recipe_texts(recipes)
    lengths = {f: [len(model.tokenizer.encode(t, add_special_tokens=True, truncation=False)) for t in values]
               for f, values in texts.items()}
    longest = max(max(v) for v in lengths.values())
    assert longest <= 32768, f'Input exceeds Qwen context: {longest}'
    model.max_seq_length = max(32, longest)
    metadata = dict(model=MODEL_NAME, revision=MODEL_REVISION, rows=len(recipes),
                    dimension=model.get_embedding_dimension(), max_seq_length=model.max_seq_length,
                    description_words=DESCRIPTION_WORDS, prompt='', device=args.device, batch_size=args.batch_size,
                    lengths={f: dict(max=max(v), p99=float(np.percentile(v, 99))) for f, v in lengths.items()})
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    plan = EMBEDDING_DIR / 'plan.json'
    if plan.exists():
        previous = json.loads(plan.read_text())
        for key in ['model', 'revision', 'rows', 'dimension', 'description_words', 'prompt']:
            assert previous[key] == metadata[key], f'Embedding settings changed: {key}'
    plan.write_text(json.dumps(metadata, indent=2))
    ids = recipes.id.to_numpy()
    if (EMBEDDING_DIR / 'id.npy').exists():
        assert np.array_equal(np.load(EMBEDDING_DIR / 'id.npy'), ids)
    np.save(EMBEDDING_DIR / 'id.npy', ids)
    numeric = np.column_stack([recipes.minutes, np.stack(recipes.nutrition), recipes.n_steps, recipes.n_ingredients]).astype(np.float32)
    assert numeric.shape == (len(recipes), 10) and np.isfinite(numeric).all() and (numeric >= 0).all()
    np.save(EMBEDDING_DIR / 'numeric.npy', numeric)
    start = time.perf_counter()
    for feature, values in texts.items():
        target = EMBEDDING_DIR / f'{feature}.npy'
        if target.exists():
            saved = np.load(target, mmap_mode='r')
            assert saved.shape == (len(recipes), metadata['dimension']) and saved.dtype == np.float16
            assert np.isfinite(saved).all()
            print(f'Reusing {feature}', flush=True)
            continue
        vectors = np.zeros((len(recipes), metadata['dimension']), dtype=np.float16)
        present = (values != '').to_numpy()
        codes, unique = pd.factorize(values[present], sort=False)
        print(f'Encoding {feature}: {len(unique):,} unique texts for {present.sum():,} recipes', flush=True)
        if present.any():
            encoded = model.encode(unique.tolist(), batch_size=args.batch_size,
                                   normalize_embeddings=True, show_progress_bar=True, prompt='')
            vectors[present] = encoded.astype(np.float16)[codes]
        assert np.isfinite(vectors).all()
        temporary = EMBEDDING_DIR / f'{feature}.partial.npy'
        np.save(temporary, vectors)
        temporary.replace(target)
        print(f'Saved {feature}; elapsed {(time.perf_counter()-start)/60:.1f} min', flush=True)
    metadata['seconds_this_run'] = time.perf_counter()-start
    (EMBEDDING_DIR / 'metadata.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
