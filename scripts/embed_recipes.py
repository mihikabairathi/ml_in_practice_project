"""Create the frozen inputs used by the two recipe-recommender notebooks."""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

from remy.model.embedding import EMBEDDING_DIR
from remy.splits import SPLIT_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind',choices=['minilm','tfidf'],default='minilm')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--batch-size',type=int,default=64)
    args = parser.parse_args()

    recipes = pd.read_parquet(SPLIT_DIR/'recipes.parquet').sort_values('id')
    directory = EMBEDDING_DIR if args.kind == 'minilm' else EMBEDDING_DIR.parent/'tfidf'
    directory.mkdir(parents=True,exist_ok=True)
    np.save(directory/'id.npy',recipes.id.to_numpy())

    if args.kind == 'tfidf':
        cutoff = pd.read_parquet(SPLIT_DIR/'windows.parquet').start.min()
        texts = recipes.name.fillna('') + ' ' + recipes.ingredients.map(' '.join)
        encoder = TfidfVectorizer(stop_words='english',ngram_range=(1,2),max_features=30000,sublinear_tf=True,dtype=np.float32)
        encoder.fit(texts[recipes.submitted < cutoff])
        sparse.save_npz(directory/'vectors.npz',encoder.transform(texts))
        np.save(directory/'idf.npy',encoder.idf_)
        (directory/'vocabulary.json').write_text(json.dumps({k:int(v) for k,v in encoder.vocabulary_.items()}))
        (directory/'metadata.json').write_text(json.dumps(dict(fit_before=str(cutoff.date()),
            features=len(encoder.vocabulary_),fields=['name','ingredients']),indent=2))
        return
    
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device=args.device)
    model.max_seq_length = 256
    fields = dict(name=recipes.name.fillna(''),ingredients=recipes.ingredients.map(', '.join))
    fields['vectors'] = 'Name: '+fields['name']+'\nIngredients: '+fields['ingredients']
    for field,texts in fields.items():
        target = directory/f'{field}.npy'
        if target.exists():
            continue
        lengths = [len(model.tokenizer.encode(t,truncation=False)) for t in texts]
        if max(lengths) > model.max_seq_length:
            raise ValueError(f'{field} exceeds MiniLM context')
        if field == 'vectors':
            texts = [model.tokenizer.decode(model.tokenizer.encode(t,add_special_tokens=False), clean_up_tokenization_spaces=False) for t in texts]
        vectors = model.encode(list(texts),batch_size=args.batch_size,normalize_embeddings=True, show_progress_bar=True).astype(np.float16)
        np.save(target,vectors)
        
    (directory/'metadata.json').write_text(json.dumps(dict(model='sentence-transformers/all-MiniLM-L6-v2',
        fields=['name','ingredients','joint'],max_seq_length=256),indent=2))


if __name__ == '__main__':
    main()