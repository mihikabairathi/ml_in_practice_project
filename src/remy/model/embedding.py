import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from remy.model.popularity import Popularity
from remy.paths import PROJECT_ROOT
from remy.splits import MIN_REVIEWS

EMBEDDING_DIR = PROJECT_ROOT / 'data/processed/embeddings/minilm_joint'

def load_embeddings(kind='joint'):
    directory = EMBEDDING_DIR.parent / 'tfidf' if kind == 'tfidf' else EMBEDDING_DIR
    ids = np.load(directory / 'id.npy')
    if kind == 'tfidf':
        return dict(id=ids, vectors=sparse.load_npz(directory / 'vectors.npz'))
    result = {key: np.load(directory / f'{key}.npy') for key in ['vectors', 'name', 'ingredients']}
    if kind == 'separate_fields':
        result['vectors'] = np.concatenate([result['name'], result['ingredients']], axis=1)
    elif kind != 'joint':
        raise ValueError('Unknown embedding inputs')
    return dict(id=ids, **result)


class EmbeddingRecommender:
    """Untrained centered, TF-IDF, or historical-match recipe ranking."""

    def __init__(self, embeddings=None, method='centered', n_candidates=1000, rating_weights=None, history_k=3):
        if method not in ['centered', 'tfidf', 'top3', 'plain'] or n_candidates < 1 or history_k < 1:
            raise ValueError('Invalid model settings')
        
        self.embeddings = load_embeddings() if embeddings is None else embeddings
        self.method = method
        self.n_candidates = n_candidates
        self.history_k = history_k
        self.rating_weights = {5:1.0, 4:0.5} if rating_weights is None else rating_weights
        ids = self.embeddings['id']
        self.recipe_row_ = pd.Series(np.arange(len(ids)), index=ids)
        self.vectors_ = normalize(self.embeddings['vectors'].astype(np.float32))

    def fit(self, interactions, recipes=None, training_quarters=()):
        pop = Popularity().fit(interactions, recipes)
        self.ranking_, self.seen_ = pop.ranking_, pop.seen_
        self.candidates_ = np.array(self.ranking_[:self.n_candidates])
        self.candidate_rows_ = self.recipe_row_.loc[self.candidates_].to_numpy()

        counts = interactions.groupby('user_id').size()
        weights = interactions.rating.map(self.rating_weights).fillna(0.)
        liked = interactions[(weights > 0) & interactions.user_id.isin(counts[counts >= MIN_REVIEWS].index)]
        self.user_row_ = pd.Series(np.arange(liked.user_id.nunique()), index=np.sort(liked.user_id.unique()))
        W = sparse.csr_matrix((weights.loc[liked.index].to_numpy(np.float32),
                            (self.user_row_.loc[liked.user_id], self.recipe_row_.loc[liked.recipe_id])),
                            shape=(len(self.user_row_), self.vectors_.shape[0]))
        
        if self.method == 'centered':
            available = self.recipe_row_.loc[recipes.id].to_numpy()
            self.user_vectors_, self.candidate_vectors_ = {}, {}
            for field in ['name', 'ingredients']:
                raw = self.embeddings[field].astype(np.float32)
                mean = raw[available].mean(axis=0)
                vectors = normalize(raw) - mean
                self.user_vectors_[field] = normalize(W @ vectors) if W.shape[0] else W @ vectors
                self.candidate_vectors_[field] = normalize(vectors[self.candidate_rows_])
        elif self.method == 'tfidf':
            texts = recipes.name.fillna('') + ' ' + recipes.ingredients.map(' '.join)
            encoder = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=30000, sublinear_tf=True, dtype=np.float32)
            encoder.fit(texts)
            catalog = pd.Series(texts.to_numpy(), index=recipes.id).reindex(self.embeddings['id'], fill_value='')
            self.representations_ = encoder.transform(catalog)
            self.candidate_vectors_ = self.representations_[self.candidate_rows_]
        else:
            self.user_vectors_ = normalize(W @ self.vectors_) if W.shape[0] else W @ self.vectors_
            self.candidate_vectors_ = self.vectors_[self.candidate_rows_]

        if self.method in ['tfidf', 'top3']:
            self.history_ = {u:g.sort_values('date', kind='stable') for u,g in liked.groupby('user_id')}
        if self.method == 'top3':
            self.fields_ = {f:normalize(self.embeddings[f].astype(np.float32)) for f in ['name','ingredients']}

        return self

    def scores(self, user):
        row = self.user_row_.loc[user]
        if self.method == 'centered':
            return sum(self.candidate_vectors_[f] @ self.user_vectors_[f][row] for f in self.user_vectors_) / 2
        if self.method in ['tfidf', 'top3']:
            history = self.history_[user]
            rows = self.recipe_row_.loc[history.recipe_id].to_numpy()
            weights = history.rating.map(self.rating_weights).to_numpy(np.float32)

            if self.method == 'tfidf':
                profile = normalize(self.representations_[rows].multiply(weights[:,None]).sum(axis=0).A, copy=False)
                values = np.asarray(self.candidate_vectors_ @ profile.T).ravel()
                return np.where((self.candidate_vectors_.getnnz(axis=1)>0) & profile.any(), values, np.nan)
            
            values = np.empty(len(self.candidates_), dtype=np.float32)
            n = min(self.history_k, len(rows))
            for begin in range(0, len(values), 256):
                candidates = self.candidate_rows_[begin:begin+256]
                pairs = sum(v[candidates] @ v[rows].T for v in self.fields_.values()) / 2
                best = np.argpartition(pairs, -n, axis=1)[:, -n:]
                selected = np.take_along_axis(pairs, best, axis=1)
                values[begin:begin+256] = (selected * weights[best]).sum(axis=1) / weights[best].sum(axis=1)
            return values
        
        profile = self.user_vectors_[row]
        if sparse.issparse(profile):
            values = (self.candidate_vectors_ @ profile.T).toarray().ravel()
            return np.where((self.candidate_vectors_.getnnz(axis=1)>0)&(profile.nnz>0),values,np.nan)
        values = self.candidate_vectors_ @ profile
        return np.where(self.candidate_vectors_.any(axis=1) & profile.any(), values, np.nan)

    def recommend_many(self, user_ids, k=None):
        if k is not None and k < 0:
            raise ValueError('k must be nonnegative')
        recommendations = {}
        for user in user_ids:
            if user not in self.user_row_.index:
                continue
            scores = self.scores(user)
            valid = np.isfinite(scores) & np.array([r not in self.seen_.get(user, ()) for r in self.candidates_])
            ids = self.candidates_[valid]
            order = np.lexsort((ids, -scores[valid]))
            recommendations[user] = ids[order].tolist()[:k]
        return recommendations
