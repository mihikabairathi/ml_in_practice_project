import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.nn import functional as F

from remy.model.embedding import EmbeddingRecommender, load_embeddings


def training_examples(quarter, embeddings, n_candidates=10000, seed=10718):
    """Historical profiles paired with future likes and five unreviewed alternatives."""
    assert (quarter.history.date < quarter.start).all()
    assert (quarter.recipes.submitted < quarter.start).all()
    assert ((quarter.outcomes.date >= quarter.start) & (quarter.outcomes.date < quarter.end)).all()

    model = EmbeddingRecommender(embeddings, method='plain', n_candidates=n_candidates)
    model.fit(quarter.history, quarter.recipes)

    positives = quarter.outcomes[quarter.outcomes.rating == 5]
    positives = positives[positives.user_id.isin(model.user_row_.index) & positives.recipe_id.isin(model.candidates_)]
    users = np.sort(positives.user_id.unique())
    user_row = pd.Series(np.arange(len(users)), index=users)
    observed = quarter.outcomes.groupby('user_id').recipe_id.agg(set).to_dict()
    rng = np.random.default_rng(seed)
    user_indices, positive_rows, negative_rows = [], [], []

    for user, group in positives.groupby('user_id'):
        blocked = model.seen_.get(user, set()) | observed.get(user, set())
        allowed = np.array([r not in blocked for r in model.candidates_])
        pool = model.candidates_[allowed]
        if not len(pool):
            continue
        sampled = rng.choice(pool, size=(len(group), 5), replace=True)
        assert not set(sampled.ravel()) & blocked
        user_indices.extend([user_row.loc[user]] * len(group))
        positive_rows.extend(model.recipe_row_.loc[group.recipe_id].tolist())
        negative_rows.extend(model.recipe_row_.loc[sampled.ravel()].to_numpy().reshape(-1, 5))

    return dict(start=quarter.start, end=quarter.end,
                profiles=model.user_vectors_[model.user_row_.loc[users].to_numpy()],
                user_rows=np.asarray(user_indices, dtype=np.int64),
                positive_rows=np.asarray(positive_rows, dtype=np.int64),
                negative_rows=np.asarray(negative_rows, dtype=np.int64).reshape(-1, 5),
                total_positives=len(quarter.outcomes[quarter.outcomes.rating == 5]))


class TrainedEmbeddingRecommender(EmbeddingRecommender):
    """Train linear scoring or nonlinear towers over frozen recipe embeddings."""

    def __init__(self, embeddings=None, n_candidates=1000, epochs=15, learning_rate=.001,
                batch_size=512, seed=10718, architecture='nonlinear', width=128,
                loss='pairwise', variant='baseline'):
        if variant not in ['baseline','softmax','separate_fields','tfidf']:
            raise ValueError('Unknown trained variant')
        if embeddings is None:
            kind = variant if variant in ['separate_fields','tfidf'] else 'joint'
            embeddings = load_embeddings(kind)
        loss = 'softmax' if variant == 'softmax' else loss
        if architecture not in ['linear', 'nonlinear'] or width < 1 or loss not in ['pairwise','softmax']:
            raise ValueError('Invalid scoring architecture')
        
        super().__init__(embeddings, method='plain', n_candidates=n_candidates)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.seed = seed
        self.architecture = architecture
        self.width = width
        self.variant = variant
        self.loss = loss

    def fit(self, interactions, recipes=None, training_quarters=(), training_data=None):
        super().fit(interactions, recipes)
        if training_data is None:
            training_data = [training_examples(q, self.embeddings, 10000, self.seed) for q in training_quarters]
        if not training_data or not sum(len(d['user_rows']) for d in training_data):
            raise ValueError('Completed historical training examples are required')
        
        torch.manual_seed(self.seed)
        torch.set_num_threads(2)
        dimension = self.vectors_.shape[1]

        if self.architecture == 'linear':
            user_layer = torch.nn.Linear(dimension, dimension, bias=False)
            with torch.no_grad():
                user_layer.weight.copy_(torch.eye(dimension))
            network = torch.nn.ModuleDict(dict(user=user_layer, recipe=torch.nn.Identity()))
        else:
            network = torch.nn.ModuleDict({key: torch.nn.Sequential(
                torch.nn.Linear(dimension, self.width), torch.nn.ReLU()) for key in ['user', 'recipe']})
            
        optimizer = torch.optim.AdamW(network.parameters(), lr=self.learning_rate, weight_decay=.0001)
        sparse_input = sparse.issparse(self.vectors_)
        if sparse_input and self.architecture != 'nonlinear':
            raise ValueError('Sparse TF-IDF inputs require nonlinear towers')
        
        vectors = self.vectors_ if sparse_input else torch.from_numpy(self.vectors_)
        offsets = np.cumsum([0] + [d['profiles'].shape[0] for d in training_data[:-1]])
        profiles = (sparse.vstack([d['profiles'] for d in training_data], format='csr') if sparse_input
                    else torch.from_numpy(np.concatenate([d['profiles'] for d in training_data])))
        users = torch.from_numpy(np.concatenate([d['user_rows'] + offset for d, offset in zip(training_data, offsets)]))
        positives = torch.from_numpy(np.concatenate([d['positive_rows'] for d in training_data]))
        negatives = torch.from_numpy(np.concatenate([d['negative_rows'] for d in training_data]))

        def project(layer, values):
            if sparse.issparse(values):
                values = values.tocsr()
                values.sum_duplicates()
                tensor = torch.sparse_csr_tensor(values.indptr, values.indices, values.data,
                            size=values.shape, dtype=torch.float32, check_invariants=True)
                first = layer[0]
                return F.relu(torch.sparse.mm(tensor, first.weight.T) + first.bias)
            return layer(values)

        self.losses_ = []
        for epoch in range(self.epochs):
            total = 0.
            for indices in torch.randperm(len(users)).split(self.batch_size):
                u = users[indices].numpy() if sparse_input else users[indices]
                p = positives[indices].numpy() if sparse_input else positives[indices]
                n = negatives[indices].numpy().ravel() if sparse_input else negatives[indices].ravel()
                preference = project(network['user'], profiles[u])
                positive = (preference * project(network['recipe'], vectors[p])).sum(dim=1)
                alternatives = project(network['recipe'], vectors[n]).reshape(len(indices), 5, -1)
                negative = (preference[:, None] * alternatives).sum(dim=2)
                if self.loss == 'softmax':
                    logits = torch.cat([positive[:, None], negative], dim=1)
                    loss = F.cross_entropy(logits, torch.zeros(len(indices), dtype=torch.long))
                else:
                    loss = F.softplus(negative - positive[:, None]).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(indices)
            self.losses_.append(total / len(users))
            print('epoch', epoch + 1, 'loss', round(self.losses_[-1], 5), flush=True)

        self.network_ = network.eval()
        self.original_user_vectors_ = self.user_vectors_.copy()
        with torch.no_grad():
            raw_user = self.original_user_vectors_ if sparse_input else torch.from_numpy(self.original_user_vectors_)
            raw_candidate = self.candidate_vectors_ if sparse_input else torch.from_numpy(self.candidate_vectors_)
            self.user_vectors_ = project(network['user'], raw_user).numpy()
            self.candidate_vectors_ = project(network['recipe'], raw_candidate).numpy()
            if sparse_input:
                self.user_vectors_[self.original_user_vectors_.getnnz(axis=1) == 0] = 0
                self.candidate_vectors_[raw_candidate.getnnz(axis=1) == 0] = 0
        if self.architecture == 'linear':
            self.transform_ = network['user'].weight.detach().numpy().copy()
        self.training_pairs_ = len(users)

        return self