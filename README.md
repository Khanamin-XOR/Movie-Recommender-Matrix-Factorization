# Movie Recommender — Matrix Factorization from Scratch

A collaborative-filtering movie recommender built on the **classical SVD++ / Netflix Prize-style** matrix-factorization model — implemented from scratch in NumPy with **SGD-trained bias terms** and Truncated SVD-initialised user/movie embeddings.

The model is the additive form:

```
ŷ_ij = μ + b_i + c_j + u_iᵀ v_j
```

where `μ` is the global mean rating, `b_i` and `c_j` are user- and movie-specific bias terms learned by SGD, and `u_i`, `v_j` are k-dimensional latent vectors obtained from SVD on the rating matrix.

## Dataset

MovieLens-style ratings dataset:
- 943 users
- 1,681 movies
- 89,992 ratings (user, movie, rating triples on a 1–5 scale)
- Modelled as a weighted bipartite graph → sparse adjacency matrix (CSR)

## What's implemented

### Task 1 — Matrix Factorization with SGD-trained biases

1. **Adjacency matrix construction** — sparse CSR representation of the user × movie rating matrix
2. **Truncated SVD initialisation** — `randomized_svd` to factorise into `U`, `Σ`, `Vᵀ` and obtain initial latent vectors
3. **Loss function** — regularised squared-error MF objective:

```
   L = α · (‖U‖² + ‖V‖² + ‖b‖² + ‖c‖²)
       + Σ_{i,j ∈ Train} (y_ij − μ − b_i − c_j − u_iᵀ v_j)²
```

4. **Analytical gradients** — `dL/db_i` and `dL/dc_j` derived and implemented as separate functions
5. **SGD training loop** — 200 epochs over all (user, movie, rating) triples, updating bias terms each pass
6. **MSE tracking** — per-epoch MSE logged and plotted

### Task 2 — Probing the learned embeddings for gender signal

A research-style question on the learned representations: *Do the latent user vectors `U` — trained purely to predict movie ratings — implicitly encode demographic information?*

- Trained an `SGDClassifier(loss='log')` on `U` (943 × 30) to predict the `is_male` label from `user_info.csv`
- Confusion matrix evaluated before and after `StandardScaler` normalisation of the embeddings
- Investigates the kind of *implicit attribute encoding* that has motivated recent fairness/auditing work in recommender systems

## Results

| Quantity | Value |
|---|---|
| Users × Movies | 943 × 1,681 |
| Training ratings | 89,992 |
| Latent dimension `k` (Task 1) | 5 |
| Latent dimension `k` (Task 2) | 30 |
| SGD epochs | 200 |
| Learning rate | 1e-3 |
| Regularisation `α` | 0.01 |
| **Training MSE (final)** | **0.833** |
| Training MSE (initial) | 0.892 |

Smooth monotonic descent over 200 epochs (plotted in the notebook), confirming the gradient derivations and SGD implementation are correct.

## Why from-scratch

`surprise.SVD()` solves this problem in three lines. The point of this project was instead to:

1. Derive the gradients of the regularised MF loss analytically and implement them in NumPy
2. Run pure-Python SGD over biases while using SVD-initialised latent factors — making the contribution of each part of the additive model visible
3. Use the resulting embeddings for downstream **embedding probing** — a methodology that is now standard in NLP fairness research, applied here to collaborative filtering

## Tech stack

Python 3 · NumPy · pandas · SciPy (`csr_matrix`) · scikit-learn (`randomized_svd`, `SGDClassifier`, metrics) · matplotlib · seaborn · Jupyter

## Run

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn jupyter
jupyter notebook Recommendation_system_assignment_sloutions.ipynb
```

**Note on data:** the `ratings_train.csv` and `user_info.csv` files used in the notebook are MovieLens-100K-style triplet data. They are not committed here; the notebook's outputs are preserved inline so the full analysis and final MSE / confusion matrices are visible end-to-end without re-execution. Equivalent data is available from [GroupLens MovieLens](https://grouplens.org/datasets/movielens/100k/).

## Related projects

This repo is part of a from-scratch series:
- **[PCA-From-Scratch](https://github.com/Khanamin-XOR/PCA-From-Scratch)** — PCA derivation and implementation in NumPy, with Medium publication
- **[ML-Metrics-From-Scratch](https://github.com/Khanamin-XOR/ML-Metrics-From-Scratch)** — classification + regression metrics implemented from definition

## License

MIT
