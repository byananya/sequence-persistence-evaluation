# Sequence Persistence Evaluation

A standardized evaluation framework for measuring identity consistency across AI-generated video sequences, introduced in the PulseAI working paper:

> Ananya Das. (2026). *Identity Persistence in AI-Generated Video: A Sequence Evaluation Framework for Founder-Led Outbound*. PulseAI Working Paper. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX

## What this repo contains

- `eval/` — code to extract face embeddings, extract style features, and compute the Sequence Persistence Score (SPS).
- `figures/` — pipeline and evaluation flow diagrams.
- `notebooks/` — a Colab-ready notebook that walks through the full evaluation.
- `data/sample/` — placeholder for a consented sample dataset.
- `tests/` — unit tests for the metric functions.

## Quick start

```bash
pip install -r requirements.txt
python -m eval.run --source data/sample/source.jpg --videos data/sample/touch1.mp4 data/sample/touch2.mp4 data/sample/touch3.mp4
```

The script will print a JSON result like:

```json
{
  "face_persistence": 0.72,
  "style_persistence": 0.68,
  "sps": 0.70,
  "pass": true
}
```

## The Sequence Persistence Score

A sequence passes only if the sender remains recognisably the same across all moments:

| Metric | What it checks | Pass threshold |
|---|---|---|
| **Face persistence** | Cosine similarity of face embeddings between the source photo and each output, and between consecutive outputs | > 0.65 |
| **Style persistence** | Distance in background / lighting / colour-grade features between outputs | within same cluster |
| **Sequence Persistence Score (SPS)** | Weighted combination of face and style consistency across the full sequence | > 0.70 |

### Why these thresholds

- **Embedding model:** Face embeddings are extracted with **InsightFace `buffalo_l`**. Different models produce different cosine-similarity ranges, so naming the model is required for reproducibility.
- **Face-persistence threshold (0.65):** This is an initial decision boundary placed between the same-person and different-person embedding-similarity distributions. It is a v1 heuristic; multi-subject calibration is deferred to v1.1.
- **SPS threshold (0.70):** A sequence passes when the weighted combination of face and style persistence crosses this boundary.
- **Weights (0.60 face / 0.40 style):** Identity inconsistency is the dominant failure mode for outbound trust, while style drift is tolerated within bounds. The 60/40 weighting reflects that priority.
- **Scope:** The current sample validates the metric on one consented founder identity across a small set of sequences. Multi-subject calibration is deferred to v1.1.

See the full methodology in the paper or in the `figures/` diagrams.

## Reproducing the evaluation

1. Place one consented source portrait and 3–5 generated videos in `data/sample/`.
2. Run the evaluation script above.
3. Inspect the printed JSON or open `notebooks/Reproduce_SPS.ipynb` for a walkthrough.

## Threshold calibration

`eval/calibrate_threshold.py` measures the similarity distributions used to place the face-persistence threshold.

- `--source` — the portrait the sequence was generated from.
- `--outputs` — directory of generated clips for one sequence.
- `--negatives` — portraits of different people, one image each.

Reports n, mean, std, min, and max for three sets: source-to-output, output-to-output, and negative. Computes d′ between the positive and negative distributions, and sweeps the face weight from 0.5 to 0.8 to show whether any pass/fail verdict flips. Raw per-pair similarities are written to `results/calibration.csv`.

**Use at least 30 distinct negative identities.** Below roughly 20, the standard deviation is too unstable to place a threshold, and a single negative pair produces a meaningless d′.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines. Please read the [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) before participating.

## Security

If you discover a security vulnerability, please report it privately following [`SECURITY.md`](SECURITY.md).

## Citation

If you use this framework or dataset in your own work, please cite the Zenodo working paper.

## License

MIT
