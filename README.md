# Sequence Persistence Evaluation

A reproducible evaluation framework for measuring identity consistency across AI-generated video sequences, introduced in the PulseAI working paper:

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

See the full methodology in the paper or in the `figures/` diagrams.

## Reproducing the evaluation

1. Place one consented source portrait and 3–5 generated videos in `data/sample/`.
2. Run the evaluation script above.
3. Inspect the printed JSON or open `notebooks/Reproduce_SPS.ipynb` for a walkthrough.

## Citation

If you use this framework or dataset in your own work, please cite the Zenodo working paper.

## License

MIT
