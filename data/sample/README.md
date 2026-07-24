# Sample dataset

This folder is a placeholder for a consented sample dataset used to reproduce the Sequence Persistence Score.

## What to add

1. `source.jpg` — a single consented portrait photo of the sender.
2. `touch1.mp4`, `touch2.mp4`, `touch3.mp4` — generated videos for a founder-led outbound sequence (e.g., first outreach, follow-up, post-call reminder).

All assets must be shared with explicit consent from the person pictured. Do not commit private or unconsented portraits to this public repo.

Once the files are in place, run:

```bash
python -m eval.run --source data/sample/source.jpg --videos data/sample/touch1.mp4 data/sample/touch2.mp4 data/sample/touch3.mp4
```
