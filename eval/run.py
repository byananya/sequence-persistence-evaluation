"""CLI entry point to run the Sequence Persistence evaluation."""

import argparse
import json
from pathlib import Path

from eval.compute_sps import compute_sps
from eval.extract_faces import get_face_embedding, get_face_embedding_from_video, load_model
from eval.extract_style import get_style_vector, get_style_vector_from_video


def main():
    parser = argparse.ArgumentParser(
        description="Run the Sequence Persistence evaluation on a generated video sequence."
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Path to the source portrait image.",
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        required=True,
        type=Path,
        help="Paths to the generated sequence videos, in order.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results.json"),
        help="Path to write the JSON results.",
    )
    parser.add_argument(
        "--frame-offset",
        type=float,
        default=0.5,
        help="Relative frame position to sample from each video (0.0–1.0).",
    )
    args = parser.parse_args()

    print("Loading face model...")
    model = load_model()

    print(f"Extracting features from {args.source} and {len(args.videos)} videos...")
    source_face = get_face_embedding(args.source, model)
    source_style = get_style_vector(args.source)

    face_embeddings = [
        get_face_embedding_from_video(v, model, args.frame_offset) for v in args.videos
    ]
    style_vectors = [
        get_style_vector_from_video(v, frame_offset=args.frame_offset) for v in args.videos
    ]

    result = compute_sps(source_face, face_embeddings, source_style, style_vectors)
    print(json.dumps(result, indent=2))

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
