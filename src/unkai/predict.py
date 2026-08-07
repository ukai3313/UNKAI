import argparse
import os

import numpy as np
import torch

from .model import OriginalUNKAI, SeenUnseenUNKAI


def load_embedding(path):
    arr = np.load(path)

    if arr.ndim == 3:
        arr = arr.squeeze(0)

    if arr.ndim != 2:
        raise ValueError(f"Expected embedding shape (L, D) or (1, L, D), got {arr.shape}")

    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
    mask = torch.ones(tensor.shape[:2], dtype=torch.bool)

    return tensor, mask


def build_model(model_name):
    if model_name == "original":
        return OriginalUNKAI()

    if model_name == "seen_unseen":
        return SeenUnseenUNKAI()

    raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Run UNKAI inference for a protein pair."
    )

    parser.add_argument(
        "--model",
        choices=["original", "seen_unseen"],
        required=True,
        help="UNKAI model variant.",
    )

    parser.add_argument(
        "--protein1",
        required=True,
        help="UniProt accession of protein 1.",
    )

    parser.add_argument(
        "--protein2",
        required=True,
        help="UniProt accession of protein 2.",
    )

    parser.add_argument(
        "--embeddings-dir",
        required=True,
        help="Directory containing *_embedding.npy files.",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint (.pth).",
    )

    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Inference device. Default: CUDA if available, otherwise CPU.",
    )

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    embedding1_path = os.path.join(
        args.embeddings_dir,
        f"{args.protein1}_embedding.npy",
    )

    embedding2_path = os.path.join(
        args.embeddings_dir,
        f"{args.protein2}_embedding.npy",
    )

    if not os.path.exists(embedding1_path):
        raise FileNotFoundError(f"Embedding not found: {embedding1_path}")

    if not os.path.exists(embedding2_path):
        raise FileNotFoundError(f"Embedding not found: {embedding2_path}")

    mat1, mask1 = load_embedding(embedding1_path)
    mat2, mask2 = load_embedding(embedding2_path)

    model = build_model(args.model)

    state_dict = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    mat1 = mat1.to(device)
    mat2 = mat2.to(device)
    mask1 = mask1.to(device)
    mask2 = mask2.to(device)

    with torch.inference_mode():
        output = model(mat1, mat2, mask1, mask2)

        if args.model == "original":
            probability = output.item()
        else:
            probability = torch.sigmoid(output).item()

    prediction = int(probability >= 0.5)

    print(f"Protein 1   : {args.protein1}")
    print(f"Protein 2   : {args.protein2}")
    print(f"Model       : {args.model}")
    print(f"Probability : {probability:.6f}")
    print(f"Prediction  : {prediction}")


if __name__ == "__main__":
    main()
