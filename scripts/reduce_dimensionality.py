import pickle
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger
from sklearn.manifold import TSNE
from tqdm import tqdm

def load_embeddings(file: Path) -> Tuple[List[Path], np.ndarray]:
    embeddings = pickle.loads(file.read_bytes())
    files, vectors = zip(*list(embeddings.items()))
    return files, vectors


def apply_tSNE(vectors: np.ndarray) -> np.ndarray:
    tSNE = TSNE(n_components=2, random_state=42)
    logger.info(f"Applying {tSNE}")
    return tSNE.fit_transform(vectors)


def reduce_dimensionality(embeddings_file: Path, output_file: Path):
    files, vectors = load_embeddings(file=embeddings_file)
    points = apply_tSNE(vectors=vectors)
    points_mapping = dict(zip(files, points))
    output_file.write_bytes(pickle.dumps(points_mapping))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('embeddings_file', type=Path)
    parser.add_argument('--output-file', type=Path, required=True)
    args = parser.parse_args()
    reduce_dimensionality(
        output_file=args.output_file,
        embeddings_file=args.embeddings_file)
