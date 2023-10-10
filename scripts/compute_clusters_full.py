import pickle
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger
from sklearn.cluster import KMeans
from tqdm import tqdm
from kneed import KneeLocator

def load_embeddings(file: Path) -> Tuple[List[Path], np.ndarray]:
    embeddings = pickle.loads(file.read_bytes())
    files, vectors = zip(*list(embeddings.items()))
    return files, vectors


def make_clusters(vectors: np.ndarray, n_clusters: int) -> np.ndarray:
    kMeans = KMeans(n_clusters=n_clusters, random_state=42)
    logger.info(f"Clustering with {kMeans}")
    return kMeans.fit_predict(vectors)


def find_optimal_number_of_clusters(values: List[int], data: np.ndarray) -> int:
    logger.info("Finding optimal value for k ...")
    inertias = [KMeans(n_clusters=n, random_state=42).fit(data).inertia_ for n in tqdm(values)]
    kneedle = KneeLocator(x=values, y=inertias, direction='decreasing', curve='convex')
    logger.info(inertias)
    logger.info(f"Optimal k={kneedle.elbow}")
    return kneedle.knee

def compute_clusters(embeddings_file: Path, output_file: Path, clusters: Optional[int] = None):
    files, vectors = load_embeddings(file=embeddings_file)
    if clusters is None:
        clusters = find_optimal_number_of_clusters(range(1, 45, 1), data=vectors)
    clusters = make_clusters(vectors=vectors, n_clusters=clusters)
    cluster_mapping = dict(zip(files, clusters))
    output_file.write_bytes(pickle.dumps(cluster_mapping))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('embeddings_file', type=Path)
    parser.add_argument('--clusters', type=lambda x: int(x) if x else None, default=None, required=False)
    parser.add_argument('--output-file', type=Path, required=True)
    args = parser.parse_args()
    compute_clusters(
        embeddings_file=args.embeddings_file,
        output_file=args.output_file,
        clusters=args.clusters)
