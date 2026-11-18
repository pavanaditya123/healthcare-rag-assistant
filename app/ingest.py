"""Build the Qdrant index from MedQuAD.

    python -m app.ingest --medquad data/MedQuAD [--limit 2000]
"""
import argparse
import time

from app.config import settings
from app.data import load_medquad
from app.retrieval import Encoders, HybridRetriever, make_client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--medquad", default="data/MedQuAD")
    ap.add_argument("--limit", type=int, default=0, help="index only the first N chunks (smoke tests)")
    args = ap.parse_args()

    chunks = load_medquad(args.medquad)
    if args.limit:
        chunks = chunks[: args.limit]
    print(f"loaded {len(chunks)} chunks from {len({c.url for c in chunks})} NIH pages")

    retriever = HybridRetriever(make_client(settings), Encoders(settings), settings)
    t0 = time.time()
    retriever.create_collection()
    retriever.index(chunks)
    print(f"indexed {len(chunks)} chunks into '{settings.collection}' in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
