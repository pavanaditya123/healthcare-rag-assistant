"""Download the fastembed models at image build time so pods start fast."""
from app.config import settings
from app.retrieval import Encoders

if __name__ == "__main__":
    Encoders(settings)
    print("models cached")
