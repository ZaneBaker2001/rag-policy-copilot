import sys
from pathlib import Path

from tqdm import tqdm

# Ensure project root is on sys.path when running as a script:
# python3 scripts/build_index.py
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.ingest import build_chunks_from_file
from app.retriever import VectorStore
from app.utils import ensure_dir, iter_files


def main() -> None:
    ensure_dir(settings.storage_dir)
    ensure_dir(settings.data_dir)

    vs = VectorStore()
    vs.reset()

    all_files = sorted(iter_files(settings.data_dir))
    if not all_files:
        print(f"No supported files found in {settings.data_dir}")
        return

    total_chunks = 0
    indexed_files = 0
    failed_files: list[tuple[str, str]] = []

    for file_path in tqdm(all_files, desc="Indexing files"):
        try:
            chunks = build_chunks_from_file(file_path)
            if not chunks:
                print(f"Skipped {file_path}: no extractable text")
                continue

            vs.add_chunks(chunks)
            total_chunks += len(chunks)
            indexed_files += 1

        except Exception as exc:
            failed_files.append((file_path, str(exc)))
            print(f"Failed to index {file_path}: {exc}")

    vs.save()

    print(
        f"Done. Indexed {indexed_files} files, {total_chunks} chunks, "
        f"failed {len(failed_files)} files."
    )

    if failed_files:
        print("\nFailed files:")
        for file_path, error in failed_files:
            print(f"- {file_path}: {error}")


if __name__ == "__main__":
    main()