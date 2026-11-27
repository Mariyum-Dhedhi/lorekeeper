"""Load the lore markdown, chunk it by section, and build the vector + BM25 index.

Run: uv run python -m lorekeeper.ingest
"""

import json
import re
import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from lorekeeper.config import INDEX_DIR, LORE_DIR, get_settings

COLLECTION = "veyra"
CHUNKS_FILE = "chunks.json"


def split_markdown_sections(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (doc title, [(section heading, section body), ...])."""
    title = ""
    sections: list[tuple[str, str]] = []
    current_heading, current_lines = "Introduction", []
    for line in text.splitlines():
        if line.startswith("# ") and not title:
            title = line[2:].strip()
            continue
        if line.startswith("## "):
            if "".join(current_lines).strip():
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading, current_lines = line[3:].strip(), []
        else:
            current_lines.append(line)
    if "".join(current_lines).strip():
        sections.append((current_heading, "\n".join(current_lines).strip()))
    return title, sections


def load_chunks(lore_dir: Path = LORE_DIR) -> list[Document]:
    s = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap
    )
    chunks: list[Document] = []
    for path in sorted(lore_dir.glob("*.md")):
        title, sections = split_markdown_sections(path.read_text(encoding="utf-8"))
        for heading, body in sections:
            for i, piece in enumerate(splitter.split_text(body)):
                source = path.stem
                chunk_id = f"{source}#{slugify(heading)}#{i}"
                chunks.append(
                    Document(
                        # Prefix with title/section so both retrievers see the context.
                        page_content=f"{title} > {heading}\n{piece}",
                        metadata={
                            "id": chunk_id,
                            "source": source,
                            "section": heading,
                            "title": title,
                        },
                    )
                )
    return chunks


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def build_index(index_dir: Path = INDEX_DIR) -> int:
    from langchain_chroma import Chroma

    from lorekeeper.retriever import chroma_settings, get_embeddings

    chunks = load_chunks()
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True)

    Chroma.from_documents(
        chunks,
        get_embeddings(),
        ids=[c.metadata["id"] for c in chunks],
        collection_name=COLLECTION,
        persist_directory=str(index_dir / "chroma"),
        client_settings=chroma_settings(),
    )
    (index_dir / CHUNKS_FILE).write_text(
        json.dumps([{"text": c.page_content, "metadata": c.metadata} for c in chunks], indent=1),
        encoding="utf-8",
    )
    return len(chunks)


def main() -> None:
    n = build_index()
    print(f"Indexed {n} chunks from {len(list(LORE_DIR.glob('*.md')))} lore documents -> {INDEX_DIR}")


if __name__ == "__main__":
    main()
