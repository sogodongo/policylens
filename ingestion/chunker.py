import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter


# cl100k_base is the tokenizer used by text-embedding-3-large
_tokenizer = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text))


def chunk_blocks(blocks: list[dict], chunk_size: int = 512, overlap: int = 77) -> list[dict]:
    """
    Takes the raw blocks from the PDF parser and splits them into
    consistently-sized chunks suitable for embedding.

    Each chunk inherits the metadata from its parent block so we don't
    lose heading/page/jurisdiction context during retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=_count_tokens,
        # Split on paragraph breaks first, then sentences, then words
        # This keeps clauses intact as long as possible before hard-splitting
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    for block in blocks:
        # Prepend heading so every chunk carries its section context
        if block["heading"]:
            text_to_split = f"{block['heading']} | {block['text']}"
        else:
            text_to_split = block["text"]

        splits = splitter.split_text(text_to_split)

        for i, split_text in enumerate(splits):
            chunks.append({
                "text": split_text,
                "heading": block["heading"],
                "page": block["page"],
                "doc_title": block["doc_title"],
                "source_url": block["source_url"],
                "doc_type": block["doc_type"],
                "jurisdiction": block["jurisdiction"],
                # Track position within parent block — useful for debugging
                "chunk_index": i,
            })

    return chunks
