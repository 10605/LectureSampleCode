"""Convert parquet shards with a 'text' column to one document per line.

Runs of whitespace (including newlines) become single spaces, and empty
documents are dropped, so the output suits hazsoup's line-oriented input.

    uv run python parquet_to_lines.py OUT.txt shard_00000.parquet ...
"""

import fire
import polars as pl
from tqdm import tqdm


def convert(dst, *srcs):
    with open(dst, 'w') as fp:
        for src in tqdm(srcs):
            docs = (pl.read_parquet(src, columns=['text'])['text']
                    .str.replace_all(r'\s+', ' ')
                    .str.strip_chars())
            fp.writelines(doc + '\n' for doc in docs if doc)


if __name__ == '__main__':
    fire.Fire(convert)
