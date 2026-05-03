# Retrieval-Augmented Generation: A Technical Guide

Retrieval-Augmented Generation (RAG) is an architecture pattern that enhances large language models
by grounding their responses in retrieved documents rather than relying solely on parametric memory.
LLMs encode knowledge in weights during training, but weights are frozen at a knowledge cutoff date
and cannot be updated cheaply. RAG sidesteps this by keeping knowledge in an external store that is
queried at inference time.

## The Two-Stage Architecture

A RAG system has two stages: a retriever and a generator. The retriever finds document chunks
relevant to the user query. The generator — a language model — reads those chunks as context and
produces an answer. The separation is deliberate: retrieval can be updated independently of the
model, and retrieval failures are auditable in a way that weight-encoded knowledge is not.

## Why RAG Over Fine-tuning

Fine-tuning a model on proprietary data is expensive, produces a frozen snapshot, and makes it hard
to audit where an answer came from. RAG addresses all three: you update the document store without
retraining, each answer cites retrievable source chunks, and the pipeline cost is dominated by
inference rather than training. For dynamic or proprietary knowledge bases, RAG is almost always
the right architectural choice.

## Chunking

Documents must be split into chunks before embedding. Fixed-size chunking (e.g., 512 characters
with 50-character overlap) is simple and predictable. The overlap ensures a sentence that straddles
a boundary appears intact in at least one chunk, preventing retrieval misses on boundary-split
content. Semantic chunking splits on paragraph or sentence boundaries, producing variable-length
chunks that may better preserve meaning but require more complex tooling. For most production systems,
fixed-size chunking with overlap is the pragmatic default.

## Embeddings and Vector Similarity

Each chunk is converted to a dense vector by an embedding model. Models like `all-MiniLM-L6-v2`
map text to 384-dimensional space such that semantically similar text lands nearby. At query time,
the question is embedded with the same model, and the nearest chunks by cosine similarity are
retrieved. Cosine similarity measures the angle between vectors, not magnitude, which makes it
robust to variable-length text.

## Why Pure Vector Search Falls Short

Cosine similarity captures semantic proximity but fails on exact-term matching. A query for "CAP"
(the distributed systems theorem) may not rank chunks mentioning "Consistency, Availability, and
Partition tolerance" highly because the acronym and its expansion are not close in embedding space.
BM25, a probabilistic term-frequency ranking function, handles this well. Hybrid search combines
both signals.

## Reciprocal Rank Fusion

RRF merges two ranked lists without requiring calibrated scores from each system. For each chunk,
`RRF score = sum(1 / (k + rank_i))` across all lists it appears in. The constant k=60 was
established empirically by Cormack, Clarke, and Buettcher (2009) as a value that dampens top-rank
dominance while still rewarding high-ranked results. A chunk appearing in both the vector and BM25
lists scores roughly twice a chunk appearing in only one.

## Limitations

RAG is not a silver bullet. Context windows cap how many chunks can fit in a single prompt.
Retrieval misses — when the relevant chunk scores too low to appear in top-k — produce silently
wrong answers. The generator can still hallucinate details not present in retrieved context.
Chunk quality matters: a chunk that contains the answer but is poorly written scores lower than
a fluent but irrelevant chunk. Monitoring retrieval quality (e.g., checking whether the expected
source appears in retrieved chunks) is essential for production systems.
