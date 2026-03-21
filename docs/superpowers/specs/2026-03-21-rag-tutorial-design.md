# Design: Notebook 06 — RAG from Scratch

**Date:** 2026-03-21
**Series:** Transformer Tutorial (01–05)
**Filename:** `06-retrieval-augmented-generation.ipynb`

---

## Goal

Add a sixth notebook to the transformer tutorial series that teaches Retrieval-Augmented Generation (RAG) from scratch. The notebook targets learners who have completed notebooks 01–05 and understand transformer architecture, attention, and embeddings.

## Audience

Same as 01–05: Chinese learners new to LLMs/RAG who understand the transformer architecture. No prior RAG or LLM experience assumed.

## Approach

Build RAG from scratch using only `numpy`, `torch`, and `matplotlib` — no external APIs, no LangChain, no vector DB libraries. This matches the series philosophy of building to understand, not using to ship.

## Dependencies

- `numpy` — vector math, TF-IDF
- `torch` — tensor operations (no training required in main flow)
- `matplotlib` — visualizations

No new installs required beyond the existing series requirements.

---

## Notebook Structure

### Section 1: What is RAG?
- The problem: LLMs have a knowledge cutoff and hallucinate facts
- The RAG solution: retrieve relevant context at query time, then generate
- ASCII diagram: `Query → Embed → Search VectorStore → Augment Prompt → Generate`
- Key insight: RAG = a retrieval system + a prompt template. The "generation" step can be as simple as filling a template.

### Section 2: The Knowledge Base
- Build a corpus of ~50 text passages on a consistent domain (e.g., animal facts: "The blue whale is the largest animal on Earth. It can grow up to 30 meters long.")
- Stored as a plain Python list of strings
- Larger corpus (50 vs 20) makes top_k comparisons meaningful and retrieval more realistic
- Motivates the need for retrieval: we can't fit all 50 facts in a prompt context

### Section 3: TF-IDF Embeddings from Scratch

**Vocabulary construction:**
- Fit vocabulary on the corpus only (not on queries), using simple whitespace tokenization + lowercase
- IDF = log(N / df) where N = number of documents, df = number of documents containing the term
- TF = count of term in document / total terms in document
- TF-IDF vector for a document = elementwise TF × IDF, then L2-normalized

**Query vectorization:**
- At search time, a query is tokenized and projected into the same vocabulary space using the corpus IDF weights
- Unknown words (not in vocabulary) are silently ignored
- Result: query and document vectors live in the same space and are directly comparable

**Honest framing:**
- TF-IDF measures *lexical* similarity (shared words), not *semantic* similarity
- Explicitly note this limitation: "The cat sat on the mat" and "A feline rested on the rug" would have zero similarity under TF-IDF
- This sets up Section 9's failure modes and Exercise 3

**Visualization:**
- PCA scatter plot of all 50 document vectors, colored by topic cluster
- Shows that TF-IDF groups lexically similar documents, but not semantically related ones

### Section 4: Vector Store from Scratch

**Data model:**
- `VectorStore` owns the corpus. It is the single source of truth for both texts and vectors.
- No separate corpus list — the `VectorStore` holds `list[tuple[str, np.ndarray]]`

**Interface:**
```python
class VectorStore:
    def add(self, text: str) -> None      # embed and append
    def search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]
    # returns list of (text, score) sorted by descending similarity
```

### Section 5: Cosine Similarity Deep Dive
- Why cosine? Measures the angle between vectors — captures direction (meaning pattern), not magnitude (document length)
- Show formula: `cos(u, v) = (u · v) / (||u|| × ||v||)`
- Implement in numpy, verify against `np.dot`
- Visualize: similarity matrix heatmap (50×50) across all corpus passages
- Show that passages about the same animal cluster together
- Connect to notebook 04: cosine similarity is a scaled dot product, just like attention scores

### Section 6: The Retriever
- `Retriever` wraps `VectorStore` with no extra state:
  ```python
  class Retriever:
      def __init__(self, store: VectorStore): ...
      def retrieve(self, query: str, top_k: int = 3) -> list[str]
      # returns texts only (strips scores for downstream use)
  ```
- Run 5 example queries, show top-3 retrieved passages for each
- Bar chart of similarity scores for a single query — shows the score gap between top result and the rest

### Section 7: The Generator (Mock LLM)

**Mechanism (fully specified):**
The `MockLLM` does not perform any language generation. It fills a deterministic prompt template and returns the highest-scoring retrieved passage as the "answer":

```python
class MockLLM:
    def generate(self, query: str, context_passages: list[str]) -> str:
        context = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(context_passages))
        # "Answer" = the first (highest-scoring) passage, truncated to 1 sentence
        answer = context_passages[0].split(".")[0] + "."
        return f"Context:\n{context}\n\nQuestion: {query}\nAnswer: {answer}"
```

**Pedagogical purpose:** This makes explicit that "generation" in RAG is just prompt construction. The quality of the answer depends entirely on retrieval quality, not on a language model. Learners see that RAG is fundamentally a retrieval + formatting problem.

### Section 8: Full RAG Pipeline

```python
class RAGPipeline:
    def __init__(self, retriever: Retriever, generator: MockLLM): ...
    def query(self, q: str, top_k: int = 3) -> str: ...
```

- Run 8 end-to-end examples: 5 in-distribution (query matches corpus topic) + 3 edge cases
- Side-by-side comparison: RAG output vs. no-retrieval baseline (MockLLM with empty context)
- Makes clear that RAG only helps when retrieval succeeds

### Section 9: Failure Modes

Three concrete demonstrations:

1. **Out-of-distribution query** — ask about a topic not in the corpus. Show that the top-1 result has a low score and is semantically irrelevant. Introduce a confidence threshold: reject retrieval if max score < 0.1.

2. **top_k sensitivity** — with 50 passages, compare top_k=1, 3, 5, 10. Show that top_k=1 misses relevant passages; top_k=10 introduces noise. Visualize as a recall curve approximation.

3. **Lexical gap failure** — query uses synonyms not present in the corpus (e.g., query "feline" when corpus uses "cat"). TF-IDF similarity = 0. Demonstrates the core limitation of lexical embeddings and motivates semantic embeddings.

### Section 10: Exercises

1. Add 10 new passages to the knowledge base. Test that new queries retrieve them correctly. Verify old queries are unaffected. (Tests: retrieval updates correctly)

2. Swap cosine similarity for dot product (remove L2 normalization). How does ranking change? Why? (Tests: understanding of normalization's role)

3. Implement a confidence threshold in `Retriever.retrieve()`: return an empty list if the top score is below a threshold. How does this change `RAGPipeline` output? (Tests: graceful degradation)

4. Change the corpus to a different domain (e.g., Python programming tips). Does the pipeline work out of the box? (Tests: generalization)

---

## Data Flow Summary

```
corpus (list[str])
    → VectorStore.add() → stores (text, tfidf_vector) pairs
    → Retriever.retrieve(query) → top-k texts
    → MockLLM.generate(query, texts) → formatted response string
    → RAGPipeline.query(q) → final answer
```

Each class has one responsibility. No class holds a reference to another's internal data.

---

## Key Design Decisions

1. **TF-IDF first, with honest framing** — Immediately teachable, no training required. Limitations are explicitly demonstrated in Section 9, not hidden.

2. **MockLLM with specified mechanism** — Returns first retrieved passage as answer. Learner sees that "generation" is prompt formatting; the bottleneck is retrieval.

3. **Corpus of 50 passages** — Large enough for top_k comparisons to be meaningful (top_k=10 = 20% of corpus, not 50%).

4. **VectorStore owns the corpus** — Single source of truth. Retriever and RAGPipeline do not hold duplicate text data.

5. **No learned embeddings in main flow** — Exercise 4 references swapping similarity metric; a full learned-embedding exercise requires a training objective which is out of scope. Exercises focus on things buildable in a few lines.

6. **Failure modes section** — RAG tutorials often skip this. Teaching failure modes builds real intuition about when to use RAG.

---

## Success Criteria

- Learner can explain what RAG is and why it helps with hallucination and knowledge cutoffs
- Learner can build a TF-IDF vector store and retriever from scratch
- Learner understands that cosine similarity measures vector direction (lexical overlap for TF-IDF) and can articulate the difference between lexical and semantic similarity
- Learner can identify three specific failure modes of RAG
- Learner has successfully completed at least one exercise (added documents and verified retrieval behavior changed)
- No external dependencies beyond existing series requirements
