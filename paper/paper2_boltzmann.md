# Boltzmann Energy Minimization for Attack Path Re-Ranking in Knowledge Retrieval

## Short/Workshop Paper — Companion to "Geodesic Retrieval over Learned Manifolds"

### Abstract
Standard retrieval systems return facts ranked by individual relevance to a query, ignoring how facts connect to form coherent procedural sequences. We propose Boltzmann Energy Re-Ranking: modeling fact-to-fact connections as an energy landscape where low-energy paths represent well-connected attack chains. Given candidate facts from any retrieval system, we construct a pairwise energy matrix (using learned classifiers, explicit relationship edges, and co-occurrence statistics), run beam search to find minimum-energy paths, then compute Boltzmann probabilities over the path ensemble to re-rank candidates. Evaluated on 75,867 cybersecurity knowledge facts, Boltzmann re-ranking improves path completeness by +17% over FAISS baseline. Combined with geometric manifold retrieval (companion paper), the two approaches achieve complementary strengths: manifold retrieval excels at multi-hop discovery (+28% path completeness), while Boltzmann re-ranking excels at attack path planning (+35% path completeness). A lightweight query router combining both achieves 80% overall path completeness — +20% over FAISS at 3.2s average latency.

---

### Citations to add to full paper (in .docx)

The following must be added to the Related Work and References sections of `Boltzmann_ReRanking_Paper.docx`:

**Related Work — Beam Search in Retrieval:**
> PropRAG [X] uses beam search over proposition-level knowledge graphs to discover multi-step reasoning chains, achieving state-of-the-art zero-shot performance on multi-hop QA benchmarks without LLM inference at query time. Our beam search differs in its scoring function: where PropRAG scores paths by embedding similarity and graph connectivity, we score by Boltzmann energy — a thermodynamic formulation that models the entire path ensemble rather than selecting a single best path.

**Related Work — Query Routing:**
> Recent work on retrieval routing includes RAGRouter [Y], which learns to route queries to the most suitable retrieval-augmented LLM from a pool of candidates, and RouterRetriever [Z], which routes over a mixture of domain-specific expert embedding models. Our query router is simpler — a lightweight keyword-based classifier that routes by query type (multi-hop vs. attack path) rather than learning routing policies — but the routing principle is shared.

**References to add:**
- PropRAG: J. Wang et al., "PropRAG: Guiding Retrieval with Beam Search over Proposition Paths," arXiv:2504.18070, 2025.
- RAGRouter: "RAGRouter: Learning to Route Queries to Multiple Retrieval-Augmented Language Models," arXiv:2505.23052, 2025.
- RouterRetriever: "RouterRetriever: Exploring the Benefits of Routing over Multiple Expert Embedding Models," arXiv:2409.02685, 2024.
