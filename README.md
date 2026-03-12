# Boltzmann Energy Minimization for Re-Ranking in Knowledge Retrieval

Companion to [Geodesic Retrieval over Learned Manifolds](https://github.com/benolenick/geodesic_retrieval).

Standard retrieval systems rank facts by individual relevance to a query. But when the answer is a **sequence of connected facts**, individual relevance isn't enough — you need facts that connect to each other, not just to the query. We model fact-to-fact connections as an energy landscape and use Boltzmann statistics to re-rank candidates so that **coherent chains rise to the top**.

## The problem

Ask a vector database for "how to diagnose intermittent brake failure" and you'll get 10 facts about brakes. But the *answer* is a procedure: check brake fluid level → inspect pad wear → test caliper slides → check ABS sensor → road test. Each step connects to the next — but FAISS doesn't know that. It ranks by similarity to the query, not by how facts connect to form a coherent sequence.

This applies everywhere knowledge forms chains:
- **Medical**: history → exam → labs → imaging → differential → treatment
- **Engineering**: requirements → design → prototype → test → iterate
- **Research**: hypothesis → method → experiment → analysis → conclusion
- **Cooking**: prep → sear → deglaze → reduce → plate

## The insight (from protein folding)

Protein folding finds the correct 3D structure by minimizing free energy across the entire chain of amino acids. We apply the same principle: **the best retrieval result is the one where the entire sequence of facts has minimum energy** — meaning every fact connects strongly to its neighbors in the chain.

| Concept | Biology | Our Implementation |
|---------|---------|-------------------|
| Co-evolutionary MI | Residues that co-mutate are close | Facts used together get high mutual information |
| MSA | Align sequences to find conserved positions | LLM decomposes query into expert search intents |
| Energy minimization | Correct fold = global energy minimum | Best fact chain = minimum-energy path |
| Boltzmann distribution | Thermal ensemble of conformations | Probability distribution over candidate chains |

## How it works

1. **Retrieve candidates** from any backend (FAISS, Elasticsearch, whatever)
2. **Build pairwise energy matrix** — how well does each pair of facts connect?
   - Learned logistic regression on concatenated embeddings (ROC AUC = 0.91)
   - Explicit relationship edges (enables / requires / follows)
   - Co-occurrence statistics (which facts appear together in known procedures)
3. **Beam search** for minimum-energy paths through the candidate set
4. **Boltzmann re-ranking** — P(path) = e^(-E/T) / Z over the path ensemble. Facts appearing in low-energy paths get boosted.

The result: chain steps cluster together and rise above distractors, even when distractors had higher individual relevance scores.

## Quick example

```bash
python examples/energy_rerank_example.py
```

```
BEFORE Boltzmann re-ranking (sorted by FAISS relevance):
  1. [0.92] Kerberoasting targets service accounts with SPNs...
  2. [0.85] Use GetUserSPNs.py to request TGS tickets...
  3. [0.78] Crack TGS tickets offline using hashcat...
  4. [0.71] Compromised service account credentials enable lateral...
  5. [0.70] SQL injection in web applications allows data exfil...  ← distractor
  6. [0.68] Buffer overflow in legacy FTP servers enables RCE...    ← distractor
  7. [0.65] Use psexec or wmiexec with service account creds...

AFTER Boltzmann re-ranking (sorted by blended score):
  1. [0.82] Kerberoasting targets service accounts with SPNs...
  2. [0.77] Use GetUserSPNs.py to request TGS tickets...
  3. [0.72] Crack TGS tickets offline using hashcat...
  4. [0.68] Compromised service account credentials enable lateral...
  5. [0.63] Use psexec or wmiexec with service account creds...
  6. [0.42] SQL injection in web applications...                    ← demoted
  7. [0.41] Buffer overflow in legacy FTP servers...                ← demoted
```

The connected chain steps stay together. The unrelated distractors drop, even though their individual relevance was competitive.

## Key results

Evaluated on a cybersecurity knowledge base (75,867 facts) as a test domain, since security procedures are inherently chain-structured.

- **+17% path completeness** over FAISS baseline
- **+35% path completeness** on planning-style queries specifically
- Combined with [manifold retrieval](https://github.com/benolenick/geodesic_retrieval): **80% overall path completeness**

### Complementary strengths

| Query Type | Manifold (geodesic) | Boltzmann (energy) | Best |
|------------|--------------------|--------------------|------|
| Multi-hop chains | **87%** | 75% | Manifold |
| Planning queries | 63% | **80%** | Boltzmann |
| Constraint bypass | 69% | 71% | ~Equal |
| Cross-domain | 63% | 67% | ~Equal |

A lightweight query router selects the best backend per query type — no LLM needed for routing.

## Repository Structure

```
src/
  methodic.py                  # Full retrieval engine with Boltzmann energy
  mitre_cooccurrence_miner.py  # Co-occurrence matrix builder (321K pairs)
  train_connection_model.py    # Logistic regression connection classifier
  smart_router.py              # Query-type routing between backends
eval/
  combined_test.py             # Combined benchmark tests
  temp_sweep.py                # Boltzmann temperature parameter sweep
  test_learned_energy.py       # Learned energy function evaluation
examples/
  energy_rerank_example.py     # Standalone demo — no dependencies needed
paper/
  paper2_boltzmann.md          # Paper (markdown)
  Boltzmann_ReRanking_Paper.docx  # Paper (formatted)
```

## Use it with your own retrieval system

Boltzmann re-ranking is backend-agnostic. The core loop is:

```python
# 1. Get candidates from any retrieval system
candidates = your_faiss_index.search(query, k=20)

# 2. Compute pairwise energy (connection strength) between all pairs
energy_matrix = compute_pairwise_energy(candidates)

# 3. Beam search for minimum-energy paths
paths = beam_search(energy_matrix, beam_width=8)

# 4. Boltzmann re-rank
path_probs = boltzmann_distribution(path_energies, temperature=0.5)
reranked = accumulate_fact_scores(candidates, paths, path_probs)
```

See `examples/energy_rerank_example.py` for a complete working implementation.

## Related work

- [Geodesic Retrieval (companion paper)](https://github.com/benolenick/geodesic_retrieval) — manifold geometry approach
- [PropRAG](https://arxiv.org/abs/2504.18070) — beam search over proposition paths (no energy scoring)
- [RAGRouter](https://arxiv.org/abs/2505.23052) — learned query routing for RAG
- [RouterRetriever](https://arxiv.org/abs/2409.02685) — routing over expert embedding models

## Requirements

- Python 3.10+
- numpy, scipy, scikit-learn
- For the full system: spacy, sentence-transformers, faiss, Ollama (qwen3.5)
- For the example: **no dependencies** (pure Python)

## Citation

```
Olenick, B. (2026). Boltzmann Energy Minimization for Re-Ranking
in Knowledge Retrieval. Zenodo.
```

## License

CC BY 4.0
