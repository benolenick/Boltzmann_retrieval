# Boltzmann Energy Minimization for Attack Path Re-Ranking

Companion to [Geodesic Retrieval over Learned Manifolds](https://github.com/benolenick/geodesic_retrieval).

Standard retrieval systems rank facts by individual relevance to a query, ignoring how facts connect to form coherent procedural sequences. We model fact-to-fact connections as an energy landscape where **low-energy paths represent well-connected attack chains**, then use Boltzmann statistics to re-rank candidates.

**Key results** (75,867 cybersecurity knowledge facts):
- **+17% path completeness** over FAISS baseline
- **+35% path completeness** on attack path queries specifically
- Combined with manifold retrieval: **80% overall path completeness**
- Lightweight query router selects the best backend per query type

## How it works

1. **Query Decomposition** — LLM (qwen3.5) breaks complex queries into expert search intents (inspired by MSA in protein folding)
2. **Multi-Strategy Retrieval** — FAISS similarity + exploit index + relationship graph + co-occurrence matrix
3. **Energy Matrix Construction** — pairwise connection scores using:
   - Learned logistic regression on concatenated embeddings (ROC AUC = 0.91)
   - Explicit relationship edges (enables/bypasses/requires/escalates)
   - Co-occurrence statistics mined from MITRE ATT&CK (321K pairs from 691 techniques, 168 groups, 784 malware families)
4. **Beam Search** — find minimum-energy paths through the candidate set
5. **Boltzmann Re-Ranking** — compute P(path) = e^(-E/T) / Z over the path ensemble, accumulate per-fact scores

## Quick example

```bash
python examples/energy_rerank_example.py
```

This demonstrates Boltzmann re-ranking on a synthetic Kerberoasting attack chain. Chain steps cluster together and rise above distractors, even when distractors have higher individual relevance scores.

## Repository Structure

```
src/
  methodic.py                  # Full Methodic retrieval engine with Boltzmann energy
  mitre_cooccurrence_miner.py  # MITRE ATT&CK co-occurrence matrix builder
  train_connection_model.py    # Logistic regression connection classifier
  smart_router.py              # Query-type routing between retrieval backends
eval/
  combined_test.py             # Combined benchmark tests
  temp_sweep.py                # Boltzmann temperature parameter sweep
  test_learned_energy.py       # Learned energy function evaluation
examples/
  energy_rerank_example.py     # Standalone demo of Boltzmann re-ranking
paper/
  paper2_boltzmann.md          # Paper (markdown)
  Boltzmann_ReRanking_Paper.docx  # Paper (formatted)
```

## The protein folding analogy

| Concept | Biology | Our Implementation |
|---------|---------|-------------------|
| Co-evolutionary MI | Residues that co-mutate are physically close | Facts used together in attacks get high mutual information |
| MSA (Multiple Sequence Alignment) | Align sequences to find conserved positions | LLM decomposes query into expert search intents |
| Energy minimization | Correct fold = global energy minimum | Best attack path = minimum-energy path through knowledge graph |
| Boltzmann distribution | Thermal ensemble of protein conformations | Probability distribution over candidate attack paths |

## Complementarity with Manifold Retrieval

| Query Type | Manifold (geodesic) | Boltzmann (energy) | Best |
|------------|--------------------|--------------------|------|
| Multi-hop | **87%** | 75% | Manifold |
| Attack path | 63% | **80%** | Boltzmann |
| Constraint bypass | 69% | 71% | ~Equal |
| Cross-context | 63% | 67% | ~Equal |

The Smart Router achieves 80% overall by sending multi-hop queries to Manifold and attack path queries to Boltzmann.

## Related work

- [Geodesic Retrieval (companion paper)](https://github.com/benolenick/geodesic_retrieval) — manifold geometry approach
- [PropRAG](https://arxiv.org/abs/2504.18070) — beam search over proposition paths (no energy scoring)
- [RAGRouter](https://arxiv.org/abs/2505.23052) — learned query routing for RAG
- [RouterRetriever](https://arxiv.org/abs/2409.02685) — routing over expert embedding models

## Requirements

- Python 3.10+
- numpy, scipy, scikit-learn
- spacy (en_core_web_sm)
- faiss-cpu or faiss-gpu
- sentence-transformers (for connection model)
- A knowledge base with embeddings (we used Memoria)
- An LLM for query decomposition (we used qwen3.5 via Ollama)

## Citation

```
Olenick, B. (2026). Boltzmann Energy Minimization for Attack Path
Re-Ranking in Knowledge Retrieval. Zenodo.
```

## License

CC BY 4.0
