#!/usr/bin/env python3
"""
Example: Boltzmann Energy Re-Ranking for Attack Path Retrieval

Demonstrates the core concept — given a set of candidate facts from any
retrieval system, re-rank them using Boltzmann energy minimization so that
facts forming coherent attack chains rise to the top.

This is a standalone example using synthetic data. For the full system
with real embeddings and a knowledge base, see src/methodic.py.
"""

import math


def pairwise_energy(fact_a: str, fact_b: str, keywords_a: set, keywords_b: set) -> float:
    """
    Compute pairwise energy between two facts.
    Low energy = strong connection (likely adjacent in an attack chain).
    High energy = weak connection.

    In the full system, this uses:
    - Learned logistic regression classifier on concatenated embeddings
    - Explicit relationship edges from a knowledge graph
    - Co-occurrence statistics from MITRE ATT&CK

    Here we use keyword overlap as a simple proxy.
    """
    if not keywords_a or not keywords_b:
        return 1.0
    overlap = len(keywords_a & keywords_b) / len(keywords_a | keywords_b)
    return 1.0 - overlap  # Jaccard similarity inverted


def boltzmann_probabilities(energies: list[float], temperature: float = 0.5) -> list[float]:
    """
    Compute Boltzmann distribution over energies.
    P(i) = e^(-E_i / T) / Z  where Z = sum_j e^(-E_j / T)

    Low-energy states get high probability.
    Temperature controls sharpness: T->0 = winner-take-all, T->inf = uniform.
    """
    if not energies or temperature <= 0:
        return [1.0 / len(energies)] * len(energies) if energies else []

    scaled = [-e / temperature for e in energies]
    max_s = max(scaled)
    log_z = max_s + math.log(sum(math.exp(s - max_s) for s in scaled))
    return [math.exp(s - log_z) for s in scaled]


def beam_search_paths(
    candidates: list[dict],
    beam_width: int = 3,
    max_path_len: int = 5,
) -> list[list[int]]:
    """
    Beam search over candidate facts to find minimum-energy paths.
    Each path is a sequence of fact indices representing a coherent attack chain.
    """
    n = len(candidates)
    if n == 0:
        return []

    # Compute full pairwise energy matrix
    energy_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                energy_matrix[i][j] = pairwise_energy(
                    candidates[i]["text"],
                    candidates[j]["text"],
                    candidates[i]["keywords"],
                    candidates[j]["keywords"],
                )

    # Beam search: start from each candidate, extend greedily
    beams = [[i] for i in range(n)]  # initial beams = single facts

    for step in range(max_path_len - 1):
        new_beams = []
        for path in beams:
            last = path[-1]
            # Score extensions by energy to last node
            extensions = []
            for j in range(n):
                if j not in path:
                    extensions.append((energy_matrix[last][j], j))
            extensions.sort()  # lowest energy first

            # Keep top beam_width extensions
            for energy, j in extensions[:beam_width]:
                new_beams.append(path + [j])

        # Prune to beam_width best paths by total energy
        def path_energy(p):
            return sum(energy_matrix[p[i]][p[i + 1]] for i in range(len(p) - 1))

        new_beams.sort(key=path_energy)
        beams = new_beams[:beam_width]

    return beams


def rerank_with_boltzmann(candidates: list[dict], temperature: float = 0.5) -> list[dict]:
    """
    Full Boltzmann re-ranking pipeline:
    1. Run beam search to find minimum-energy paths
    2. Compute path energies
    3. Apply Boltzmann distribution over path ensemble
    4. Accumulate per-fact scores from all paths containing that fact
    5. Re-rank by accumulated Boltzmann score
    """
    paths = beam_search_paths(candidates, beam_width=3, max_path_len=4)

    if not paths:
        return candidates

    # Compute path energies
    path_energies = []
    for path in paths:
        energy = 0.0
        for i in range(len(path) - 1):
            energy += pairwise_energy(
                candidates[path[i]]["text"],
                candidates[path[i + 1]]["text"],
                candidates[path[i]]["keywords"],
                candidates[path[i + 1]]["keywords"],
            )
        path_energies.append(energy)

    # Boltzmann probabilities over paths
    path_probs = boltzmann_probabilities(path_energies, temperature)

    # Accumulate per-fact Boltzmann scores
    fact_scores = [0.0] * len(candidates)
    for path, prob in zip(paths, path_probs):
        for idx in path:
            fact_scores[idx] += prob

    # Blend with original relevance score
    alpha = 0.6  # original relevance weight
    beta = 0.4   # Boltzmann path weight
    for i, cand in enumerate(candidates):
        cand["boltzmann_score"] = fact_scores[i]
        cand["blended_score"] = alpha * cand.get("relevance", 0.5) + beta * fact_scores[i]

    # Sort by blended score descending
    return sorted(candidates, key=lambda c: c["blended_score"], reverse=True)


# ---------------------------------------------------------------------------
# Example: Kerberoasting attack chain
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Simulate candidates returned by a standard FAISS retrieval
    # Some are relevant chain steps, some are distractors
    candidates = [
        {
            "text": "Kerberoasting targets service accounts with SPNs registered in Active Directory",
            "keywords": {"kerberoasting", "spn", "service", "account", "active directory"},
            "relevance": 0.92,
        },
        {
            "text": "Use GetUserSPNs.py to request TGS tickets for service accounts",
            "keywords": {"getuserspns", "tgs", "ticket", "service", "account", "kerberos"},
            "relevance": 0.85,
        },
        {
            "text": "Crack TGS tickets offline using hashcat mode 13100 or john",
            "keywords": {"tgs", "ticket", "hashcat", "crack", "offline", "john"},
            "relevance": 0.78,
        },
        {
            "text": "Compromised service account credentials enable lateral movement",
            "keywords": {"service", "account", "credentials", "lateral", "movement"},
            "relevance": 0.71,
        },
        {
            "text": "Use psexec or wmiexec with service account creds for remote execution",
            "keywords": {"psexec", "wmiexec", "credentials", "remote", "execution"},
            "relevance": 0.65,
        },
        {
            "text": "SQL injection in web applications allows data exfiltration",  # distractor
            "keywords": {"sql", "injection", "web", "exfiltration"},
            "relevance": 0.70,
        },
        {
            "text": "Buffer overflow in legacy FTP servers enables RCE",  # distractor
            "keywords": {"buffer", "overflow", "ftp", "rce"},
            "relevance": 0.68,
        },
    ]

    print("=" * 70)
    print("BEFORE Boltzmann re-ranking (sorted by FAISS relevance):")
    print("=" * 70)
    by_relevance = sorted(candidates, key=lambda c: c["relevance"], reverse=True)
    for i, c in enumerate(by_relevance):
        print(f"  {i+1}. [{c['relevance']:.2f}] {c['text'][:70]}...")

    print()
    reranked = rerank_with_boltzmann(candidates, temperature=0.5)

    print("=" * 70)
    print("AFTER Boltzmann re-ranking (sorted by blended score):")
    print("=" * 70)
    for i, c in enumerate(reranked):
        print(f"  {i+1}. [{c['blended_score']:.2f}] (boltz={c['boltzmann_score']:.3f}) {c['text'][:60]}...")

    print()
    print("Notice: the attack chain steps (Kerberoasting -> GetUserSPNs -> crack")
    print("-> lateral movement -> psexec) cluster together and rise above the")
    print("distractors (SQL injection, buffer overflow), even though some")
    print("distractors had higher individual relevance scores.")
