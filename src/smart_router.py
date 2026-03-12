#!/usr/bin/env python3
"""
Smart Router — Query-type-aware routing between Methodic and Manifold engines.

Classifies queries into categories and routes to the optimal engine(s):
  - multi_hop     → Manifold discovery + Methodic re-ranking
  - attack_path   → Methodic full pipeline (LLM decomposition + energy)
  - constraint_bypass → Manifold discovery + Methodic re-ranking
  - cross_context → Manifold discovery + Methodic re-ranking

The key insight: Manifold finds the RIGHT facts (93% path completeness on
multi-hop), Methodic puts them in the RIGHT ORDER (Boltzmann energy re-ranking).
"""

import json
import re
import time
from urllib.request import Request, urlopen

METHODIC_URL = "http://192.168.0.224:8002"
MANIFOLD_URL = "http://192.168.0.224:8003"
TIMEOUT = 120


# ---------------------------------------------------------------------------
# Query Type Classifier (keyword heuristics)
# ---------------------------------------------------------------------------

# Patterns that indicate multi-hop chains: A → B → C → shell
MULTI_HOP_PATTERNS = [
    r'\bto\b.*\bto\b',                    # "X to Y to Z"
    r'→|->|=>',                            # arrow notation
    r'\bthen\b',                           # sequential steps
    r'\bchain\b',                          # explicit chain mention
    r'\bescalat',                          # privilege escalation (multi-step)
    r'\breverse\s+shell\b',               # usually multi-hop to get shell
    r'\bpivot',                            # lateral movement
    r'\bSSH.*inject|inject.*SSH',          # SSH key injection chain
    r'\bWAR\b.*deploy|deploy.*WAR',        # WAR deployment chain
    r'\bSUID\b.*root|root.*SUID',          # SUID to root chain
]

MULTI_HOP_KEYWORDS = {
    "privilege escalation", "lateral movement", "post exploitation",
    "reverse shell", "war file", "war deploy", "ssh key injection",
    "suid binary", "gtfobins", "redis unauthenticated",
    "tomcat manager", "kerberoasting", "domain admin",
}

# Patterns for attack path planning queries
ATTACK_PATH_PATTERNS = [
    r'\bman\s+in\s+the\s+middle\b',
    r'\bMITM\b',
    r'\bARP\s+spoof',
    r'\bcredential\s+(capture|harvest|steal|sniff)',
    r'\bkerberoast',
    r'\bdomain\s+admin\b',
    r'\bbloodhound\b',
    r'\bactive\s+directory\b.*\b(attack|compromise|escalat)',
    r'\bpass.the.hash\b',
    r'\bgolden\s+ticket\b',
    r'\bsilver\s+ticket\b',
    r'\bDCSync\b',
    r'\brecon.*exploit|exploit.*recon',
]

ATTACK_PATH_KEYWORDS = {
    "arp spoofing", "mitm", "man in the middle", "credential capture",
    "credential harvesting", "kerberoasting", "bloodhound",
    "pass the hash", "golden ticket", "silver ticket", "dcsync",
    "attack path", "attack chain", "penetration test",
    "network sniff", "arp poison", "ettercap", "bettercap",
    "responder", "ntlm relay", "domain admin",
}

# Patterns for constraint bypass queries
CONSTRAINT_BYPASS_PATTERNS = [
    r'\bbypass\b',
    r'\bfilter(ed|ing)?\b',
    r'\bWAF\b',
    r'\bevasion\b',
    r'\bobfuscat',
    r'\bencode|encoding',
    r'\bblocked\b',
    r'\brestrict',
    r'\bblind\b.*\b(inject|command|sql)',
    r'\bno\s+output\b',
    r'\bout.of.band\b',
    r'\bexfiltrat',
    r'\bwhen\b.*\b(filtered|blocked|restricted)',
    r'\bwithout\b.*\b(space|quote|slash)',
]

CONSTRAINT_BYPASS_KEYWORDS = {
    "bypass", "waf", "filter", "evasion", "obfuscation",
    "double encoding", "blind injection", "out-of-band",
    "exfiltration", "blocked", "restricted", "blacklist",
    "whitelist", "sanitize", "escape", "without spaces",
}

# Cross-context: spans multiple domains (e.g., SQL injection → OS command)
CROSS_CONTEXT_PATTERNS = [
    r'\bSQL\b.*\b(OS|command|shell|RCE)',
    r'\bLOL(Bin|BAS)\b',
    r'\bliving\s+off\s+the\s+land\b',
    r'\bcertutil\b',
    r'\bbitsadmin\b',
    r'\bxp_cmdshell\b',
    r'\bINTO\s+OUTFILE\b',
    r'\b(web|sql|injection).*\b(system|command|execute)',
]

CROSS_CONTEXT_KEYWORDS = {
    "lolbin", "lolbas", "living off the land", "certutil",
    "bitsadmin", "xp_cmdshell", "into outfile", "os-shell",
    "cross context", "sql injection to",
}


def classify_query(query: str) -> str:
    """
    Classify a query into one of: multi_hop, attack_path, constraint_bypass, cross_context.

    Uses keyword matching and regex pattern scoring. Returns the category with
    the highest confidence score.
    """
    q_lower = query.lower().strip()

    scores = {
        "multi_hop": 0.0,
        "attack_path": 0.0,
        "constraint_bypass": 0.0,
        "cross_context": 0.0,
    }

    # Score each category by pattern matches
    for pattern in MULTI_HOP_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            scores["multi_hop"] += 1.5

    for pattern in ATTACK_PATH_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            scores["attack_path"] += 1.5

    for pattern in CONSTRAINT_BYPASS_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            scores["constraint_bypass"] += 1.5

    for pattern in CROSS_CONTEXT_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            scores["cross_context"] += 1.5

    # Score by keyword presence
    for kw in MULTI_HOP_KEYWORDS:
        if kw in q_lower:
            scores["multi_hop"] += 1.0

    for kw in ATTACK_PATH_KEYWORDS:
        if kw in q_lower:
            scores["attack_path"] += 1.0

    for kw in CONSTRAINT_BYPASS_KEYWORDS:
        if kw in q_lower:
            scores["constraint_bypass"] += 1.0

    for kw in CROSS_CONTEXT_KEYWORDS:
        if kw in q_lower:
            scores["cross_context"] += 1.0

    # "X to Y" pattern is a strong multi_hop signal
    if re.search(r'\b\w+\s+to\s+\w+', q_lower) and "to" in q_lower:
        # Count "to" occurrences — multiple suggests chaining
        to_count = len(re.findall(r'\bto\b', q_lower))
        if to_count >= 2:
            scores["multi_hop"] += 2.0
        elif to_count == 1:
            scores["multi_hop"] += 0.5

    # Pick highest score; default to multi_hop (Manifold+Methodic fusion)
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        # No signals — default to multi_hop (fusion is the safest bet)
        return "multi_hop"

    return best


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def http_post(url, payload, timeout=TIMEOUT):
    """POST JSON and return parsed response, or None on error."""
    data = json.dumps(payload).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Engine wrappers
# ---------------------------------------------------------------------------

def search_manifold(query, goal=None, top_k=10):
    """Manifold geodesic retrieval."""
    payload = {"query": query, "top_k": top_k}
    if goal:
        payload["goal"] = goal
    data = http_post(f"{MANIFOLD_URL}/manifold-search", payload)
    if data:
        return data.get("results", [])
    return []


def search_methodic(query, top_k=10):
    """Methodic full pipeline (LLM decomposition + Boltzmann energy)."""
    data = http_post(f"{METHODIC_URL}/smart-search", {"query": query, "top_k": top_k})
    if data:
        return data.get("results", [])
    return []


def rerank_with_methodic(candidates, query, top_k=10):
    """
    Send pre-fetched candidates to Methodic's /rerank endpoint
    for Boltzmann energy minimization re-ranking.

    Falls back to client-side relevance sorting if the endpoint
    is unavailable.
    """
    payload = {
        "query": query,
        "candidates": candidates[:20],  # send up to 20 for re-ranking
        "top_k": top_k,
    }
    data = http_post(f"{METHODIC_URL}/rerank", payload)
    if data and data.get("results"):
        return data["results"]

    # Fallback: sort by whatever relevance/score fields exist
    for c in candidates:
        c.setdefault("relevance", c.get("blended_score", c.get("combined_score", 0.5)))
    candidates.sort(key=lambda x: x.get("relevance", 0), reverse=True)
    return candidates[:top_k]


# ---------------------------------------------------------------------------
# Smart Combined Pipeline
# ---------------------------------------------------------------------------

def smart_search(query, goal=None, top_k=10):
    """
    Smart router + fusion pipeline.

    1. Classify query type
    2. Route to optimal engine(s)
    3. For discovery queries: Manifold finds candidates, Methodic re-ranks
    4. For attack_path: Methodic handles everything (LLM decomposition is key)
    """
    t0 = time.time()
    query_type = classify_query(query)

    if query_type == "attack_path":
        # Methodic excels here — use its full pipeline
        # (LLM query decomposition + multi-strategy retrieval + Boltzmann energy)
        results = search_methodic(query, top_k=top_k)
        engine_used = "methodic_full"

    elif query_type == "multi_hop":
        # Manifold excels at discovery (93% path completeness)
        # Get wide candidate pool from Manifold, then re-rank with Methodic energy
        manifold_candidates = search_manifold(query, goal=goal, top_k=top_k * 2)

        # Normalize candidate format for re-ranking
        normalized = _normalize_candidates(manifold_candidates)
        results = rerank_with_methodic(normalized, query, top_k=top_k)
        engine_used = "manifold_discovery+methodic_rerank"

    elif query_type == "constraint_bypass":
        # Manifold is better at finding bypass techniques (70% vs 60%)
        # But Methodic's edge graph can connect related bypasses
        manifold_candidates = search_manifold(query, goal=goal, top_k=top_k * 2)
        normalized = _normalize_candidates(manifold_candidates)
        results = rerank_with_methodic(normalized, query, top_k=top_k)
        engine_used = "manifold_discovery+methodic_rerank"

    else:  # cross_context — both engines are equal (88%)
        # Use Manifold for discovery + Methodic re-ranking
        manifold_candidates = search_manifold(query, goal=goal, top_k=top_k * 2)
        normalized = _normalize_candidates(manifold_candidates)
        results = rerank_with_methodic(normalized, query, top_k=top_k)
        engine_used = "manifold_discovery+methodic_rerank"

    elapsed = (time.time() - t0) * 1000

    return {
        "results": results[:top_k],
        "routing": {
            "query_type": query_type,
            "engine_used": engine_used,
            "elapsed_ms": round(elapsed, 1),
        },
    }


def _normalize_candidates(candidates):
    """
    Normalize candidate format so Methodic's re-ranker can process them.
    Manifold returns text/blended_score, Methodic expects text/relevance.
    """
    normalized = []
    for c in candidates:
        text = c.get("text") or c.get("fact") or c.get("content") or ""
        if not text:
            continue
        normalized.append({
            "text": text,
            "relevance": c.get("blended_score", c.get("relevance", c.get("combined_score", 0.5))),
            "source": c.get("retrieval_method", "manifold"),
        })
    return normalized


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_queries = [
        ("Redis unauthenticated access to SSH key injection to shell", "multi_hop"),
        ("Kerberoasting to domain admin in Active Directory", "attack_path"),
        ("command injection when spaces are filtered", "constraint_bypass"),
        ("SQL injection to operating system command execution", "cross_context"),
        ("man in the middle ARP spoofing to credential capture", "attack_path"),
        ("bypass WAF blocking slashes and dots in path traversal", "constraint_bypass"),
        ("living off the land binaries for file download on Windows", "cross_context"),
        ("Apache Tomcat manager to reverse shell", "multi_hop"),
        ("SUID binary to root privilege escalation on Linux", "multi_hop"),
        ("exfiltrate data from blind command injection with no output", "constraint_bypass"),
    ]

    print("Query Type Classifier Test")
    print("=" * 80)
    correct = 0
    for query, expected in test_queries:
        predicted = classify_query(query)
        match = predicted == expected
        correct += match
        status = "OK" if match else "MISS"
        print(f"  [{status}] {query[:55]:<55} -> {predicted:<20} (expected: {expected})")

    print(f"\nAccuracy: {correct}/{len(test_queries)} ({correct/len(test_queries):.0%})")
