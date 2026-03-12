#!/usr/bin/env python3
"""
A/B test: keyword_overlap vs learned_energy in Methodic Retrieval.

Tests the same queries with learned_energy on vs off (falling back to keyword overlap),
and compares Recall@5, Recall@10, and Path Completeness.
"""

import json
import time
import sys
from urllib.request import Request, urlopen
from urllib.error import URLError

METHODIC_URL = "http://192.168.0.224:8002"
TIMEOUT = 120

TEST_CASES = [
    {
        "id": "CB-01", "cat": "constraint_bypass",
        "query": "command injection when spaces are filtered",
        "keywords": ["${IFS}", "brace expansion", "$IFS", "tab", "command injection", "space"],
        "chain": ["space filtering", "IFS substitute", "brace expansion", "command execution"],
    },
    {
        "id": "CB-02", "cat": "constraint_bypass",
        "query": "exfiltrate data from blind command injection with no output",
        "keywords": ["out-of-band", "DNS", "curl", "wget", "time-based", "blind", "exfiltration"],
        "chain": ["blind injection", "OOB listener", "DNS exfiltration", "data extraction"],
    },
    {
        "id": "MH-01", "cat": "multi_hop",
        "query": "Tomcat manager default credentials to reverse shell",
        "keywords": ["tomcat", "manager", "WAR", "deploy", "reverse shell", "msfvenom", "default"],
        "chain": ["find tomcat manager", "default credentials", "WAR deployment", "reverse shell"],
    },
    {
        "id": "MH-02", "cat": "multi_hop",
        "query": "SUID binary to root shell on Linux",
        "keywords": ["SUID", "find", "GTFOBins", "privilege escalation", "root", "/bin/bash"],
        "chain": ["find SUID binaries", "check GTFOBins", "escalate privileges", "root shell"],
    },
    {
        "id": "AP-01", "cat": "attack_path",
        "query": "SQL injection to RCE on MSSQL",
        "keywords": ["SQL injection", "xp_cmdshell", "MSSQL", "RCE", "stacked queries", "exec"],
        "chain": ["find SQL injection", "enable xp_cmdshell", "execute OS commands", "RCE"],
    },
    {
        "id": "AP-02", "cat": "attack_path",
        "query": "Kerberoasting attack chain",
        "keywords": ["Kerberoasting", "SPN", "TGS", "hashcat", "service account", "domain"],
        "chain": ["domain user access", "enumerate SPNs", "request TGS tickets", "offline crack"],
    },
    {
        "id": "CC-01", "cat": "cross_context",
        "query": "Docker container escape to host",
        "keywords": ["docker", "escape", "socket", "privileged", "mount", "nsenter", "container"],
        "chain": ["identify docker access", "check socket/privileged", "escape to host"],
    },
    {
        "id": "CC-02", "cat": "cross_context",
        "query": "Redis unauthenticated access to SSH key injection",
        "keywords": ["Redis", "CONFIG SET", "SSH", "authorized_keys", "unauthenticated"],
        "chain": ["connect to Redis", "CONFIG SET dir/dbfilename", "inject SSH key", "SSH access"],
    },
]


def http_post(url: str, payload: dict) -> dict | None:
    try:
        data = json.dumps(payload).encode()
        req = Request(url, data=data, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def set_features(features: dict) -> dict | None:
    return http_post(f"{METHODIC_URL}/features", features)


def run_query(query: str) -> dict | None:
    return http_post(f"{METHODIC_URL}/retrieve", {"query": query, "top_k": 10})


def score_results(results: list[dict], test_case: dict) -> dict:
    """Score results against expected keywords and chain."""
    texts = [r.get("text", "").lower() for r in results]
    all_text = " ".join(texts)

    # Recall@5 and Recall@10
    kw_hits_5 = sum(1 for kw in test_case["keywords"] if any(kw.lower() in t for t in texts[:5]))
    kw_hits_10 = sum(1 for kw in test_case["keywords"] if any(kw.lower() in t for t in texts[:10]))
    total_kw = len(test_case["keywords"])

    recall_5 = kw_hits_5 / total_kw if total_kw else 0
    recall_10 = kw_hits_10 / total_kw if total_kw else 0

    # Path completeness: how many chain steps are covered in all results
    chain_hits = sum(1 for step in test_case["chain"] if step.lower() in all_text or
                     any(word.lower() in all_text for word in step.split() if len(word) > 4))
    path_completeness = chain_hits / len(test_case["chain"]) if test_case["chain"] else 0

    return {
        "recall_5": round(recall_5, 3),
        "recall_10": round(recall_10, 3),
        "path": round(path_completeness, 3),
    }


def run_suite(config_name: str, features: dict) -> dict:
    """Run all test cases with given feature config."""
    print(f"\n{'='*60}")
    print(f"Config: {config_name}")
    print(f"Features: {features}")
    print(f"{'='*60}")

    # Set features
    resp = set_features(features)
    if resp:
        print(f"  Features set: {resp}")
    else:
        print("  WARNING: Could not set features")

    time.sleep(1)

    results = {}
    total_r5 = total_r10 = total_path = 0

    for tc in TEST_CASES:
        print(f"\n  [{tc['id']}] {tc['query'][:50]}...", end=" ", flush=True)
        t0 = time.time()
        resp = run_query(tc["query"])
        elapsed = time.time() - t0

        if not resp or "results" not in resp:
            print(f"FAILED ({elapsed:.1f}s)")
            results[tc["id"]] = {"recall_5": 0, "recall_10": 0, "path": 0, "latency": elapsed}
            continue

        scores = score_results(resp["results"], tc)
        scores["latency"] = round(elapsed, 1)

        # Check energy method
        path_opt = resp.get("path_optimization", {})
        energy_method = path_opt.get("energy_method", "unknown")
        scores["energy_method"] = energy_method

        total_r5 += scores["recall_5"]
        total_r10 += scores["recall_10"]
        total_path += scores["path"]

        print(f"R@5={scores['recall_5']:.0%} R@10={scores['recall_10']:.0%} Path={scores['path']:.0%} ({elapsed:.1f}s) [{energy_method}]")
        results[tc["id"]] = scores

    n = len(TEST_CASES)
    avg = {
        "avg_recall_5": round(total_r5 / n, 3),
        "avg_recall_10": round(total_r10 / n, 3),
        "avg_path": round(total_path / n, 3),
    }
    print(f"\n  AVERAGES: R@5={avg['avg_recall_5']:.0%}  R@10={avg['avg_recall_10']:.0%}  Path={avg['avg_path']:.0%}")

    return {"per_case": results, "averages": avg}


def main():
    print("=" * 60)
    print("A/B TEST: keyword_overlap vs learned_energy")
    print("=" * 60)

    # Test A: keyword_overlap (learned_energy OFF)
    results_keyword = run_suite("keyword_overlap", {
        "triangle_inequality": True,
        "energy_minimization": True,
        "cooccurrence": True,
        "learned_energy": False,
    })

    # Test B: learned_energy (ON)
    results_learned = run_suite("learned_energy", {
        "triangle_inequality": True,
        "energy_minimization": True,
        "cooccurrence": True,
        "learned_energy": True,
    })

    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON: keyword_overlap vs learned_energy")
    print("=" * 60)
    print(f"{'Case':<8} {'R@5 KW':>8} {'R@5 LE':>8} {'dR@5':>8} {'R@10 KW':>8} {'R@10 LE':>8} {'dR@10':>8} {'Path KW':>8} {'Path LE':>8} {'dPath':>8}")
    print("-" * 80)

    for tc in TEST_CASES:
        tid = tc["id"]
        kw = results_keyword["per_case"].get(tid, {})
        le = results_learned["per_case"].get(tid, {})

        dr5 = le.get("recall_5", 0) - kw.get("recall_5", 0)
        dr10 = le.get("recall_10", 0) - kw.get("recall_10", 0)
        dp = le.get("path", 0) - kw.get("path", 0)

        print(f"{tid:<8} {kw.get('recall_5',0):>7.0%} {le.get('recall_5',0):>7.0%} {dr5:>+7.0%} "
              f"{kw.get('recall_10',0):>7.0%} {le.get('recall_10',0):>7.0%} {dr10:>+7.0%} "
              f"{kw.get('path',0):>7.0%} {le.get('path',0):>7.0%} {dp:>+7.0%}")

    # Averages
    kw_avg = results_keyword["averages"]
    le_avg = results_learned["averages"]
    print("-" * 80)
    dr5 = le_avg["avg_recall_5"] - kw_avg["avg_recall_5"]
    dr10 = le_avg["avg_recall_10"] - kw_avg["avg_recall_10"]
    dp = le_avg["avg_path"] - kw_avg["avg_path"]
    print(f"{'AVG':<8} {kw_avg['avg_recall_5']:>7.0%} {le_avg['avg_recall_5']:>7.0%} {dr5:>+7.0%} "
          f"{kw_avg['avg_recall_10']:>7.0%} {le_avg['avg_recall_10']:>7.0%} {dr10:>+7.0%} "
          f"{kw_avg['avg_path']:>7.0%} {le_avg['avg_path']:>7.0%} {dp:>+7.0%}")

    print("\nDone.")


if __name__ == "__main__":
    main()
