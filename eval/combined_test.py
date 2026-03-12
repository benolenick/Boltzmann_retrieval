#!/usr/bin/env python3
"""
Combined Pipeline Test — Manifold Discovery + Boltzmann Re-ranking

Tests 5 configurations:
  A) FAISS alone (baseline)
  B) Methodic v2.0 (Boltzmann re-ranking of FAISS results)
  C) Manifold alone (geodesic retrieval)
  D) Combined: naive merge of all sources (no intelligence)
  E) Smart Combined: query-type classifier routes to optimal engine(s),
     Manifold discovery + Methodic Boltzmann re-ranking for non-attack-path queries,
     Methodic full pipeline for attack_path queries.

Runs from Windows against jagg services.
"""

import json
import time
from urllib.request import Request, urlopen

from smart_router import smart_search, classify_query
from expanded_test_cases import EXPANDED_TEST_CASES

MEMORIA_URL = "http://192.168.0.224:8000"
METHODIC_URL = "http://192.168.0.224:8002"
MANIFOLD_URL = "http://192.168.0.224:8003"
TIMEOUT = 120

TEST_CASES = EXPANDED_TEST_CASES

ORIGINAL_TEST_CASES = [
    {
        "id": "CB-01", "cat": "constraint_bypass",
        "query": "command injection when spaces are filtered",
        "goal": "achieve command execution without literal spaces",
        "keywords": ["${IFS}", "{cat,/etc/passwd}", "brace expansion", "$IFS$9", "tab", "%09", "command injection"],
        "chain": ["identify space filtering in input", "use ${IFS} or $IFS$9 as space substitute",
                  "construct payload with brace expansion {cmd,arg}", "achieve command execution without literal spaces"],
    },
    {
        "id": "CB-02", "cat": "constraint_bypass",
        "query": "exfiltrate data from blind command injection with no output",
        "goal": "extract data from target via out-of-band channel",
        "keywords": ["out-of-band", "DNS exfiltration", "curl", "wget", "ping", "sleep", "time-based", "blind"],
        "chain": ["confirm blind injection via time delay", "set up OOB listener",
                  "exfiltrate via DNS subdomain encoding", "or exfiltrate via HTTP request to attacker server"],
    },
    {
        "id": "CB-03", "cat": "constraint_bypass",
        "query": "bypass WAF blocking slashes and dots in path traversal",
        "goal": "read sensitive files despite WAF filtering",
        "keywords": ["double encoding", "%252e%252e", "..%c0%af", "path traversal", "null byte", "%00", "overlong UTF-8", "WAF bypass"],
        "chain": ["identify WAF blocking ../ patterns", "attempt URL encoding",
                  "attempt double encoding", "attempt overlong UTF-8 encoding", "read sensitive file"],
    },
    {
        "id": "MH-01", "cat": "multi_hop",
        "query": "Apache Tomcat manager to reverse shell",
        "goal": "get a reverse shell on the server running Tomcat",
        "keywords": ["Tomcat", "manager", "WAR", "msfvenom", "deploy", "reverse shell", "default credentials", "/manager/html"],
        "chain": ["discover Tomcat manager interface", "authenticate with default credentials",
                  "generate malicious WAR file", "deploy WAR via manager upload", "trigger reverse shell"],
    },
    {
        "id": "MH-02", "cat": "multi_hop",
        "query": "SUID binary to root privilege escalation on Linux",
        "goal": "escalate from user to root via SUID binary",
        "keywords": ["SUID", "find / -perm", "GTFOBins", "privilege escalation", "/bin/bash -p", "nmap --interactive", "cp /bin/bash", "chmod +s"],
        "chain": ["enumerate SUID binaries", "identify exploitable SUID binary",
                  "abuse SUID binary to spawn root shell", "verify root access"],
    },
    {
        "id": "MH-03", "cat": "multi_hop",
        "query": "Redis unauthenticated access to SSH key injection to shell",
        "goal": "get SSH shell access via Redis key injection",
        "keywords": ["Redis", "CONFIG SET dir", "CONFIG SET dbfilename", "authorized_keys", "SSH", "redis-cli", ".ssh"],
        "chain": ["connect to unauthenticated Redis", "set dir to .ssh directory",
                  "set dbfilename to authorized_keys", "write SSH public key", "SSH in with key"],
    },
    {
        "id": "CC-01", "cat": "cross_context",
        "query": "living off the land binaries for file download on Windows",
        "goal": "download files to Windows target using built-in binaries",
        "keywords": ["certutil", "bitsadmin", "PowerShell", "Invoke-WebRequest", "LOLBAS", "LOLBin", "download", "transfer"],
        "chain": ["identify need to download file", "use certutil to download",
                  "or use bitsadmin", "or use PowerShell Invoke-WebRequest", "execute downloaded payload"],
    },
    {
        "id": "CC-02", "cat": "cross_context",
        "query": "SQL injection to operating system command execution",
        "goal": "get OS command execution from SQL injection",
        "keywords": ["xp_cmdshell", "INTO OUTFILE", "LOAD_FILE", "COPY TO", "stacked queries", "SQL injection", "os-shell", "sqlmap --os-shell"],
        "chain": ["identify SQL injection", "determine database type",
                  "escalate to OS command execution", "achieve command execution"],
    },
    {
        "id": "AP-01", "cat": "attack_path",
        "query": "Kerberoasting to domain admin in Active Directory",
        "goal": "escalate from domain user to domain admin via Kerberoasting",
        "keywords": ["Kerberoast", "GetUserSPNs", "TGS", "hashcat", "SPN", "service account", "domain admin", "impacket"],
        "chain": ["enumerate SPNs", "request TGS tickets",
                  "extract ticket hashes", "crack TGS hashes offline", "use cracked password to escalate"],
    },
    {
        "id": "AP-02", "cat": "attack_path",
        "query": "man in the middle ARP spoofing to credential capture",
        "goal": "capture credentials by intercepting network traffic via ARP spoofing",
        "keywords": ["ARP spoofing", "arpspoof", "ettercap", "bettercap", "MITM", "credential", "sniff", "IP forwarding"],
        "chain": ["enable IP forwarding", "perform ARP spoofing",
                  "intercept network traffic", "capture credentials from cleartext protocols"],
    },
]


def http_post(url, payload, timeout=TIMEOUT):
    data = json.dumps(payload).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return None


def score(results, keywords, chain):
    """Score results by recall@10 and path completeness."""
    texts = []
    for r in results[:10]:
        if isinstance(r, dict):
            texts.append((r.get("text") or r.get("fact") or r.get("content") or str(r)).lower())
        elif isinstance(r, str):
            texts.append(r.lower())
    blob = "\n".join(texts)

    r10 = sum(1 for kw in keywords if kw.lower() in blob) / len(keywords) if keywords else 0
    covered = 0
    for step in chain:
        words = [w.lower() for w in step.split() if len(w) > 3]
        if not words:
            covered += 1
            continue
        if sum(1 for w in words if w in blob) / len(words) >= 0.5:
            covered += 1
    pc = covered / len(chain) if chain else 0
    return r10, pc


def search_faiss(query, top_k=10):
    """Raw FAISS via Memoria."""
    data = http_post(f"{MEMORIA_URL}/search", {"query": query, "top_k": top_k})
    if data:
        return data.get("results", [])
    return []


def search_methodic(query, top_k=10):
    """Methodic v2.0 (Boltzmann + co-occurrence + triangle)."""
    data = http_post(f"{METHODIC_URL}/smart-search", {"query": query, "top_k": top_k})
    if data:
        return data.get("results", [])
    return []


def search_manifold(query, goal=None, top_k=10):
    """Manifold geodesic retrieval."""
    payload = {"query": query, "top_k": top_k}
    if goal:
        payload["goal"] = goal
    data = http_post(f"{MANIFOLD_URL}/manifold-search", payload)
    if data:
        return data.get("results", [])
    return []


def search_combined(query, goal=None, top_k=10):
    """
    COMBINED PIPELINE:
    1. Get candidates from BOTH manifold AND FAISS
    2. Send merged candidates to Methodic for Boltzmann re-ranking

    This is the key test — does manifold discovery + energy re-ranking
    beat either alone?
    """
    # Get manifold candidates (geodesic + shape-guided)
    manifold_results = search_manifold(query, goal=goal, top_k=top_k * 2)

    # Get FAISS candidates (direct semantic search)
    faiss_results = search_faiss(query, top_k=top_k)

    # Also get Methodic results (which include LLM query decomposition)
    methodic_results = search_methodic(query, top_k=top_k)

    # Merge all candidates, deduplicate
    seen = set()
    merged = []
    for r in manifold_results + methodic_results + faiss_results:
        text = r.get("text") or r.get("fact") or r.get("content") or ""
        key = text[:200].lower().strip()
        if key and key not in seen:
            seen.add(key)
            merged.append(r)

    return merged[:top_k]


def search_smart_combined(query, goal=None, top_k=10):
    """
    SMART COMBINED PIPELINE:
    1. Classify query type (multi_hop, attack_path, constraint_bypass, cross_context)
    2. Route to optimal engine(s) based on classification
    3. For multi_hop/constraint_bypass/cross_context: Manifold discovery + Methodic re-ranking
    4. For attack_path: Methodic full pipeline (LLM decomposition is the key advantage)

    Uses the smart_router module for classification and fusion.
    """
    result = smart_search(query, goal=goal, top_k=top_k)
    return result.get("results", [])


def main():
    print("=" * 100)
    print("COMBINED PIPELINE TEST: FAISS vs Methodic vs Manifold vs Combined vs Smart")
    print("=" * 100)

    # Check all services
    for name, url in [("Memoria/FAISS", MEMORIA_URL), ("Methodic", METHODIC_URL), ("Manifold", MANIFOLD_URL)]:
        try:
            with urlopen(Request(f"{url}/health"), timeout=5) as resp:
                print(f"  {name}: UP")
        except Exception as e:
            print(f"  {name}: DOWN ({e})")

    # Enable all Methodic features
    http_post(f"{METHODIC_URL}/features", {
        "triangle_inequality": True,
        "energy_minimization": True,
        "cooccurrence": True,
    })
    print()

    configs = {
        "FAISS": lambda tc: search_faiss(tc["query"]),
        "Methodic": lambda tc: search_methodic(tc["query"]),
        "Manifold": lambda tc: search_manifold(tc["query"], goal=tc.get("goal")),
        "Combined": lambda tc: search_combined(tc["query"], goal=tc.get("goal")),
        "Smart": lambda tc: search_smart_combined(tc["query"], goal=tc.get("goal")),
    }

    # Run all tests
    results = {name: {"r10": [], "pc": [], "lat": []} for name in configs}

    for i, tc in enumerate(TEST_CASES):
        route = classify_query(tc["query"])
        print(f"[{i+1}/{len(TEST_CASES)}] {tc['id']}: {tc['query'][:50]}... [route: {route}]")
        for name, search_fn in configs.items():
            t0 = time.time()
            res = search_fn(tc)
            lat = (time.time() - t0) * 1000
            r10, pc = score(res, tc["keywords"], tc["chain"])
            results[name]["r10"].append(r10)
            results[name]["pc"].append(pc)
            results[name]["lat"].append(lat)
            print(f"  {name:>10}: R@10={r10:.0%}  Path={pc:.0%}  {lat:.0f}ms")

    # Summary
    print("\n" + "=" * 100)
    print("FINAL RESULTS")
    print("=" * 100)
    header = f"{'Config':<12} {'R@10':>6} {'Path':>6} {'Lat':>7}  {'dR@10':>6} {'dPath':>6}"
    print(header)
    print("-" * len(header))

    base_r10 = sum(results["FAISS"]["r10"]) / len(TEST_CASES)
    base_pc = sum(results["FAISS"]["pc"]) / len(TEST_CASES)

    for name in configs:
        avg_r10 = sum(results[name]["r10"]) / len(TEST_CASES)
        avg_pc = sum(results[name]["pc"]) / len(TEST_CASES)
        avg_lat = sum(results[name]["lat"]) / len(TEST_CASES)
        dr10 = avg_r10 - base_r10
        dpc = avg_pc - base_pc
        sign = lambda v: f"+{v:.0%}" if v > 0 else f"{v:.0%}" if v < 0 else "  0%"
        delta_r10 = sign(dr10) if name != "FAISS" else "  ---"
        delta_pc = sign(dpc) if name != "FAISS" else "  ---"
        print(f"{name:<12} {avg_r10:>5.0%}  {avg_pc:>5.0%}  {avg_lat:>6.0f}  {delta_r10:>6} {delta_pc:>6}")

    print("=" * 100)

    # Per-category breakdown
    categories = {}
    for tc in TEST_CASES:
        cat = tc["cat"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(tc["id"])

    print("\nPER-CATEGORY BREAKDOWN (Path Completeness):")
    print(f"{'Category':<22}", end="")
    for name in configs:
        print(f" {name:>10}", end="")
    print()
    print("-" * 77)

    for cat, ids in categories.items():
        print(f"{cat:<22}", end="")
        for name in configs:
            indices = [i for i, tc in enumerate(TEST_CASES) if tc["id"] in ids]
            avg_pc = sum(results[name]["pc"][i] for i in indices) / len(indices)
            print(f" {avg_pc:>9.0%}", end="")
        print()

    # Save
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {name: {
            "avg_r10": sum(results[name]["r10"]) / len(TEST_CASES),
            "avg_pc": sum(results[name]["pc"]) / len(TEST_CASES),
            "avg_lat": sum(results[name]["lat"]) / len(TEST_CASES),
        } for name in configs},
        "per_query": {name: [
            {"id": tc["id"], "r10": results[name]["r10"][i],
             "pc": results[name]["pc"][i], "lat": results[name]["lat"][i]}
            for i, tc in enumerate(TEST_CASES)
        ] for name in configs},
        "smart_routing": {tc["id"]: classify_query(tc["query"]) for tc in TEST_CASES},
    }
    with open("combined_test_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to combined_test_results.json")


if __name__ == "__main__":
    main()
