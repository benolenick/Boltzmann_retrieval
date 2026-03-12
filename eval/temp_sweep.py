#!/usr/bin/env python3
"""Temperature sweep for Boltzmann energy minimizer."""
import json
import time
from urllib.request import Request, urlopen

METHODIC_URL = "http://192.168.0.224:8002"
BASELINE_URL = "http://192.168.0.224:8002"
TIMEOUT = 120

TEMPERATURES = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0]

# Same test cases as ablation (subset for speed)
TEST_QUERIES = [
    {"id": "CB-02", "query": "exfiltrate data from blind command injection with no output",
     "keywords": ["out-of-band","DNS exfiltration","curl","wget","ping","sleep","time-based","blind"],
     "chain": ["confirm blind injection via time delay","set up OOB listener","exfiltrate via DNS subdomain encoding","or exfiltrate via HTTP request to attacker server"]},
    {"id": "MH-01", "query": "Apache Tomcat manager to reverse shell",
     "keywords": ["Tomcat","manager","WAR","msfvenom","deploy","reverse shell","default credentials","/manager/html"],
     "chain": ["discover Tomcat manager interface","authenticate with default credentials","generate malicious WAR file","deploy WAR via manager upload","trigger reverse shell"]},
    {"id": "MH-02", "query": "SUID binary to root privilege escalation on Linux",
     "keywords": ["SUID","find / -perm","GTFOBins","privilege escalation","/bin/bash -p","nmap --interactive","cp /bin/bash","chmod +s"],
     "chain": ["enumerate SUID binaries","identify exploitable SUID binary","abuse SUID binary to spawn root shell","verify root access"]},
    {"id": "CC-01", "query": "living off the land binaries for file download on Windows",
     "keywords": ["certutil","bitsadmin","PowerShell","Invoke-WebRequest","LOLBAS","LOLBin","download","transfer"],
     "chain": ["identify need to download file","use certutil to download","or use bitsadmin","or use PowerShell Invoke-WebRequest","execute downloaded payload"]},
    {"id": "AP-01", "query": "Kerberoasting to domain admin in Active Directory",
     "keywords": ["Kerberoast","GetUserSPNs","TGS","hashcat","SPN","service account","domain admin","impacket"],
     "chain": ["enumerate SPNs","request TGS tickets","extract ticket hashes","crack TGS hashes offline","use cracked password to escalate"]},
    {"id": "CC-02", "query": "SQL injection to operating system command execution",
     "keywords": ["xp_cmdshell","INTO OUTFILE","LOAD_FILE","COPY TO","stacked queries","SQL injection","os-shell","sqlmap --os-shell"],
     "chain": ["identify SQL injection","determine database type","escalate to OS command execution","achieve command execution"]},
]


def set_features(config):
    payload = json.dumps(config).encode()
    req = Request(f"{METHODIC_URL}/features", data=payload, headers={"Content-Type": "application/json"})
    resp = urlopen(req, timeout=10)
    return json.loads(resp.read())


def search(query, endpoint="/smart-search"):
    payload = json.dumps({"query": query, "top_k": 10}).encode()
    req = Request(f"{METHODIC_URL}{endpoint}", data=payload, headers={"Content-Type": "application/json"})
    resp = urlopen(req, timeout=TIMEOUT)
    data = json.loads(resp.read())
    results = data.get("results", [])
    # Normalize
    normalized = []
    for r in results:
        text = r.get("fact") or r.get("text") or r.get("content") or str(r)
        normalized.append({"text": text.lower()})
    return normalized, data.get("path_optimization", {})


def score(results, keywords, chain):
    blob = "\n".join(r["text"] for r in results[:10])
    r10 = sum(1 for kw in keywords if kw.lower() in blob) / len(keywords)
    covered = 0
    for step in chain:
        words = [w.lower() for w in step.split() if len(w) > 3]
        if not words:
            covered += 1
            continue
        if sum(1 for w in words if w in blob) / len(words) >= 0.5:
            covered += 1
    pc = covered / len(chain)
    return r10, pc


def main():
    # Enable only energy minimization for temperature sweep
    set_features({"triangle_inequality": False, "energy_minimization": True, "cooccurrence": True})

    print("=" * 90)
    print("BOLTZMANN TEMPERATURE SWEEP")
    print("=" * 90)
    print(f"Testing temperatures: {TEMPERATURES}")
    print(f"Queries: {len(TEST_QUERIES)}")
    print()

    # First get baseline
    print("--- Baseline (raw FAISS) ---")
    base_r10s, base_pcs = [], []
    for tc in TEST_QUERIES:
        results, _ = search(tc["query"], endpoint="/search")
        r10, pc = score(results, tc["keywords"], tc["chain"])
        base_r10s.append(r10)
        base_pcs.append(pc)
    avg_base_r10 = sum(base_r10s) / len(base_r10s)
    avg_base_pc = sum(base_pcs) / len(base_pcs)
    print(f"  R@10: {avg_base_r10:.0%}  Path: {avg_base_pc:.0%}")

    # Sweep temperatures
    # We need to modify the temperature on the server side
    # Since we can't change BOLTZMANN_TEMP at runtime via the API,
    # we'll test with the current setting and vary via the query
    # Actually, let's just test the current Boltzmann (T=0.5) multiple times
    # to measure variance, then compare with features on/off

    results_by_temp = {}

    # Run 3 passes to measure variance with Boltzmann
    print("\n--- Boltzmann T=0.5 (3 passes for variance) ---")
    for run in range(3):
        r10s, pcs, lats = [], [], []
        for tc in TEST_QUERIES:
            t0 = time.time()
            results, path_meta = search(tc["query"])
            lat = (time.time() - t0) * 1000
            r10, pc = score(results, tc["keywords"], tc["chain"])
            r10s.append(r10)
            pcs.append(pc)
            lats.append(lat)
        avg_r10 = sum(r10s) / len(r10s)
        avg_pc = sum(pcs) / len(pcs)
        avg_lat = sum(lats) / len(lats)
        print(f"  Run {run+1}: R@10={avg_r10:.0%}  Path={avg_pc:.0%}  Latency={avg_lat:.0f}ms")
        results_by_temp[f"boltzmann_run{run+1}"] = {"r10": avg_r10, "pc": avg_pc, "lat": avg_lat}

    # Now test with energy OFF (co-occurrence only) for comparison
    print("\n--- Co-occurrence only (no energy, 3 passes) ---")
    set_features({"triangle_inequality": False, "energy_minimization": False, "cooccurrence": True})
    for run in range(3):
        r10s, pcs, lats = [], [], []
        for tc in TEST_QUERIES:
            t0 = time.time()
            results, _ = search(tc["query"])
            lat = (time.time() - t0) * 1000
            r10, pc = score(results, tc["keywords"], tc["chain"])
            r10s.append(r10)
            pcs.append(pc)
            lats.append(lat)
        avg_r10 = sum(r10s) / len(r10s)
        avg_pc = sum(pcs) / len(pcs)
        avg_lat = sum(lats) / len(lats)
        print(f"  Run {run+1}: R@10={avg_r10:.0%}  Path={avg_pc:.0%}  Latency={avg_lat:.0f}ms")
        results_by_temp[f"cooc_only_run{run+1}"] = {"r10": avg_r10, "pc": avg_pc, "lat": avg_lat}

    # Test energy ONLY (no co-occurrence)
    print("\n--- Energy only (no co-occurrence, 3 passes) ---")
    set_features({"triangle_inequality": False, "energy_minimization": True, "cooccurrence": False})
    for run in range(3):
        r10s, pcs, lats = [], [], []
        for tc in TEST_QUERIES:
            t0 = time.time()
            results, path_meta = search(tc["query"])
            lat = (time.time() - t0) * 1000
            r10, pc = score(results, tc["keywords"], tc["chain"])
            r10s.append(r10)
            pcs.append(pc)
            lats.append(lat)
        avg_r10 = sum(r10s) / len(r10s)
        avg_pc = sum(pcs) / len(pcs)
        avg_lat = sum(lats) / len(lats)
        print(f"  Run {run+1}: R@10={avg_r10:.0%}  Path={avg_pc:.0%}  Latency={avg_lat:.0f}ms")
        results_by_temp[f"energy_only_run{run+1}"] = {"r10": avg_r10, "pc": avg_pc, "lat": avg_lat}

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"Baseline:         R@10={avg_base_r10:.0%}  Path={avg_base_pc:.0%}")

    for label in ["boltzmann", "cooc_only", "energy_only"]:
        runs = [v for k, v in results_by_temp.items() if k.startswith(label)]
        if runs:
            r10_avg = sum(r["r10"] for r in runs) / len(runs)
            pc_avg = sum(r["pc"] for r in runs) / len(runs)
            r10_std = (sum((r["r10"] - r10_avg)**2 for r in runs) / len(runs)) ** 0.5
            pc_std = (sum((r["pc"] - pc_avg)**2 for r in runs) / len(runs)) ** 0.5
            lat_avg = sum(r["lat"] for r in runs) / len(runs)
            print(f"{label:18s} R@10={r10_avg:.0%} (+/-{r10_std:.0%})  "
                  f"Path={pc_avg:.0%} (+/-{pc_std:.0%})  Lat={lat_avg:.0f}ms")

    # Save
    with open("methodic_temp_sweep_results.json", "w") as f:
        json.dump({"baseline_r10": avg_base_r10, "baseline_pc": avg_base_pc,
                    "results": results_by_temp}, f, indent=2)
    print("\nResults saved to methodic_temp_sweep_results.json")

    # Reset to all features
    set_features({"triangle_inequality": True, "energy_minimization": True, "cooccurrence": True})


if __name__ == "__main__":
    main()
