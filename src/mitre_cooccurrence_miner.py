"""
MITRE ATT&CK Co-occurrence Mining Pipeline
-------------------------------------------
Extracts technique co-occurrence pairs from MITRE ATT&CK data,
maps them to Memoria facts, and seeds the methodic.db co_occurrences table.
"""

import json
import hashlib
import time
import sys
import sqlite3
import tempfile
import os
from collections import defaultdict
from itertools import combinations

import requests
import paramiko

# ─── Configuration ───────────────────────────────────────────────────────────
ATTACK_JSON = r"C:\Users\om\Desktop\enterprise-attack.json"
MEMORIA_URL = "http://192.168.0.224:8000/search"
MEMORIA_TOP_K = 5
MEMORIA_DELAY = 0.15  # seconds between Memoria requests

JAGG_HOST = "192.168.0.224"
JAGG_USER = "om"
JAGG_PASS = "aintnosunshinewhenshesgone"
REMOTE_DB = "/home/om/htb-autopwn/methodic.db"

SUMMARY_FILE = r"C:\Users\om\Desktop\mitre_cooccurrence_mining.md"

# Kill chain phase ordering for adjacency detection
KILL_CHAIN_ORDER = [
    "reconnaissance",
    "resource-development",
    "initial-access",
    "execution",
    "persistence",
    "privilege-escalation",
    "defense-evasion",
    "credential-access",
    "discovery",
    "lateral-movement",
    "collection",
    "command-and-control",
    "exfiltration",
    "impact",
]


def md5_hash(text):
    """MD5 hash of first 200 chars, lowercased."""
    return hashlib.md5(text[:200].lower().encode()).hexdigest()


def load_attack_data(path):
    """Load and index ATT&CK JSON."""
    print(f"[*] Loading ATT&CK data from {path}...")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    techniques = {}  # stix_id -> {name, ext_id, phases, description}
    relationships = []  # list of (source_ref, target_ref, rel_type)
    intrusion_sets = {}  # stix_id -> {name, ...}
    id_to_obj = {}

    for obj in data["objects"]:
        stix_id = obj.get("id", "")
        id_to_obj[stix_id] = obj

        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue

        if obj["type"] == "attack-pattern":
            ext_refs = obj.get("external_references", [])
            ext_id = ext_refs[0].get("external_id", "") if ext_refs else ""
            phases = [p["phase_name"] for p in obj.get("kill_chain_phases", [])]
            techniques[stix_id] = {
                "name": obj["name"],
                "ext_id": ext_id,
                "phases": phases,
                "description": obj.get("description", ""),
            }

        elif obj["type"] == "relationship":
            relationships.append(
                (obj["source_ref"], obj["target_ref"], obj.get("relationship_type", ""))
            )

        elif obj["type"] == "intrusion-set":
            intrusion_sets[stix_id] = {"name": obj["name"]}

    print(f"    {len(techniques)} active techniques")
    print(f"    {len(relationships)} relationships")
    print(f"    {len(intrusion_sets)} intrusion sets")
    return techniques, relationships, intrusion_sets, id_to_obj


def extract_technique_pairs(techniques, relationships, intrusion_sets, id_to_obj):
    """
    Build co-occurring technique pairs from three sources:
    1. Group profiles (techniques used by the same group)
    2. Kill chain adjacency (techniques in adjacent phases)
    3. Shared tool/malware usage (techniques linked to same malware/tool)
    """
    pair_counts = defaultdict(int)  # (tech_a_stix, tech_b_stix) -> count
    tech_stix_ids = set(techniques.keys())

    # ─── Source 1: Group profiles ────────────────────────────────────────────
    print("[*] Extracting group-based co-occurrences...")
    group_techniques = defaultdict(set)  # group_stix_id -> set of technique stix_ids
    for src, tgt, rel_type in relationships:
        if rel_type == "uses":
            # intrusion-set uses attack-pattern
            if src in intrusion_sets and tgt in tech_stix_ids:
                group_techniques[src].add(tgt)

    group_pairs = 0
    for group_id, techs in group_techniques.items():
        techs_list = sorted(techs)
        for a, b in combinations(techs_list, 2):
            pair_counts[(a, b)] += 1
            group_pairs += 1

    print(f"    {len(group_techniques)} groups with techniques, {group_pairs} pairs")

    # ─── Source 2: Kill chain adjacency ──────────────────────────────────────
    print("[*] Extracting kill-chain adjacency co-occurrences...")
    phase_to_techs = defaultdict(set)
    for stix_id, info in techniques.items():
        for phase in info["phases"]:
            phase_to_techs[phase].add(stix_id)

    kc_pairs = 0
    for i in range(len(KILL_CHAIN_ORDER) - 1):
        phase_a = KILL_CHAIN_ORDER[i]
        phase_b = KILL_CHAIN_ORDER[i + 1]
        techs_a = phase_to_techs.get(phase_a, set())
        techs_b = phase_to_techs.get(phase_b, set())
        for a in techs_a:
            for b in techs_b:
                key = tuple(sorted([a, b]))
                pair_counts[key] += 1
                kc_pairs += 1

    print(f"    {kc_pairs} kill-chain adjacency pairs")

    # ─── Source 3: Shared malware/tool usage ─────────────────────────────────
    print("[*] Extracting shared malware/tool co-occurrences...")
    tool_techniques = defaultdict(set)  # tool/malware stix_id -> set of technique stix_ids
    for src, tgt, rel_type in relationships:
        if rel_type == "uses":
            src_obj = id_to_obj.get(src, {})
            if src_obj.get("type") in ("malware", "tool") and tgt in tech_stix_ids:
                tool_techniques[src].add(tgt)

    tool_pairs = 0
    for tool_id, techs in tool_techniques.items():
        techs_list = sorted(techs)
        for a, b in combinations(techs_list, 2):
            pair_counts[(a, b)] += 1
            tool_pairs += 1

    print(f"    {len(tool_techniques)} tools/malware with techniques, {tool_pairs} pairs")

    # Deduplicate: ensure a < b ordering
    final_pairs = {}
    for (a, b), count in pair_counts.items():
        key = tuple(sorted([a, b]))
        final_pairs[key] = final_pairs.get(key, 0) + count

    print(f"    {len(final_pairs)} unique technique pairs total")
    return final_pairs


def search_memoria(query, top_k=MEMORIA_TOP_K):
    """Search Memoria for matching facts."""
    try:
        resp = requests.post(
            MEMORIA_URL,
            json={"query": query, "top_k": top_k},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        # Filter to reasonable relevance
        return [r for r in results if r.get("relevance", 0) >= 0.65]
    except Exception as e:
        return []


def map_techniques_to_facts(techniques, pair_counts):
    """
    For each technique that appears in a pair, search Memoria for matching facts.
    Returns tech_stix_id -> list of fact dicts.
    """
    # Collect all technique stix_ids that appear in pairs
    needed_techs = set()
    for a, b in pair_counts.keys():
        needed_techs.add(a)
        needed_techs.add(b)

    print(f"[*] Mapping {len(needed_techs)} techniques to Memoria facts...")

    tech_facts = {}  # stix_id -> [{"text": ..., "relevance": ...}, ...]
    searched = 0
    cached_miss = 0

    for stix_id in sorted(needed_techs):
        info = techniques.get(stix_id)
        if not info:
            continue

        # Build search query from technique name + key terms
        query = info["name"]
        # Add first sentence of description for context
        desc = info.get("description", "")
        first_sentence = desc.split(".")[0] if desc else ""
        if first_sentence and len(first_sentence) < 150:
            query = f"{query}: {first_sentence}"

        facts = search_memoria(query)
        if facts:
            tech_facts[stix_id] = facts

        searched += 1
        if searched % 50 == 0:
            print(f"    Searched {searched}/{len(needed_techs)} techniques, {len(tech_facts)} with facts")

        time.sleep(MEMORIA_DELAY)

    print(f"    Done: {len(tech_facts)}/{len(needed_techs)} techniques matched to facts")
    return tech_facts


def build_fact_cooccurrences(pair_counts, tech_facts, techniques):
    """
    For each technique pair where both have Memoria facts,
    create fact-level co-occurrence records.
    """
    print("[*] Building fact-level co-occurrences...")
    cooccurrences = []  # list of (fact_a_hash, fact_b_hash, fact_a_text, fact_b_text, count)

    # Deduplicate at the fact pair level
    fact_pair_counts = defaultdict(lambda: {"count": 0, "a_text": "", "b_text": ""})

    pairs_with_facts = 0
    for (tech_a, tech_b), tech_count in pair_counts.items():
        facts_a = tech_facts.get(tech_a, [])
        facts_b = tech_facts.get(tech_b, [])
        if not facts_a or not facts_b:
            continue

        pairs_with_facts += 1

        # Take top facts for each technique (limit to top 3 to avoid explosion)
        for fa in facts_a[:3]:
            for fb in facts_b[:3]:
                if fa["text"] == fb["text"]:
                    continue
                ha = md5_hash(fa["text"])
                hb = md5_hash(fb["text"])
                # Canonical ordering
                if ha > hb:
                    ha, hb = hb, ha
                    fa, fb = fb, fa
                key = (ha, hb)
                fact_pair_counts[key]["count"] += tech_count
                fact_pair_counts[key]["a_text"] = fa["text"]
                fact_pair_counts[key]["b_text"] = fb["text"]

    for (ha, hb), info in fact_pair_counts.items():
        cooccurrences.append((ha, hb, info["a_text"], info["b_text"], info["count"]))

    print(f"    {pairs_with_facts} technique pairs had facts on both sides")
    print(f"    {len(cooccurrences)} unique fact co-occurrence pairs generated")
    return cooccurrences


def seed_database(cooccurrences):
    """SSH to jagg and insert co-occurrences into methodic.db."""
    print(f"[*] Connecting to jagg to seed {len(cooccurrences)} co-occurrences...")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        JAGG_HOST,
        username=JAGG_USER,
        password=JAGG_PASS,
        allow_agent=False,
        look_for_keys=False,
    )

    # First, download the DB, modify locally, then upload back
    # This is more reliable than running sqlite3 commands over SSH
    sftp = ssh.open_sftp()

    local_db = os.path.join(tempfile.gettempdir(), "methodic_temp.db")
    print(f"    Downloading {REMOTE_DB} -> {local_db}")
    sftp.get(REMOTE_DB, local_db)

    # Check existing state
    conn = sqlite3.connect(local_db)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM co_occurrences")
    before_count = cur.fetchone()[0]
    print(f"    Existing co-occurrences: {before_count}")

    # Insert/update
    inserted = 0
    updated = 0
    for fa_hash, fb_hash, fa_text, fb_text, count in cooccurrences:
        # Try insert
        cur.execute(
            "SELECT count FROM co_occurrences WHERE fact_a_hash = ? AND fact_b_hash = ?",
            (fa_hash, fb_hash),
        )
        row = cur.fetchone()
        if row is None:
            cur.execute(
                """INSERT INTO co_occurrences (fact_a_hash, fact_b_hash, fact_a_text, fact_b_text, count, last_session)
                   VALUES (?, ?, ?, ?, ?, 'mitre_attack_seed')""",
                (fa_hash, fb_hash, fa_text, fb_text, count),
            )
            inserted += 1
        else:
            cur.execute(
                "UPDATE co_occurrences SET count = count + ?, last_session = 'mitre_attack_seed' WHERE fact_a_hash = ? AND fact_b_hash = ?",
                (count, fa_hash, fb_hash),
            )
            updated += 1

    conn.commit()

    cur.execute("SELECT COUNT(*) FROM co_occurrences")
    after_count = cur.fetchone()[0]

    # Get top co-occurrences from the mitre seed
    cur.execute(
        """SELECT fact_a_text, fact_b_text, count FROM co_occurrences
           WHERE last_session = 'mitre_attack_seed'
           ORDER BY count DESC LIMIT 20"""
    )
    top_pairs = cur.fetchall()

    conn.close()

    # Upload back
    print(f"    Uploading modified DB back to {REMOTE_DB}")
    sftp.put(local_db, REMOTE_DB)
    sftp.close()
    ssh.close()

    # Cleanup
    os.remove(local_db)

    print(f"    Inserted: {inserted}, Updated: {updated}")
    print(f"    Before: {before_count}, After: {after_count}")

    return {
        "before": before_count,
        "after": after_count,
        "inserted": inserted,
        "updated": updated,
        "top_pairs": top_pairs,
    }


def write_summary(stats, summary_file):
    """Write mining summary to markdown file."""
    lines = [
        "# MITRE ATT&CK Co-occurrence Mining Results",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Statistics",
        "",
        f"- **ATT&CK techniques processed:** {stats['techniques_count']}",
        f"- **Technique pairs found:** {stats['technique_pairs']}",
        f"- **Techniques with Memoria matches:** {stats['techniques_with_facts']}",
        f"- **Fact co-occurrence pairs generated:** {stats['fact_pairs']}",
        f"- **Co-occurrences inserted (new):** {stats['db_inserted']}",
        f"- **Co-occurrences updated (existing):** {stats['db_updated']}",
        f"- **DB rows before:** {stats['db_before']}",
        f"- **DB rows after:** {stats['db_after']}",
        "",
        "## Sources of Co-occurrence",
        "",
        f"- **Group profiles:** Techniques used by the same threat group ({stats['group_count']} groups)",
        f"- **Kill chain adjacency:** Techniques in adjacent ATT&CK phases",
        f"- **Shared tools/malware:** Techniques used by the same tool or malware family",
        "",
        "## Top Co-occurrences (by count)",
        "",
        "| # | Fact A | Fact B | Count |",
        "|---|--------|--------|-------|",
    ]

    for i, (fa, fb, count) in enumerate(stats["top_pairs"][:20], 1):
        fa_short = fa[:80].replace("|", "/")
        fb_short = fb[:80].replace("|", "/")
        lines.append(f"| {i} | {fa_short} | {fb_short} | {count} |")

    lines.append("")

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[*] Summary written to {summary_file}")


def main():
    start_time = time.time()

    # Step 1: Load ATT&CK data
    techniques, relationships, intrusion_sets, id_to_obj = load_attack_data(ATTACK_JSON)

    # Step 2: Extract technique pairs
    pair_counts = extract_technique_pairs(techniques, relationships, intrusion_sets, id_to_obj)

    # Count groups for stats
    group_techniques = defaultdict(set)
    tech_stix_ids = set(techniques.keys())
    for src, tgt, rel_type in relationships:
        if rel_type == "uses" and src in intrusion_sets and tgt in tech_stix_ids:
            group_techniques[src].add(tgt)

    # Step 3: Map techniques to Memoria facts
    tech_facts = map_techniques_to_facts(techniques, pair_counts)

    # Step 4: Build fact-level co-occurrences
    cooccurrences = build_fact_cooccurrences(pair_counts, tech_facts, techniques)

    if not cooccurrences:
        print("[!] No co-occurrences to seed. Exiting.")
        return

    # Step 5: Seed the database
    db_stats = seed_database(cooccurrences)

    # Step 6: Write summary
    stats = {
        "techniques_count": len(techniques),
        "technique_pairs": len(pair_counts),
        "techniques_with_facts": len(tech_facts),
        "fact_pairs": len(cooccurrences),
        "db_inserted": db_stats["inserted"],
        "db_updated": db_stats["updated"],
        "db_before": db_stats["before"],
        "db_after": db_stats["after"],
        "top_pairs": db_stats["top_pairs"],
        "group_count": len(group_techniques),
    }

    write_summary(stats, SUMMARY_FILE)

    elapsed = time.time() - start_time
    print(f"\n[*] Pipeline complete in {elapsed:.1f}s")
    print(f"    Techniques: {stats['techniques_count']}")
    print(f"    Technique pairs: {stats['technique_pairs']}")
    print(f"    Memoria matches: {stats['techniques_with_facts']}")
    print(f"    Fact co-occurrences: {stats['fact_pairs']}")
    print(f"    DB inserted: {stats['db_inserted']}, updated: {stats['db_updated']}")
    print(f"    DB total: {stats['db_before']} -> {stats['db_after']}")


if __name__ == "__main__":
    main()
