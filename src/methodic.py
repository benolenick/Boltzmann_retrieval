#!/usr/bin/env python3
"""
Methodic Retrieval Engine — protein-folding-inspired cybersecurity knowledge retrieval.

Layers:
  1. Query Reasoner (qwen3.5 LLM) — classifies query, generates expert search intents
  2. Multi-Strategy Retrieval — FAISS/Memoria, Exploit Index, Relationship Graph, Co-occurrence
  3. Path Assembler — deduplication, scoring, ranking

Listens on port 8002.  Backend: Memoria (:8000), Ollama/qwen3.5 (:11434).
Database: SQLite at /home/om/htb-autopwn/methodic.db
"""

import hashlib
import json
import logging
import math
import os
import pickle
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import urlopen, Request
from urllib.error import URLError

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PORT = 8002
MEMORIA_URL = "http://localhost:8000"
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3.5:latest"
DB_PATH = "/home/om/htb-autopwn/methodic.db"
LLM_TIMEOUT = 30
RETRIEVAL_WORKERS = 8

# Feature flags — enable/disable protein-folding-inspired methods
ENABLE_TRIANGLE_INEQUALITY = True   # Transitive closure over edge graph
ENABLE_ENERGY_MINIMIZATION = True   # Path optimizer (energy landscape)
ENABLE_COOCCURRENCE = True          # Co-occurrence matrix boosting
ENABLE_LEARNED_ENERGY = True        # Learned connection classifier (replaces keyword overlap)

# Learned connection model path
CONNECTION_MODEL_PATH = "/home/om/htb-autopwn/connection_model.pkl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("methodic")

# ---------------------------------------------------------------------------
# spaCy — loaded once
# ---------------------------------------------------------------------------
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_OK = True
    log.info("spaCy en_core_web_sm loaded")
except Exception as e:
    nlp = None
    SPACY_OK = False
    log.warning(f"spaCy unavailable: {e}")

# ---------------------------------------------------------------------------
# Learned connection model — loaded once at startup
# ---------------------------------------------------------------------------
_connection_model = None
_st_model = None

def _load_connection_model():
    """Load the trained connection classifier and sentence-transformers model."""
    global _connection_model, _st_model
    if not ENABLE_LEARNED_ENERGY:
        log.info("Learned energy function disabled (ENABLE_LEARNED_ENERGY=False)")
        return

    if not os.path.exists(CONNECTION_MODEL_PATH):
        log.warning(f"Connection model not found at {CONNECTION_MODEL_PATH} — falling back to keyword overlap")
        return

    try:
        import numpy as np
        with open(CONNECTION_MODEL_PATH, "rb") as f:
            _connection_model = pickle.load(f)
        log.info(
            f"Loaded connection model v{_connection_model.get('version', '?')} "
            f"({_connection_model.get('training_samples', '?')} training samples, "
            f"feature_dim={_connection_model.get('feature_dim', '?')})"
        )

        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
        log.info("Loaded sentence-transformers model (all-MiniLM-L6-v2) for learned energy")
    except Exception as e:
        log.warning(f"Failed to load connection model: {e} — falling back to keyword overlap")
        _connection_model = None
        _st_model = None

_load_connection_model()


def _learned_connection_strength(text_a: str, text_b: str) -> float:
    """
    Predict connection strength between two facts using the trained classifier.

    Returns probability in [0, 1] that fact_b logically follows fact_a in an attack chain.
    Falls back to keyword overlap if model unavailable.
    """
    if _connection_model is None or _st_model is None:
        return None  # Caller should fall back to keyword overlap

    try:
        import numpy as np
        emb_a = _st_model.encode(text_a[:500], normalize_embeddings=True)
        emb_b = _st_model.encode(text_b[:500], normalize_embeddings=True)

        product = emb_a * emb_b
        diff = np.abs(emb_a - emb_b)
        cosine = np.dot(emb_a, emb_b)

        feat = np.concatenate([emb_a, emb_b, product, diff, [cosine]]).reshape(1, -1)
        prob = _connection_model["pipeline"].predict_proba(feat)[0][1]
        return float(prob)
    except Exception as e:
        log.warning(f"Learned connection scoring failed: {e}")
        return None


def _batch_learned_connection(texts: list[str]) -> list[list[float]] | None:
    """
    Compute pairwise connection strengths for a batch of texts using the learned model.

    Returns n×n matrix of probabilities, or None if model unavailable.
    Much faster than calling _learned_connection_strength() in a loop because
    we encode all texts at once.
    """
    if _connection_model is None or _st_model is None:
        return None

    try:
        import numpy as np
        n = len(texts)
        truncated = [t[:500] for t in texts]

        # Batch encode all texts at once
        embeddings = _st_model.encode(truncated, normalize_embeddings=True, batch_size=64)

        # Build feature matrix for all pairs
        scores = [[0.0] * n for _ in range(n)]
        pipeline = _connection_model["pipeline"]

        # Build all pair features at once for batch prediction
        pair_indices = []
        pair_features = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                emb_a = embeddings[i]
                emb_b = embeddings[j]
                product = emb_a * emb_b
                diff = np.abs(emb_a - emb_b)
                cosine = np.dot(emb_a, emb_b)
                feat = np.concatenate([emb_a, emb_b, product, diff, [cosine]])
                pair_features.append(feat)
                pair_indices.append((i, j))

        if not pair_features:
            return scores

        X = np.array(pair_features, dtype=np.float32)
        probs = pipeline.predict_proba(X)[:, 1]

        for (i, j), prob in zip(pair_indices, probs):
            scores[i][j] = float(prob)

        return scores
    except Exception as e:
        log.warning(f"Batch learned connection scoring failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Regex tag patterns (CVE, service names, techniques, OS, phases)
# ---------------------------------------------------------------------------
TAG_PATTERNS = {
    "cve": re.compile(r"CVE-\d{4}-\d{4,}", re.IGNORECASE),
    "service": re.compile(
        r"\b(apache|nginx|tomcat|iis|docker|redis|mysql|mssql|postgres|"
        r"ssh|ftp|smb|ldap|kerberos|winrm|rdp|snmp|dns|nfs|"
        r"wordpress|drupal|joomla|jenkins|gitlab|grafana|"
        r"cacti|wing-?ftp|needrestart|vsftpd|proftpd|"
        r"elasticsearch|kibana|mongo|rabbitmq|memcached)\b",
        re.IGNORECASE,
    ),
    "technique": re.compile(
        r"\b(sqli|sql injection|xss|csrf|ssrf|rce|lfi|rfi|"
        r"command injection|path traversal|directory traversal|"
        r"buffer overflow|heap overflow|use.after.free|"
        r"privilege escalation|privesc|lateral movement|"
        r"kerberoasting|pass.the.hash|golden ticket|"
        r"deserialization|xxe|ssti|idor|open redirect|"
        r"file upload|webshell|reverse shell|bind shell)\b",
        re.IGNORECASE,
    ),
    "os": re.compile(
        r"\b(linux|windows|macos|freebsd|openbsd|android|ios)\b",
        re.IGNORECASE,
    ),
    "phase": re.compile(
        r"\b(recon|enumeration|scanning|exploitation|post.exploitation|"
        r"persistence|exfiltration|lateral.movement|initial.access|"
        r"credential.access|defense.evasion)\b",
        re.IGNORECASE,
    ),
}

# ---------------------------------------------------------------------------
# Database init
# ---------------------------------------------------------------------------
_db_lock = threading.Lock()

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS fact_edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_text TEXT NOT NULL,
            target_text TEXT NOT NULL,
            relation TEXT NOT NULL,
            confidence REAL DEFAULT 0.5,
            context TEXT DEFAULT '',
            created_at REAL DEFAULT (unixepoch()),
            UNIQUE(source_text, target_text, relation)
        );
        CREATE TABLE IF NOT EXISTS co_occurrences (
            fact_a_hash TEXT NOT NULL,
            fact_b_hash TEXT NOT NULL,
            fact_a_text TEXT NOT NULL,
            fact_b_text TEXT NOT NULL,
            count INTEGER DEFAULT 1,
            last_session TEXT DEFAULT '',
            PRIMARY KEY (fact_a_hash, fact_b_hash)
        );
        CREATE TABLE IF NOT EXISTS exploits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            service TEXT NOT NULL,
            version TEXT DEFAULT '',
            cve TEXT DEFAULT '',
            technique TEXT NOT NULL,
            auth_required TEXT DEFAULT 'unknown',
            payload TEXT NOT NULL,
            gotchas TEXT DEFAULT '',
            post_exploitation TEXT DEFAULT '',
            created_at REAL DEFAULT (unixepoch())
        );
        CREATE TABLE IF NOT EXISTS query_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            query_type TEXT DEFAULT '',
            intents_json TEXT DEFAULT '[]',
            results_count INTEGER DEFAULT 0,
            timestamp REAL DEFAULT (unixepoch())
        );
        CREATE INDEX IF NOT EXISTS idx_edges_source ON fact_edges(source_text);
        CREATE INDEX IF NOT EXISTS idx_edges_target ON fact_edges(target_text);
        CREATE INDEX IF NOT EXISTS idx_exploits_service ON exploits(service);
        CREATE INDEX IF NOT EXISTS idx_exploits_cve ON exploits(cve);
    """)
    conn.commit()
    conn.close()

# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------
SEED_EXPLOITS = [
    {
        "service": "cacti", "version": "1.2.28", "cve": "CVE-2025-24367",
        "technique": "rrdtool newline injection", "auth_required": "low-priv",
        "payload": "newline in right_axis_label",
        "gotchas": "no spaces in label",
        "post_exploitation": "PHP webshell",
    },
    {
        "service": "docker-desktop", "version": "*", "cve": "CVE-2025-9074",
        "technique": "Docker API escape", "auth_required": "no (from container)",
        "payload": "192.168.65.7:2375 create privileged container",
        "gotchas": r"C:\ at /mnt/host/c/ in WSL2",
        "post_exploitation": "host filesystem",
    },
    {
        "service": "wing-ftp", "version": "*", "cve": "CVE-2025-47812",
        "technique": "Lua RCE", "auth_required": "admin",
        "payload": "os.execute() in scripting",
        "gotchas": "",
        "post_exploitation": "SYSTEM/root",
    },
    {
        "service": "needrestart", "version": "<3.8", "cve": "CVE-2024-48990",
        "technique": "PYTHONPATH injection", "auth_required": "local user",
        "payload": "set PYTHONPATH + trigger needrestart",
        "gotchas": "",
        "post_exploitation": "root",
    },
]

SEED_EDGES = [
    ("rrdtool splits arguments on spaces", "no spaces in payload", "blocks", 0.9, "cacti exploit constraint"),
    ("no spaces in payload", "${IFS} replaces spaces in bash", "bypassed_by", 0.95, "classic space bypass"),
    ("${IFS} replaces spaces in bash", "brace expansion {cmd,arg} avoids spaces", "related", 0.8, "alternative space bypass"),
    ("Cacti CVE-2025-24367", "rrdtool newline injection", "enables", 0.95, "exploit chain"),
    ("rrdtool newline injection", "PHP webshell via CSV output", "enables", 0.9, "post-exploitation"),
    ("Docker socket mounted", "container escape via docker API", "enables", 0.95, "docker escape"),
    ("SUID binary found", "privilege escalation via GTFOBins", "enables", 0.9, "privesc path"),
    ("SQL injection found", "xp_cmdshell on MSSQL", "enables", 0.85, "mssql rce"),
    ("SQL injection found", "INTO OUTFILE on MySQL", "enables", 0.85, "mysql file write"),
    ("blind command injection", "out-of-band exfiltration channel", "requires", 0.9, "blind exploit requirement"),
    ("WAF blocks path traversal", "double URL encoding %252e", "bypassed_by", 0.85, "WAF bypass"),
    ("WAF blocks path traversal", "overlong UTF-8 encoding", "bypassed_by", 0.8, "WAF bypass"),
    ("Tomcat manager found", "WAR file deployment for RCE", "enables", 0.9, "tomcat exploit"),
    ("Redis unauthenticated", "SSH key injection via CONFIG SET", "enables", 0.9, "redis exploit"),
    ("Kerberoasting", "domain user credentials", "requires", 0.95, "AD attack prereq"),
    ("Kerberoasting", "TGS ticket hashes for offline cracking", "enables", 0.9, "AD attack output"),
    ("certutil", "living off the land file download Windows", "related", 0.85, "LOLBin"),
    ("bitsadmin", "living off the land file download Windows", "related", 0.85, "LOLBin"),
]

def seed_data():
    conn = get_db()
    cur = conn.cursor()

    # Seed exploits
    cur.execute("SELECT COUNT(*) FROM exploits")
    if cur.fetchone()[0] == 0:
        for ex in SEED_EXPLOITS:
            cur.execute(
                "INSERT INTO exploits (service,version,cve,technique,auth_required,payload,gotchas,post_exploitation) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (ex["service"], ex["version"], ex["cve"], ex["technique"],
                 ex["auth_required"], ex["payload"], ex["gotchas"], ex["post_exploitation"]),
            )
        log.info(f"Seeded {len(SEED_EXPLOITS)} exploits")

    # Seed edges
    cur.execute("SELECT COUNT(*) FROM fact_edges")
    if cur.fetchone()[0] == 0:
        for src, tgt, rel, conf, ctx in SEED_EDGES:
            cur.execute(
                "INSERT OR IGNORE INTO fact_edges (source_text,target_text,relation,confidence,context) "
                "VALUES (?,?,?,?,?)",
                (src, tgt, rel, conf, ctx),
            )
        log.info(f"Seeded {len(SEED_EDGES)} relationship edges")

    conn.commit()
    conn.close()

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def fact_hash(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()

def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from qwen output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def http_post(url: str, payload: dict, timeout: int = 10) -> dict | None:
    """Simple HTTP POST returning parsed JSON, or None on failure."""
    try:
        data = json.dumps(payload).encode()
        req = Request(url, data=data, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        log.warning(f"HTTP POST {url} failed: {e}")
        return None

def text_similarity(a: str, b: str) -> float:
    """Quick similarity ratio between two texts."""
    if not a or not b:
        return 0.0
    # For speed, compare lowercased truncated versions
    a_lower = a[:500].lower()
    b_lower = b[:500].lower()
    return SequenceMatcher(None, a_lower, b_lower).ratio()

# ---------------------------------------------------------------------------
# Tag extraction (spaCy + regex)
# ---------------------------------------------------------------------------
def extract_tags(text: str) -> dict:
    tags = {}
    for category, pattern in TAG_PATTERNS.items():
        matches = list(set(pattern.findall(text)))
        if matches:
            tags[category] = [m if isinstance(m, str) else m[0] for m in matches]

    if SPACY_OK and nlp:
        doc = nlp(text[:5000])  # limit for speed
        entities = []
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PRODUCT", "GPE", "PERSON", "WORK_OF_ART"):
                entities.append({"text": ent.text, "label": ent.label_})
        if entities:
            tags["ner"] = entities

    return tags

# ---------------------------------------------------------------------------
# Layer 1: Query Reasoner
# ---------------------------------------------------------------------------
QUERY_REASONER_PROMPT = """You are an elite penetration tester and exploit developer. A colleague asks you for help with a cybersecurity problem. Your job is to figure out EXACTLY what they need and generate the search queries a HUMAN EXPERT would type.

Think like a pentester, not a search engine. Generate queries that target:
- Specific exploit techniques, CVEs, tool flags
- Bypass methods for common defenses (WAFs, filters, ASLR, sandboxes)
- Post-exploitation steps and persistence
- Related attack surfaces the user might not have considered

QUERY: {query}

Respond in STRICT JSON (no markdown, no explanation outside JSON):
{{
  "query_type": "vulnerability_lookup|technique_search|constraint_bypass|attack_path|tool_usage|recon|general",
  "entities": {{
    "services": ["service names mentioned or implied"],
    "cves": ["CVE-XXXX-XXXXX if any"],
    "techniques": ["attack techniques"],
    "constraints": ["filters, WAFs, limitations mentioned"]
  }},
  "search_intents": [
    {{"intent_type": "technique_search|vulnerability_lookup|constraint_bypass|attack_path|tool_usage", "search_query": "the exact search a pentester would run"}},
    ...3-5 intents...
  ]
}}"""

def query_reasoner(query: str) -> dict | None:
    """Layer 1: Use qwen to classify query and generate expert search intents."""
    prompt = QUERY_REASONER_PROMPT.replace("{query}", query)
    result = http_post(
        f"{OLLAMA_URL}/api/generate",
        {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 500},
        },
        timeout=LLM_TIMEOUT,
    )
    if not result or "response" not in result:
        return None

    raw = strip_think_tags(result["response"])

    # Try to extract JSON from the response
    try:
        # Find JSON object in response
        json_match = re.search(r"\{[\s\S]*\}", raw)
        if json_match:
            parsed = json.loads(json_match.group())
            return parsed
    except json.JSONDecodeError:
        log.warning(f"Failed to parse reasoner JSON: {raw[:200]}")

    return None

def fallback_intents(query: str) -> dict:
    """Generate basic intents without LLM, using regex/tag extraction."""
    tags = extract_tags(query)
    intents = [{"intent_type": "technique_search", "search_query": query}]

    # Add service-specific searches
    for svc in tags.get("service", []):
        intents.append({"intent_type": "vulnerability_lookup", "search_query": f"{svc} exploit vulnerability"})

    # Add CVE lookups
    for cve in tags.get("cve", []):
        intents.append({"intent_type": "vulnerability_lookup", "search_query": cve})

    # Add technique variations
    for tech in tags.get("technique", []):
        intents.append({"intent_type": "technique_search", "search_query": f"{tech} bypass payload"})

    # Always add a broader search
    words = query.lower().split()
    if len(words) > 3:
        intents.append({"intent_type": "technique_search", "search_query": " ".join(words[:4])})

    return {
        "query_type": "general",
        "entities": tags,
        "search_intents": intents[:5],
    }

# ---------------------------------------------------------------------------
# Layer 2a: FAISS via Memoria
# ---------------------------------------------------------------------------
def search_memoria(query: str, top_k: int = 10) -> list[dict]:
    result = http_post(f"{MEMORIA_URL}/search", {"query": query, "top_k": top_k})
    if result and "results" in result:
        return result["results"]
    return []

def multi_intent_memoria(intents: list[dict], top_k_per: int = 8) -> list[dict]:
    """Search Memoria for each intent in parallel, deduplicate.
    Results from the original query get a relevance boost."""
    all_results = []
    seen_texts = set()

    def _search(intent):
        results = search_memoria(intent["search_query"], top_k_per)
        # Tag each result with its source intent type
        for r in results:
            r["_intent_type"] = intent.get("intent_type", "")
        return results

    with ThreadPoolExecutor(max_workers=RETRIEVAL_WORKERS) as pool:
        futures = {pool.submit(_search, i): i for i in intents}
        for fut in as_completed(futures):
            try:
                results = fut.result()
                for r in results:
                    text_key = r.get("text", "").strip().lower()[:200]
                    if text_key and text_key not in seen_texts:
                        seen_texts.add(text_key)
                        # Boost results from the original query — they're most targeted
                        if r.get("_intent_type") == "original":
                            r["relevance"] = r.get("relevance", 0.5) * 1.3
                        all_results.append(r)
            except Exception as e:
                log.warning(f"Memoria search error: {e}")

    return all_results

# ---------------------------------------------------------------------------
# Layer 2b: Exploit Index (SQLite exact match)
# ---------------------------------------------------------------------------
def search_exploits(service: str = "", cve: str = "", technique: str = "") -> list[dict]:
    conn = get_db()
    cur = conn.cursor()
    conditions = []
    params = []

    if service:
        conditions.append("(service LIKE ? OR payload LIKE ? OR gotchas LIKE ?)")
        pat = f"%{service}%"
        params.extend([pat, pat, pat])
    if cve:
        conditions.append("cve LIKE ?")
        params.append(f"%{cve}%")
    if technique:
        conditions.append("(technique LIKE ? OR payload LIKE ? OR gotchas LIKE ?)")
        pat = f"%{technique}%"
        params.extend([pat, pat, pat])

    if not conditions:
        conn.close()
        return []

    query = f"SELECT * FROM exploits WHERE {' OR '.join(conditions)}"
    rows = cur.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def search_exploits_from_intents(intents: list[dict], entities: dict) -> list[dict]:
    """Search exploit index based on extracted entities and intents."""
    all_exploits = []
    seen_ids = set()

    # Search by entities
    for svc in entities.get("services", []):
        for ex in search_exploits(service=svc):
            if ex["id"] not in seen_ids:
                seen_ids.add(ex["id"])
                all_exploits.append(ex)

    for cve in entities.get("cves", []):
        for ex in search_exploits(cve=cve):
            if ex["id"] not in seen_ids:
                seen_ids.add(ex["id"])
                all_exploits.append(ex)

    for tech in entities.get("techniques", []):
        for ex in search_exploits(technique=tech):
            if ex["id"] not in seen_ids:
                seen_ids.add(ex["id"])
                all_exploits.append(ex)

    return all_exploits

# ---------------------------------------------------------------------------
# Layer 2c: Relationship Graph
# ---------------------------------------------------------------------------
def traverse_edges(fact_text: str, max_hops: int = 1) -> list[dict]:
    """Find facts related to the given fact via relationship edges."""
    conn = get_db()
    cur = conn.cursor()
    related = []

    # Search for edges where this fact is source or target
    # Require at least 2 keyword matches for relevance
    words = fact_text.strip().lower().split()
    key_words = [w for w in words if len(w) > 4][:8]  # min 5 chars, more selective

    if len(key_words) < 3:
        conn.close()
        return []

    # Get all edges, filter by multi-word match
    # With 280+ edges, require 3+ keyword matches to avoid noise
    all_edges = cur.execute(
        "SELECT source_text, target_text, relation, confidence, context FROM fact_edges "
        "WHERE relation IN ('enables', 'bypasses', 'blocks', 'requires', 'escalates')"  # skip 'related' — too broad
    ).fetchall()

    min_matches = 3 if len(all_edges) > 50 else 2  # scale threshold with graph size

    for edge in all_edges:
        src_lower = edge["source_text"].lower()
        tgt_lower = edge["target_text"].lower()
        src_matches = sum(1 for w in key_words if w in src_lower)
        tgt_matches = sum(1 for w in key_words if w in tgt_lower)

        if src_matches >= min_matches:
            related.append({
                "text": edge["target_text"],
                "relation": edge["relation"],
                "confidence": edge["confidence"],
                "context": edge["context"],
                "source": "edge",
                "edge_from": fact_text[:100],
            })
        elif tgt_matches >= min_matches:
            related.append({
                "text": edge["source_text"],
                "relation": edge["relation"],
                "confidence": edge["confidence"],
                "context": edge["context"],
                "source": "edge",
                "edge_from": fact_text[:100],
            })

    conn.close()

    # Deduplicate
    seen = set()
    deduped = []
    for item in related:
        key = item["text"].lower().strip()
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    return deduped

def get_edge_related_facts(memoria_results: list[dict]) -> list[dict]:
    """For each Memoria result, traverse edges to pull in related facts."""
    all_related = []
    seen = set()

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(traverse_edges, r.get("text", "")): r for r in memoria_results[:15]}
        for fut in as_completed(futures):
            try:
                related = fut.result()
                for item in related:
                    key = item["text"].lower().strip()[:200]
                    if key not in seen:
                        seen.add(key)
                        all_related.append(item)
            except Exception as e:
                log.warning(f"Edge traversal error: {e}")

    return all_related


# ---------------------------------------------------------------------------
# Layer 2c-ext: Triangle Inequality — Transitive Closure
# ---------------------------------------------------------------------------
# Protein folding analogy: if residue i is close to j (d<r) and j is close
# to k (d<r), then d(i,k) < 2r. This geometric constraint prunes impossible
# folds.
#
# For knowledge graphs: if fact A enables B (conf=0.9) and B enables C
# (conf=0.8), then A transitively enables C with conf = 0.9 * 0.8 = 0.72.
# We compute transitive closure to discover facts that are indirectly
# related to query results, expanding coverage of multi-step attack chains.
# ---------------------------------------------------------------------------

def compute_transitive_closure(seed_facts: list[dict], max_depth: int = 2) -> list[dict]:
    """
    Given seed facts from FAISS retrieval, compute transitive closure over
    the edge graph to find indirectly related facts.

    Uses BFS with multiplicative confidence decay:
        transitive_conf(A→C) = conf(A→B) × conf(B→C)

    This enforces a soft triangle inequality: if A is strongly related to B,
    and B to C, then A has a bounded (decayed) relationship to C.

    Args:
        seed_facts: Initial facts from FAISS retrieval
        max_depth: Maximum number of hops to traverse

    Returns:
        List of transitively discovered facts with decayed confidence
    """
    if not ENABLE_TRIANGLE_INEQUALITY:
        return []

    conn = get_db()
    cur = conn.cursor()

    # Load full edge graph into memory for BFS
    all_edges = cur.execute(
        "SELECT source_text, target_text, relation, confidence FROM fact_edges "
        "WHERE relation IN ('enables', 'bypasses', 'requires', 'escalates') "
        "AND confidence > 0.4"
    ).fetchall()
    conn.close()

    if not all_edges:
        return []

    # Build adjacency list (both directions for undirected traversal)
    adj = {}
    for e in all_edges:
        src = e["source_text"].lower().strip()
        tgt = e["target_text"].lower().strip()
        if src not in adj:
            adj[src] = []
        adj[src].append({
            "text": e["target_text"],
            "relation": e["relation"],
            "confidence": e["confidence"],
        })
        # Reverse direction (weaker confidence for reverse traversal)
        if tgt not in adj:
            adj[tgt] = []
        adj[tgt].append({
            "text": e["source_text"],
            "relation": f"rev_{e['relation']}",
            "confidence": e["confidence"] * 0.7,  # Reverse edges are weaker
        })

    # Find seed nodes in the graph via keyword matching
    visited = set()
    transitive_results = []

    for seed in seed_facts[:10]:
        seed_text = seed.get("text", "").lower().strip()
        seed_words = [w for w in seed_text.split() if len(w) > 4][:6]

        if len(seed_words) < 2:
            continue

        # Find matching nodes in adjacency list
        start_nodes = []
        for node_key in adj:
            matches = sum(1 for w in seed_words if w in node_key)
            if matches >= min(3, len(seed_words)):
                start_nodes.append(node_key)

        # BFS with confidence decay
        queue = []
        for sn in start_nodes:
            for neighbor in adj.get(sn, []):
                queue.append((
                    neighbor["text"],
                    neighbor["confidence"],
                    1,  # depth
                    neighbor["relation"],
                    [sn, neighbor["text"].lower()],
                ))

        while queue:
            text, acc_conf, depth, relation, path = queue.pop(0)
            text_lower = text.lower().strip()

            if text_lower in visited:
                continue
            visited.add(text_lower)

            # Only include if transitive confidence is meaningful
            if acc_conf > 0.25:
                transitive_results.append({
                    "text": text,
                    "confidence": acc_conf,
                    "source": "transitive",
                    "relation": relation,
                    "depth": depth,
                    "path_length": len(path),
                })

            # Continue BFS if within depth limit
            if depth < max_depth:
                for neighbor in adj.get(text_lower, []):
                    n_lower = neighbor["text"].lower().strip()
                    if n_lower not in visited:
                        # Multiplicative decay — triangle inequality
                        new_conf = acc_conf * neighbor["confidence"]
                        if new_conf > 0.15:  # Prune very weak paths
                            queue.append((
                                neighbor["text"],
                                new_conf,
                                depth + 1,
                                neighbor["relation"],
                                path + [n_lower],
                            ))

    # Sort by confidence descending, cap results
    transitive_results.sort(key=lambda x: x["confidence"], reverse=True)
    return transitive_results[:10]

# ---------------------------------------------------------------------------
# Layer 2d: Co-occurrence Matrix
# ---------------------------------------------------------------------------
def get_cooccurrence_boosts(fact_texts: list[str]) -> dict[str, float]:
    """For given facts, find co-occurring facts and their boost scores."""
    if not fact_texts:
        return {}

    conn = get_db()
    cur = conn.cursor()
    boosts = {}  # text -> boost score

    hashes = {fact_hash(t): t for t in fact_texts}

    for h in hashes:
        rows = cur.execute(
            "SELECT fact_b_text, count FROM co_occurrences WHERE fact_a_hash = ? "
            "UNION "
            "SELECT fact_a_text, count FROM co_occurrences WHERE fact_b_hash = ?",
            (h, h),
        ).fetchall()
        for r in rows:
            text = r[0]
            count = r[1]
            boost = 0.1 * math.log(count + 1)
            if text in boosts:
                boosts[text] = max(boosts[text], boost)
            else:
                boosts[text] = boost

    conn.close()
    return boosts

# ---------------------------------------------------------------------------
# Layer 4: Energy Minimization — Attack Path Optimizer (Boltzmann)
# ---------------------------------------------------------------------------
# Protein folding analogy: the correct 3D structure is the one that minimizes
# the global free energy. The probability of a conformation is given by the
# Boltzmann distribution:
#
#   P(state) = e^(-E/kT) / Z    where Z = Σ e^(-E_i/kT)
#
# For attack paths:
#
#   E_total(path) = Σ E_node(f_i) + Σ E_edge(f_i, f_{i+1}) + E_coverage
#
#   E_node(f) = 1 - relevance(f, query)           [irrelevant = high energy]
#   E_edge(f_i, f_{i+1}) = 1 - connection(i,j)    [disconnected = high energy]
#   E_coverage = α × (1 - |covered_steps| / |total_steps|)  [missing steps = penalty]
#
# Temperature T controls exploration:
#   T → 0: greedy, only the single lowest-energy path
#   T → ∞: uniform, all paths equally likely
#   T ≈ 0.5: balanced exploration (our default)
#
# Each candidate fact receives a Boltzmann-weighted score:
#   boost(f) = Σ_paths P(path) × position_weight(f, path)
#
# This means a fact can be boosted by appearing in MULTIPLE good paths,
# not just the single best one. This is analogous to how AlphaFold2
# considers an ensemble of conformations weighted by their energies.
# ---------------------------------------------------------------------------

# Temperature controls exploration/exploitation tradeoff
BOLTZMANN_TEMP = 0.5   # Lower = more greedy; higher = more diverse
COVERAGE_PENALTY = 0.3  # Weight for uncovered chain steps

def _build_edge_index() -> dict:
    """Load edge graph into a dict for O(1) lookup."""
    conn = get_db()
    cur = conn.cursor()
    edges = cur.execute(
        "SELECT source_text, target_text, confidence FROM fact_edges "
        "WHERE relation IN ('enables', 'bypasses', 'requires', 'escalates')"
    ).fetchall()
    conn.close()

    index = {}
    for e in edges:
        key_fwd = (e["source_text"].lower().strip()[:120], e["target_text"].lower().strip()[:120])
        key_rev = (e["target_text"].lower().strip()[:120], e["source_text"].lower().strip()[:120])
        index[key_fwd] = e["confidence"]
        index[key_rev] = e["confidence"] * 0.6  # Reverse traversal is weaker
    return index


def _keyword_overlap(text_a: str, text_b: str) -> float:
    """Fast keyword-overlap similarity (Jaccard on 5+ char words)."""
    words_a = set(w for w in text_a.lower().split() if len(w) > 4)
    words_b = set(w for w in text_b.lower().split() if len(w) > 4)
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def _boltzmann_probabilities(energies: list[float], temperature: float) -> list[float]:
    """
    Compute Boltzmann distribution over a set of energies.

    P(i) = e^(-E_i / T) / Z   where Z = Σ_j e^(-E_j / T)

    Uses log-sum-exp trick for numerical stability.
    """
    if not energies or temperature <= 0:
        return [1.0 / len(energies)] * len(energies) if energies else []

    # Scale energies by temperature
    scaled = [-e / temperature for e in energies]
    # Log-sum-exp for numerical stability
    max_s = max(scaled)
    log_z = max_s + math.log(sum(math.exp(s - max_s) for s in scaled))
    probs = [math.exp(s - log_z) for s in scaled]
    return probs


def energy_minimize_paths(
    candidates: list[dict],
    query: str,
    beam_width: int = 8,
    max_path_len: int = 6,
    temperature: float = BOLTZMANN_TEMP,
) -> tuple[list[dict], dict]:
    """
    Find minimum-energy attack paths and re-rank candidates using
    Boltzmann-weighted ensemble scoring.

    Unlike greedy beam search that picks ONE best path, this computes
    Boltzmann probabilities over the top-k paths and weights each fact
    by its expected contribution across the ensemble.

    Returns:
        (re-ranked candidates, path_metadata dict)
    """
    if not ENABLE_ENERGY_MINIMIZATION or len(candidates) < 3:
        return candidates, {"energy_minimization": "disabled or too few candidates"}

    edge_index = _build_edge_index()

    # Work with top candidates (limit search space)
    working = candidates[:12]
    texts = [c.get("text", "")[:200] for c in working]
    n = len(texts)

    # --- Precompute energy landscape ---

    # Node energies: E_node = 1 - relevance
    node_energies = []
    for c in working:
        rel = c.get("relevance", c.get("combined_score", 0.5))
        node_energies.append(1.0 - min(max(rel, 0.0), 1.0))

    # Edge energies: E_edge = 1 - connection_strength (n×n matrix)
    edge_energies = [[0.0] * n for _ in range(n)]
    energy_method = "keyword_overlap"  # Track which method was used

    # Try learned model first (batch mode for efficiency)
    learned_scores = None
    if ENABLE_LEARNED_ENERGY:
        learned_scores = _batch_learned_connection(texts)
        if learned_scores is not None:
            energy_method = "learned_blended"

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            t_i = texts[i].lower()[:120]
            t_j = texts[j].lower()[:120]
            # Check explicit edges first (always highest priority)
            conf = edge_index.get((t_i, t_j))
            if conf:
                edge_energies[i][j] = 1.0 - conf
            elif learned_scores is not None:
                # Blend learned classifier with keyword overlap:
                # learned captures semantic attack-chain structure,
                # keyword overlap captures surface-level vocabulary matches.
                # Blend ratio: 0.6 learned + 0.4 keyword (learned is primary signal)
                learned_score = learned_scores[i][j]
                overlap = _keyword_overlap(texts[i], texts[j])
                kw_score = min(overlap * 0.5, 0.5)
                blended = 0.6 * learned_score + 0.4 * kw_score
                edge_energies[i][j] = 1.0 - blended
            else:
                # Fall back to keyword overlap (weakest signal)
                overlap = _keyword_overlap(texts[i], texts[j])
                edge_energies[i][j] = 1.0 - min(overlap * 0.5, 0.5)

    # --- Beam search for top-k paths ---
    beams = [{"path": [i], "energy": node_energies[i]} for i in range(n)]

    for _step in range(min(max_path_len - 1, n - 1)):
        new_beams = []
        for beam in beams:
            last = beam["path"][-1]
            path_set = set(beam["path"])
            for nxt in range(n):
                if nxt in path_set:
                    continue
                step_energy = edge_energies[last][nxt] + node_energies[nxt]
                new_beams.append({
                    "path": beam["path"] + [nxt],
                    "energy": beam["energy"] + step_energy,
                })
        if not new_beams:
            break
        # Keep top beams by normalized energy
        new_beams.sort(key=lambda b: b["energy"] / len(b["path"]))
        beams = new_beams[:beam_width * 2]  # Keep extra for diversity

    if not beams:
        return candidates, {"energy_minimization": "no paths found"}

    # Filter to paths of length >= 3 (meaningful chains)
    long_beams = [b for b in beams if len(b["path"]) >= 3]
    if not long_beams:
        long_beams = beams

    # --- Boltzmann ensemble scoring ---
    # Compute normalized energies for each path
    norm_energies = [b["energy"] / len(b["path"]) for b in long_beams]
    path_probs = _boltzmann_probabilities(norm_energies, temperature)

    # Accumulate Boltzmann-weighted boosts for each fact
    # A fact's boost = Σ P(path) × position_weight(fact_in_path)
    fact_boosts = [0.0] * n
    fact_path_count = [0] * n

    for beam, prob in zip(long_beams, path_probs):
        path = beam["path"]
        path_len = len(path)
        for pos, idx in enumerate(path):
            # Position weight: earlier in path = bigger weight
            # Decays linearly from 1.0 to 0.3 along the path
            pos_weight = 1.0 - 0.7 * (pos / max(path_len - 1, 1))
            fact_boosts[idx] += prob * pos_weight
            fact_path_count[idx] += 1

    # Apply boosts to candidates
    max_boost = max(fact_boosts) if any(b > 0 for b in fact_boosts) else 1.0
    for i, c in enumerate(working):
        if fact_boosts[i] > 0:
            # Normalize boost to [0, 0.3] range
            normalized_boost = 0.3 * (fact_boosts[i] / max_boost)
            c["boltzmann_boost"] = round(normalized_boost, 4)
            c["combined_score"] = c.get("combined_score", 0.5) * (1.0 + normalized_boost)
            c["in_optimal_paths"] = fact_path_count[i]
            c["path_probability"] = round(fact_boosts[i], 4)
        else:
            c["boltzmann_boost"] = 0.0
            c["in_optimal_paths"] = 0
            c["path_probability"] = 0.0

    # Re-sort all candidates
    candidates.sort(key=lambda x: x.get("combined_score", 0), reverse=True)

    # Find the single best path for reporting
    best_idx = norm_energies.index(min(norm_energies))
    best = long_beams[best_idx]

    # Compute free energy: F = -T × ln(Z)
    # This is the thermodynamic quantity that tells us how "confident"
    # the ensemble is — low free energy = strong consensus on path structure
    max_s = max(-e / temperature for e in norm_energies)
    log_z = max_s + math.log(sum(math.exp(-e / temperature - max_s) for e in norm_energies))
    free_energy = -temperature * log_z

    path_metadata = {
        "method": "boltzmann_ensemble",
        "energy_method": energy_method,
        "temperature": temperature,
        "paths_evaluated": len(long_beams),
        "best_path_length": len(best["path"]),
        "best_path_energy": round(best["energy"], 4),
        "best_normalized_energy": round(min(norm_energies), 4),
        "free_energy": round(free_energy, 4),
        "ensemble_entropy": round(-sum(p * math.log(p + 1e-10) for p in path_probs), 4),
        "best_path_facts": [texts[i][:80] for i in best["path"]],
        "fact_boltzmann_boosts": {texts[i][:50]: round(fact_boosts[i], 4) for i in range(n) if fact_boosts[i] > 0},
    }

    return candidates, path_metadata


def record_cooccurrence(facts: list[str], session: str = ""):
    """Record that these facts were used together."""
    if len(facts) < 2:
        return

    conn = get_db()
    cur = conn.cursor()

    for i in range(len(facts)):
        for j in range(i + 1, len(facts)):
            a, b = facts[i].strip(), facts[j].strip()
            ha, hb = fact_hash(a), fact_hash(b)
            # Ensure consistent ordering
            if ha > hb:
                a, b, ha, hb = b, a, hb, ha

            cur.execute(
                "INSERT INTO co_occurrences (fact_a_hash, fact_b_hash, fact_a_text, fact_b_text, count, last_session) "
                "VALUES (?, ?, ?, ?, 1, ?) "
                "ON CONFLICT(fact_a_hash, fact_b_hash) DO UPDATE SET count = count + 1, last_session = ?",
                (ha, hb, a, b, session, session),
            )

    conn.commit()
    conn.close()

# ---------------------------------------------------------------------------
# Layer 3: Path Assembler
# ---------------------------------------------------------------------------
def assemble_results(
    memoria_results: list[dict],
    edge_results: list[dict],
    cooccurrence_boosts: dict[str, float],
    top_k: int = 10,
) -> list[dict]:
    """Deduplicate, score, and rank all candidate facts."""
    candidates = []

    # Add Memoria results
    for r in memoria_results:
        candidates.append({
            "text": r.get("text", ""),
            "relevance": r.get("relevance", 0.5),
            "source": r.get("source", "memoria"),
            "id": r.get("id", ""),
            "via_edge": False,
        })

    # Add edge-traversed results (capped — they supplement, not replace)
    for r in edge_results[:5]:
        candidates.append({
            "text": r.get("text", ""),
            "relevance": r.get("confidence", 0.3) * 0.5,  # Lower base: edges supplement FAISS
            "source": "edge",
            "id": "",
            "via_edge": True,
            "relation": r.get("relation", ""),
            "edge_from": r.get("edge_from", ""),
        })

    # Deduplicate by text similarity
    deduped = []
    for cand in candidates:
        is_dup = False
        for existing in deduped:
            if text_similarity(cand["text"], existing["text"]) > 0.90:
                # Keep the one with higher relevance
                if cand["relevance"] > existing["relevance"]:
                    existing.update(cand)
                is_dup = True
                break
        if not is_dup:
            deduped.append(cand)

    # Score each fact
    for cand in deduped:
        edge_bonus = 0.3 if cand.get("via_edge") else 0.0
        cooc_bonus = cooccurrence_boosts.get(cand["text"], 0.0)
        cand["combined_score"] = cand["relevance"] * (1.0 + edge_bonus + cooc_bonus)
        cand["edge_bonus"] = edge_bonus
        cand["cooccurrence_bonus"] = round(cooc_bonus, 4)

    # Sort by combined score descending
    deduped.sort(key=lambda x: x["combined_score"], reverse=True)

    return deduped[:top_k]

# ---------------------------------------------------------------------------
# LLM relationship extraction
# ---------------------------------------------------------------------------
RELATION_PROMPT = """You are a cybersecurity knowledge graph expert. Analyze the relationship between these two facts:

FACT A: {fact_a}
FACT B: {fact_b}

What is the relationship? Choose one: enables, bypasses, blocks, requires, escalates, related, none

Respond in STRICT JSON:
{{"relation": "enables|bypasses|blocks|requires|escalates|related|none", "confidence": 0.0-1.0, "explanation": "brief reason"}}"""

def extract_relationship(fact_a: str, fact_b: str) -> dict | None:
    prompt = RELATION_PROMPT.replace("{fact_a}", fact_a).replace("{fact_b}", fact_b)
    result = http_post(
        f"{OLLAMA_URL}/api/generate",
        {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 200},
        },
        timeout=LLM_TIMEOUT,
    )
    if not result or "response" not in result:
        return None

    raw = strip_think_tags(result["response"])
    try:
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        pass
    return None

def store_edge(fact_a: str, fact_b: str, relation: str, confidence: float, context: str = ""):
    with _db_lock:
        conn = get_db()
        conn.execute(
            "INSERT OR REPLACE INTO fact_edges (source_text, target_text, relation, confidence, context) "
            "VALUES (?, ?, ?, ?, ?)",
            (fact_a, fact_b, relation, confidence, context),
        )
        conn.commit()
        conn.close()

# ---------------------------------------------------------------------------
# Full retrieval pipeline (Layers 1→2→3)
# ---------------------------------------------------------------------------
def full_retrieve(query: str, top_k: int = 10) -> dict:
    t0 = time.time()

    # Layer 1: Query Reasoner
    reasoning = query_reasoner(query)
    if reasoning is None:
        log.info("LLM reasoner unavailable, using fallback intents")
        reasoning = fallback_intents(query)

    intents = reasoning.get("search_intents", [])
    entities = reasoning.get("entities", {})
    query_type = reasoning.get("query_type", "general")

    # ALWAYS include original query as first intent — LLM intents supplement, not replace
    intents.insert(0, {"intent_type": "original", "search_query": query})

    if len(intents) < 2:
        intents.append({"intent_type": "technique_search", "search_query": query})

    # Layer 2: Multi-Strategy Retrieval (parallel)
    memoria_results = []
    exploit_results = []
    edge_results = []
    cooccurrence_boosts = {}

    with ThreadPoolExecutor(max_workers=4) as pool:
        # 2a: FAISS via Memoria
        fut_memoria = pool.submit(multi_intent_memoria, intents, top_k)
        # 2b: Exploit Index
        fut_exploits = pool.submit(search_exploits_from_intents, intents, entities)

        memoria_results = fut_memoria.result()
        exploit_results = fut_exploits.result()

    # 2c: Relationship graph (needs Memoria results first)
    edge_results = get_edge_related_facts(memoria_results)

    # 2c-ext: Triangle inequality — transitive closure
    transitive_results = compute_transitive_closure(memoria_results)
    # Merge transitive results into edge results (they supplement)
    edge_results.extend(transitive_results)

    # 2d: Co-occurrence boosts
    fact_texts = [r.get("text", "") for r in memoria_results if r.get("text")]
    cooccurrence_boosts = get_cooccurrence_boosts(fact_texts) if ENABLE_COOCCURRENCE else {}

    # Layer 3: Path Assembler
    results = assemble_results(memoria_results, edge_results, cooccurrence_boosts, top_k)

    # Layer 4: Energy Minimization — path optimizer
    path_metadata = {}
    results, path_metadata = energy_minimize_paths(results, query)
    results = results[:top_k]

    elapsed = time.time() - t0

    # Log query
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO query_log (query, query_type, intents_json, results_count) VALUES (?, ?, ?, ?)",
            (query, query_type, json.dumps(intents), len(results)),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

    return {
        "results": results,
        "exploits": exploit_results,
        "reasoning": {
            "query_type": query_type,
            "entities": entities,
            "llm_available": reasoning is not None,
        },
        "search_intents": intents,
        "stats": {
            "memoria_hits": len(memoria_results),
            "edge_hits": len(edge_results),
            "transitive_hits": len(transitive_results),
            "exploit_hits": len(exploit_results),
            "elapsed_sec": round(elapsed, 3),
            "features": {
                "triangle_inequality": ENABLE_TRIANGLE_INEQUALITY,
                "energy_minimization": ENABLE_ENERGY_MINIMIZATION,
                "cooccurrence": ENABLE_COOCCURRENCE,
                "learned_energy": ENABLE_LEARNED_ENERGY,
                "learned_model_loaded": _connection_model is not None,
            },
        },
        "path_optimization": path_metadata,
    }

# ---------------------------------------------------------------------------
# Known attack chains — used to seed co-occurrence matrix
# ---------------------------------------------------------------------------
# These come from actual HTB boxes we've pwned. Each chain is a list of
# search queries representing the steps of the attack. We search Memoria
# for matching facts, then record all pairwise co-occurrences.
KNOWN_ATTACK_CHAINS = [
    {
        "name": "Conversor (HTB Easy)",
        "steps": [
            "XSLT server-side processing",
            "XSLT code execution exsl:document write file",
            "Python reverse shell script",
            "cron job executing scripts in web directory",
            "CVE-2024-48990 needrestart PYTHONPATH injection privilege escalation",
        ],
    },
    {
        "name": "MonitorsFour (HTB Easy)",
        "steps": [
            "Cacti network monitoring default credentials",
            "CVE-2025-24367 Cacti rrdtool newline injection",
            "PHP webshell upload via rrdtool graph CSV output",
            "Docker container escape via exposed Docker API",
            "Docker Desktop WSL2 host filesystem mount",
        ],
    },
    {
        "name": "Redis SSH Key Injection (classic)",
        "steps": [
            "Redis unauthenticated access port 6379",
            "Redis CONFIG SET dir /root/.ssh",
            "Redis CONFIG SET dbfilename authorized_keys",
            "SSH public key injection via Redis SET command",
            "SSH login with injected key",
        ],
    },
    {
        "name": "Tomcat WAR Deployment (classic)",
        "steps": [
            "Apache Tomcat manager interface discovery",
            "Tomcat default credentials brute force",
            "WAR file generation msfvenom reverse shell",
            "WAR file deployment via Tomcat manager upload",
            "reverse shell callback from deployed JSP",
        ],
    },
    {
        "name": "Kerberoasting to Domain Admin",
        "steps": [
            "Active Directory domain user enumeration",
            "SPN service principal name enumeration",
            "Kerberoasting TGS ticket request GetUserSPNs",
            "TGS hash extraction for offline cracking",
            "hashcat mode 13100 Kerberos TGS cracking",
            "service account password domain admin escalation",
        ],
    },
    {
        "name": "SQLi to OS Command (MSSQL)",
        "steps": [
            "SQL injection detection and confirmation",
            "MSSQL database identification",
            "xp_cmdshell enable via sp_configure",
            "xp_cmdshell OS command execution",
            "reverse shell from xp_cmdshell",
        ],
    },
    {
        "name": "SUID Privesc (Linux)",
        "steps": [
            "find SUID binaries find / -perm -4000",
            "GTFOBins SUID binary exploitation",
            "SUID binary shell escape /bin/bash -p",
            "root privilege escalation verification",
        ],
    },
    {
        "name": "ARP MITM Credential Capture",
        "steps": [
            "IP forwarding enable sysctl net.ipv4.ip_forward",
            "ARP spoofing arpspoof ettercap bettercap",
            "man in the middle traffic interception",
            "credential sniffing HTTP FTP cleartext protocols",
        ],
    },
    {
        "name": "LOLBins File Download (Windows)",
        "steps": [
            "Windows living off the land binary LOLBin LOLBAS",
            "certutil -urlcache -split -f download file",
            "bitsadmin /transfer job download",
            "PowerShell Invoke-WebRequest wget download",
            "downloaded payload execution",
        ],
    },
    {
        "name": "Path Traversal WAF Bypass",
        "steps": [
            "path traversal directory traversal ../ detection",
            "WAF blocks path traversal patterns",
            "double URL encoding %252e%252e%252f bypass",
            "overlong UTF-8 encoding ..%c0%af bypass",
            "null byte %00 truncation bypass",
            "sensitive file read /etc/passwd /etc/shadow",
        ],
    },
    {
        "name": "Blind Command Injection Exfiltration",
        "steps": [
            "blind command injection detection time delay sleep",
            "out-of-band exfiltration channel setup",
            "DNS exfiltration via subdomain encoding",
            "curl wget HTTP request to attacker server",
            "data reconstruction from OOB callbacks",
        ],
    },
    {
        "name": "Command Injection Space Bypass",
        "steps": [
            "command injection space character filtered",
            "${IFS} internal field separator space bypass",
            "$IFS$9 alternative space bypass",
            "brace expansion {cat,/etc/passwd} command execution",
            "tab character %09 space alternative",
        ],
    },
]


def seed_cooccurrences_worker(chains: list[dict] | None = None):
    """
    Seed the co-occurrence matrix from known attack chains.

    For each chain, search Memoria for facts matching each step, then
    record all pairwise co-occurrences between matched facts.
    """
    if chains is None:
        chains = KNOWN_ATTACK_CHAINS

    log.info(f"Seeding co-occurrences from {len(chains)} attack chains...")
    total_pairs = 0

    for chain in chains:
        chain_name = chain.get("name", "unknown")
        steps = chain.get("steps", [])
        if len(steps) < 2:
            continue

        # Search Memoria for facts matching each step
        chain_facts = []
        for step in steps:
            results = search_memoria(step, top_k=3)
            for r in results:
                text = r.get("text", "").strip()
                if text and len(text) > 20:
                    chain_facts.append(text)

        # Deduplicate
        seen = set()
        unique_facts = []
        for f in chain_facts:
            key = f[:200].lower()
            if key not in seen:
                seen.add(key)
                unique_facts.append(f)

        if len(unique_facts) < 2:
            log.info(f"  {chain_name}: too few matching facts ({len(unique_facts)})")
            continue

        # Record co-occurrences
        record_cooccurrence(unique_facts, session=f"seed:{chain_name}")
        n_pairs = len(unique_facts) * (len(unique_facts) - 1) // 2
        total_pairs += n_pairs
        log.info(f"  {chain_name}: {len(unique_facts)} facts, {n_pairs} pairs recorded")

    log.info(f"Co-occurrence seeding complete: {total_pairs} total pairs from {len(chains)} chains")


# ---------------------------------------------------------------------------
# Background batch indexing
# ---------------------------------------------------------------------------
_indexing_lock = threading.Lock()
_indexing_active = False

def batch_index_worker():
    """Pull facts from Memoria and extract pairwise relationships."""
    global _indexing_active
    if _indexing_active:
        log.info("Batch indexing already running, skipping")
        return

    with _indexing_lock:
        _indexing_active = True

    try:
        log.info("Starting batch index...")
        # Pull a sample of facts from Memoria
        sample_queries = [
            "privilege escalation", "command injection", "SQL injection",
            "file upload bypass", "container escape", "WAF bypass",
            "reverse shell", "password cracking", "lateral movement",
            "persistence mechanism",
        ]

        all_facts = []
        seen = set()
        for q in sample_queries:
            results = search_memoria(q, top_k=10)
            for r in results:
                text = r.get("text", "").strip()
                key = text[:200].lower()
                if key and key not in seen:
                    seen.add(key)
                    all_facts.append(text)

        log.info(f"Batch index: {len(all_facts)} unique facts collected")

        # Extract pairwise relationships for a subset
        pairs_checked = 0
        edges_created = 0
        for i in range(min(len(all_facts), 30)):
            for j in range(i + 1, min(len(all_facts), 30)):
                if pairs_checked > 50:
                    break
                rel = extract_relationship(all_facts[i], all_facts[j])
                pairs_checked += 1
                if rel and rel.get("relation", "none") != "none" and rel.get("confidence", 0) > 0.5:
                    store_edge(
                        all_facts[i], all_facts[j],
                        rel["relation"], rel["confidence"],
                        rel.get("explanation", ""),
                    )
                    edges_created += 1
            if pairs_checked > 50:
                break

        log.info(f"Batch index complete: {pairs_checked} pairs checked, {edges_created} edges created")
    except Exception as e:
        log.error(f"Batch index error: {e}")
    finally:
        _indexing_active = False

# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------
class MethodicHandler(BaseHTTPRequestHandler):
    server_version = "Methodic/2.0"

    def log_message(self, format, *args):
        log.info(f"{self.address_string()} {format % args}")

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def _respond(self, code: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self._handle_health()
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?")[0]
        handlers = {
            "/retrieve": self._handle_retrieve,
            "/smart-search": self._handle_retrieve,
            "/search": self._handle_search_proxy,
            "/exploit-lookup": self._handle_exploit_lookup,
            "/exploit-add": self._handle_exploit_add,
            "/feedback": self._handle_feedback,
            "/relate": self._handle_relate,
            "/tag": self._handle_tag,
            "/batch-index": self._handle_batch_index,
            "/seed-cooccurrences": self._handle_seed_cooccurrences,
            "/features": self._handle_feature_toggle,
        }
        handler = handlers.get(path)
        if handler:
            try:
                handler()
            except Exception as e:
                log.error(f"Handler error on {path}: {e}", exc_info=True)
                self._respond(500, {"error": str(e)})
        else:
            self._respond(404, {"error": "not found"})

    # --- Endpoint handlers ---

    def _handle_retrieve(self):
        body = self._read_body()
        query = body.get("query", "").strip()
        if not query:
            self._respond(400, {"error": "query required"})
            return
        top_k = int(body.get("top_k", 10))
        result = full_retrieve(query, top_k)
        self._respond(200, result)

    def _handle_search_proxy(self):
        """Proxy to raw Memoria for baseline comparison."""
        body = self._read_body()
        query = body.get("query", "").strip()
        if not query:
            self._respond(400, {"error": "query required"})
            return
        top_k = int(body.get("top_k", 10))
        result = http_post(f"{MEMORIA_URL}/search", {"query": query, "top_k": top_k})
        if result is not None:
            self._respond(200, result)
        else:
            self._respond(502, {"error": "Memoria unavailable"})

    def _handle_exploit_lookup(self):
        body = self._read_body()
        service = body.get("service", "")
        cve = body.get("cve", "")
        technique = body.get("technique", "")
        if not (service or cve or technique):
            self._respond(400, {"error": "at least one of service, cve, technique required"})
            return
        results = search_exploits(service, cve, technique)
        self._respond(200, {"exploits": results, "count": len(results)})

    def _handle_exploit_add(self):
        body = self._read_body()
        required = ["service", "technique", "payload"]
        for field in required:
            if not body.get(field):
                self._respond(400, {"error": f"{field} required"})
                return

        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO exploits (service,version,cve,technique,auth_required,payload,gotchas,post_exploitation) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                body["service"], body.get("version", ""), body.get("cve", ""),
                body["technique"], body.get("auth_required", "unknown"),
                body["payload"], body.get("gotchas", ""), body.get("post_exploitation", ""),
            ),
        )
        conn.commit()
        exploit_id = cur.lastrowid
        conn.close()
        self._respond(201, {"id": exploit_id, "status": "created"})

    def _handle_feedback(self):
        body = self._read_body()
        facts = body.get("facts_used", [])
        session = body.get("session", "")
        if len(facts) < 2:
            self._respond(400, {"error": "need at least 2 facts_used"})
            return
        record_cooccurrence(facts, session)
        n_pairs = len(facts) * (len(facts) - 1) // 2
        self._respond(200, {"status": "recorded", "pairs": n_pairs})

    def _handle_relate(self):
        body = self._read_body()
        fact_a = body.get("fact_a", "").strip()
        fact_b = body.get("fact_b", "").strip()
        if not fact_a or not fact_b:
            self._respond(400, {"error": "fact_a and fact_b required"})
            return

        rel = extract_relationship(fact_a, fact_b)
        if rel is None:
            self._respond(502, {"error": "LLM unavailable for relationship extraction"})
            return

        # Store if meaningful
        if rel.get("relation", "none") != "none" and rel.get("confidence", 0) > 0.3:
            store_edge(fact_a, fact_b, rel["relation"], rel["confidence"], rel.get("explanation", ""))

        self._respond(200, rel)

    def _handle_tag(self):
        body = self._read_body()
        text = body.get("text", "").strip()
        if not text:
            self._respond(400, {"error": "text required"})
            return
        tags = extract_tags(text)
        self._respond(200, {"tags": tags})

    def _handle_batch_index(self):
        thread = threading.Thread(target=batch_index_worker, daemon=True)
        thread.start()
        self._respond(202, {"status": "indexing started in background"})

    def _handle_seed_cooccurrences(self):
        """Seed co-occurrence matrix from known attack chains."""
        body = self._read_body()
        chains = body.get("chains", KNOWN_ATTACK_CHAINS)
        thread = threading.Thread(
            target=seed_cooccurrences_worker, args=(chains,), daemon=True
        )
        thread.start()
        self._respond(202, {"status": "co-occurrence seeding started", "chains": len(chains)})

    def _handle_feature_toggle(self):
        """Toggle feature flags at runtime for ablation testing."""
        global ENABLE_TRIANGLE_INEQUALITY, ENABLE_ENERGY_MINIMIZATION, ENABLE_COOCCURRENCE, ENABLE_LEARNED_ENERGY
        body = self._read_body()
        if "triangle_inequality" in body:
            ENABLE_TRIANGLE_INEQUALITY = bool(body["triangle_inequality"])
        if "energy_minimization" in body:
            ENABLE_ENERGY_MINIMIZATION = bool(body["energy_minimization"])
        if "cooccurrence" in body:
            ENABLE_COOCCURRENCE = bool(body["cooccurrence"])
        if "learned_energy" in body:
            ENABLE_LEARNED_ENERGY = bool(body["learned_energy"])
            if ENABLE_LEARNED_ENERGY and _connection_model is None:
                _load_connection_model()  # Try loading if toggled on
        self._respond(200, {
            "triangle_inequality": ENABLE_TRIANGLE_INEQUALITY,
            "energy_minimization": ENABLE_ENERGY_MINIMIZATION,
            "cooccurrence": ENABLE_COOCCURRENCE,
            "learned_energy": ENABLE_LEARNED_ENERGY,
            "learned_model_loaded": _connection_model is not None,
        })

    def _handle_health(self):
        conn = get_db()
        cur = conn.cursor()

        edge_count = cur.execute("SELECT COUNT(*) FROM fact_edges").fetchone()[0]
        exploit_count = cur.execute("SELECT COUNT(*) FROM exploits").fetchone()[0]
        cooc_count = cur.execute("SELECT COUNT(*) FROM co_occurrences").fetchone()[0]
        query_count = cur.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]

        # Check Memoria
        memoria_ok = False
        try:
            from urllib.request import urlopen as _urlopen
            with _urlopen(f"{MEMORIA_URL}/health", timeout=3) as resp:
                memoria_ok = resp.status == 200
        except Exception:
            # Try a search as health check
            r = http_post(f"{MEMORIA_URL}/search", {"query": "test", "top_k": 1}, timeout=3)
            memoria_ok = r is not None

        # Check Ollama
        ollama_ok = False
        try:
            from urllib.request import urlopen as _urlopen
            with _urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as resp:
                ollama_ok = resp.status == 200
        except Exception:
            pass

        conn.close()

        self._respond(200, {
            "status": "healthy",
            "engine": "methodic-retrieval",
            "version": "2.0.0",
            "edges": edge_count,
            "exploits": exploit_count,
            "co_occurrences": cooc_count,
            "queries_served": query_count,
            "spacy_loaded": SPACY_OK,
            "memoria_reachable": memoria_ok,
            "ollama_reachable": ollama_ok,
        })

# ---------------------------------------------------------------------------
# Threaded HTTP server
# ---------------------------------------------------------------------------
class ThreadedHTTPServer(HTTPServer):
    """Handle each request in a new thread."""
    allow_reuse_address = True

    def process_request(self, request, client_address):
        thread = threading.Thread(target=self._handle, args=(request, client_address), daemon=True)
        thread.start()

    def _handle(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info("=" * 60)
    log.info("Methodic Retrieval Engine v2.0.0 — protein-folding-inspired retrieval")
    log.info("=" * 60)

    # Init database
    init_db()
    seed_data()

    # Print startup stats
    conn = get_db()
    cur = conn.cursor()
    edges = cur.execute("SELECT COUNT(*) FROM fact_edges").fetchone()[0]
    exploits = cur.execute("SELECT COUNT(*) FROM exploits").fetchone()[0]
    coocs = cur.execute("SELECT COUNT(*) FROM co_occurrences").fetchone()[0]
    conn.close()

    log.info(f"Database: {DB_PATH}")
    log.info(f"  Edges: {edges}  |  Exploits: {exploits}  |  Co-occurrences: {coocs}")
    log.info(f"Memoria: {MEMORIA_URL}")
    log.info(f"Ollama:  {OLLAMA_URL} (model: {OLLAMA_MODEL})")
    log.info(f"spaCy:   {'loaded' if SPACY_OK else 'unavailable'}")
    log.info(f"Listening on 0.0.0.0:{PORT}")
    log.info("=" * 60)

    server = ThreadedHTTPServer(("0.0.0.0", PORT), MethodicHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down...")
        server.shutdown()

if __name__ == "__main__":
    main()
