#!/usr/bin/env python3
"""
Train a learned energy function for Methodic Retrieval.

Replaces keyword-overlap connection scoring with a classifier that predicts
whether fact_B logically follows fact_A in an attack chain.

Data sources:
  1. MITRE ATT&CK enterprise-attack.json — technique chains from procedures
  2. methodic.db co_occurrences table (3,189 pairs)
  3. methodic.db fact_edges table (282 typed edges)

Model: Logistic regression on concatenated sentence embeddings (all-MiniLM-L6-v2).
Output: /home/om/htb-autopwn/connection_model.pkl

Usage: python3 train_connection_model.py
"""

import hashlib
import json
import logging
import os
import pickle
import random
import sqlite3
import time
from urllib.request import urlopen, Request

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_connection")

DB_PATH = "/home/om/htb-autopwn/methodic.db"
MODEL_PATH = "/home/om/htb-autopwn/connection_model.pkl"
ATTACK_JSON_URL = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
ATTACK_JSON_CACHE = "/tmp/enterprise-attack.json"
MEMORIA_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Step 1: Gather training data
# ---------------------------------------------------------------------------

def fetch_attack_json() -> dict:
    """Download or load cached MITRE ATT&CK JSON."""
    if os.path.exists(ATTACK_JSON_CACHE):
        age_hours = (time.time() - os.path.getmtime(ATTACK_JSON_CACHE)) / 3600
        if age_hours < 24:
            log.info(f"Using cached ATT&CK JSON ({age_hours:.1f}h old)")
            with open(ATTACK_JSON_CACHE, "r") as f:
                return json.load(f)

    log.info("Downloading MITRE ATT&CK enterprise-attack.json...")
    req = Request(ATTACK_JSON_URL, headers={"User-Agent": "methodic-trainer/1.0"})
    with urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    with open(ATTACK_JSON_CACHE, "w") as f:
        json.dump(data, f)
    log.info(f"Downloaded ATT&CK JSON: {len(data.get('objects', []))} objects")
    return data


def extract_attack_pairs(attack_data: dict) -> list[tuple[str, str]]:
    """
    Extract positive pairs from ATT&CK.

    Strategy: relationships of type "uses" link intrusion-sets/malware to techniques.
    If the same intrusion-set uses technique A and technique B, that's a positive pair.
    Also extract technique descriptions for embedding.
    """
    objects = attack_data.get("objects", [])

    # Build lookup tables
    techniques = {}  # id -> {name, description}
    relationships = []  # (source_ref, target_ref, relationship_type)
    intrusion_uses = {}  # intrusion_id -> [technique_ids]

    for obj in objects:
        obj_type = obj.get("type", "")

        if obj_type == "attack-pattern":
            ext_refs = obj.get("external_references", [])
            attack_id = ""
            for ref in ext_refs:
                if ref.get("source_name") == "mitre-attack":
                    attack_id = ref.get("external_id", "")
                    break
            if not attack_id:
                continue

            name = obj.get("name", "")
            desc = obj.get("description", "")[:300]
            techniques[obj.get("id", "")] = {
                "attack_id": attack_id,
                "name": name,
                "description": desc,
                "text": f"{attack_id} {name}: {desc}",
            }

        elif obj_type == "relationship":
            rel_type = obj.get("relationship_type", "")
            src = obj.get("source_ref", "")
            tgt = obj.get("target_ref", "")
            if rel_type == "uses" and tgt.startswith("attack-pattern"):
                relationships.append((src, tgt, rel_type))

    # Group techniques by intrusion-set/malware
    for src, tgt, _ in relationships:
        if src not in intrusion_uses:
            intrusion_uses[src] = []
        intrusion_uses[src].append(tgt)

    # Build positive pairs: techniques co-used by same actor
    positive_pairs = []
    seen_pairs = set()

    for actor_id, tech_ids in intrusion_uses.items():
        valid_techs = [t for t in tech_ids if t in techniques]
        if len(valid_techs) < 2:
            continue
        # Take pairs (limit to avoid combinatorial explosion)
        for i in range(min(len(valid_techs), 10)):
            for j in range(i + 1, min(len(valid_techs), 10)):
                t_a = techniques[valid_techs[i]]
                t_b = techniques[valid_techs[j]]
                pair_key = tuple(sorted([t_a["attack_id"], t_b["attack_id"]]))
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    positive_pairs.append((t_a["text"], t_b["text"]))

    log.info(f"ATT&CK: extracted {len(positive_pairs)} positive technique pairs from {len(intrusion_uses)} actors")
    return positive_pairs


def load_db_pairs() -> tuple[list[tuple[str, str, float]], list[tuple[str, str, float]]]:
    """Load positive pairs from methodic.db (edges + co_occurrences)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # fact_edges: strong positive signal (curated relationships)
    edges = conn.execute(
        "SELECT source_text, target_text, confidence FROM fact_edges "
        "WHERE relation IN ('enables', 'bypassed_by', 'requires', 'escalates', 'related')"
    ).fetchall()
    edge_pairs = [(e["source_text"], e["target_text"], e["confidence"]) for e in edges]
    log.info(f"DB edges: {len(edge_pairs)} pairs")

    # co_occurrences: moderate positive signal (co-occurred in same session)
    cooccs = conn.execute(
        "SELECT fact_a_text, fact_b_text, count FROM co_occurrences WHERE count >= 1"
    ).fetchall()
    coocc_pairs = [(c["fact_a_text"], c["fact_b_text"], min(c["count"] / 5.0, 1.0)) for c in cooccs]
    log.info(f"DB co-occurrences: {len(coocc_pairs)} pairs")

    conn.close()
    return edge_pairs, coocc_pairs


def build_negative_pairs(all_texts: list[str], num_negatives: int) -> list[tuple[str, str]]:
    """Build negative pairs by random pairing of unrelated texts."""
    negatives = []
    n = len(all_texts)
    if n < 2:
        return negatives

    for _ in range(num_negatives):
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        while j == i:
            j = random.randint(0, n - 1)
        negatives.append((all_texts[i], all_texts[j]))

    return negatives


# ---------------------------------------------------------------------------
# Step 2: Feature engineering & training
# ---------------------------------------------------------------------------

def build_features(
    text_pairs: list[tuple[str, str]],
    labels: list[float],
    model: SentenceTransformer,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build feature vectors from text pairs.

    Features for each pair (a, b):
      - emb_a (384d)
      - emb_b (384d)
      - emb_a * emb_b (element-wise product, 384d) — interaction term
      - |emb_a - emb_b| (absolute difference, 384d)
      - cosine_similarity(a, b) (1d)
    Total: 384*4 + 1 = 1537 features
    """
    all_texts_a = [p[0] for p in text_pairs]
    all_texts_b = [p[1] for p in text_pairs]

    log.info(f"Encoding {len(all_texts_a)} text-A and {len(all_texts_b)} text-B...")

    # Encode all unique texts at once for efficiency
    unique_texts = list(set(all_texts_a + all_texts_b))
    log.info(f"  {len(unique_texts)} unique texts to encode")

    embeddings = model.encode(
        unique_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    text_to_emb = {t: embeddings[i] for i, t in enumerate(unique_texts)}

    features = []
    for a_text, b_text in text_pairs:
        emb_a = text_to_emb[a_text]
        emb_b = text_to_emb[b_text]

        product = emb_a * emb_b
        diff = np.abs(emb_a - emb_b)
        cosine = np.dot(emb_a, emb_b)  # already normalized

        feat = np.concatenate([emb_a, emb_b, product, diff, [cosine]])
        features.append(feat)

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)

    log.info(f"Feature matrix: {X.shape}, labels: {y.shape}")
    return X, y


def train_model(X: np.ndarray, y: np.ndarray) -> Pipeline:
    """Train logistic regression classifier."""
    # Binarize labels for classification (threshold at 0.5)
    y_binary = (y >= 0.5).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )

    log.info(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    log.info(f"  Positive rate (train): {y_train.mean():.3f}")
    log.info(f"  Positive rate (test):  {y_test.mean():.3f}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )),
    ])

    log.info("Training logistic regression...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["not-connected", "connected"]))

    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC AUC: {auc:.4f}")

    accuracy = (y_pred == y_test).mean()
    print(f"Accuracy: {accuracy:.4f}")

    return pipeline


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(42)
    np.random.seed(42)

    log.info("=== Step 1: Gathering training data ===")

    # 1a. MITRE ATT&CK pairs
    try:
        attack_data = fetch_attack_json()
        attack_pairs = extract_attack_pairs(attack_data)
    except Exception as e:
        log.warning(f"Failed to fetch ATT&CK data: {e}")
        attack_pairs = []

    # 1b. Database pairs
    edge_pairs, coocc_pairs = load_db_pairs()

    # Combine positive pairs
    all_positive_texts = []
    all_positive_labels = []

    # ATT&CK pairs: label = 0.8 (strong signal, but not as curated as edges)
    for a, b in attack_pairs:
        all_positive_texts.append((a[:500], b[:500]))
        all_positive_labels.append(1.0)

    # Edge pairs: label = confidence (curated, strongest signal)
    for a, b, conf in edge_pairs:
        all_positive_texts.append((a[:500], b[:500]))
        all_positive_labels.append(1.0)

    # Co-occurrence pairs: label = scaled count (moderate signal)
    for a, b, score in coocc_pairs:
        all_positive_texts.append((a[:500], b[:500]))
        all_positive_labels.append(1.0)

    log.info(f"Total positive pairs: {len(all_positive_texts)}")

    # Build negative pairs (2x negatives for class balance)
    all_unique_texts = list(set(
        [t[0] for t in all_positive_texts] + [t[1] for t in all_positive_texts]
    ))
    num_negatives = len(all_positive_texts)  # 1:1 ratio
    negative_pairs = build_negative_pairs(all_unique_texts, num_negatives)

    all_pairs = all_positive_texts + [(a[:500], b[:500]) for a, b in negative_pairs]
    all_labels = all_positive_labels + [0.0] * len(negative_pairs)

    log.info(f"Total training pairs: {len(all_pairs)} ({sum(1 for l in all_labels if l >= 0.5)} pos, {sum(1 for l in all_labels if l < 0.5)} neg)")

    # Shuffle
    combined = list(zip(all_pairs, all_labels))
    random.shuffle(combined)
    all_pairs, all_labels = zip(*combined)
    all_pairs = list(all_pairs)
    all_labels = list(all_labels)

    log.info("=== Step 2: Building features ===")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    X, y = build_features(all_pairs, all_labels, st_model, batch_size=256)

    log.info("=== Step 3: Training model ===")
    pipeline = train_model(X, y)

    log.info("=== Step 4: Saving model ===")
    model_data = {
        "pipeline": pipeline,
        "feature_dim": X.shape[1],
        "training_samples": X.shape[0],
        "positive_rate": float((y >= 0.5).mean()),
        "trained_at": time.time(),
        "version": "1.0",
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)
    log.info(f"Model saved to {MODEL_PATH} ({os.path.getsize(MODEL_PATH) / 1024:.1f} KB)")

    # Quick sanity check
    log.info("=== Sanity check ===")
    test_pairs = [
        ("SQL injection found", "xp_cmdshell on MSSQL"),
        ("SQL injection found", "DNS zone transfer"),
        ("Docker socket mounted", "container escape via docker API"),
        ("Docker socket mounted", "Kerberoasting TGS tickets"),
    ]
    for a, b in test_pairs:
        emb_a = st_model.encode(a, normalize_embeddings=True)
        emb_b = st_model.encode(b, normalize_embeddings=True)
        feat = np.concatenate([emb_a, emb_b, emb_a * emb_b, np.abs(emb_a - emb_b), [np.dot(emb_a, emb_b)]])
        prob = pipeline.predict_proba(feat.reshape(1, -1))[0][1]
        log.info(f"  P('{a}' -> '{b}') = {prob:.4f}")

    print("\nDone. Model at:", MODEL_PATH)


if __name__ == "__main__":
    main()
