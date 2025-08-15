# app.py
import os, json, time, itertools, functools
from typing import Dict, List, Tuple
import requests
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# -----------------------------
# Config via environment
# -----------------------------
AML_ENDPOINT_URL = os.getenv("AML_ENDPOINT_URL")  # e.g., https://<endpoint>.<region>.inference.ml.azure.com/score
AML_ENDPOINT_KEY = os.getenv("AML_ENDPOINT_KEY")  # Managed online endpoint key
AML_DEPLOYMENT   = os.getenv("AML_DEPLOYMENT")    # optional: target a specific deployment
SEQ_LEN          = int(os.getenv("SEQ_LEN", "50"))
REQUEST_TIMEOUT  = float(os.getenv("REQUEST_TIMEOUT_SECS", "30"))
QVC_API_TMPL     = "https://api.qvc.com/api/sales/presentation/v3/de/products/{sku}?response-depth=summary"

if not AML_ENDPOINT_URL or not AML_ENDPOINT_KEY:
    st.warning("AML endpoint URL/KEY not set. Set env vars AML_ENDPOINT_URL and AML_ENDPOINT_KEY.")
    st.stop()

# -----------------------------
# Simple local assets
# -----------------------------
@functools.lru_cache(maxsize=1)
def load_top_skus() -> List[str]:
    # optional helper file with popular SKUs for type-ahead suggestions
    path_candidates = ["top_skus.json", "./assets/top_skus.json"]
    for p in path_candidates:
        if os.path.exists(p):
            with open(p, "r") as f:
                data = json.load(f)
            # Accept either a list of strings or list of {"sku": "..."}
            if isinstance(data, list):
                if data and isinstance(data[0], dict) and "sku" in data[0]:
                    return [d["sku"] for d in data]
                return [str(x) for x in data]
    return []  # no suggestions

TOP_SKUS = load_top_skus()

# -----------------------------
# Utilities
# -----------------------------
def _softmax(x: np.ndarray, T: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x / max(T, 1e-8)
    x = x - x.max()  # stability
    e = np.exp(x)
    return e / e.sum()

def aml_headers() -> Dict[str, str]:
    h = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AML_ENDPOINT_KEY}",
    }
    if AML_DEPLOYMENT:
        h["azureml-model-deployment"] = AML_DEPLOYMENT
    return h

@st.cache_data(show_spinner=False, ttl=120)
def aml_predict_recent_seq(seq: Tuple[str, ...], k: int) -> Dict:
    """Call AML endpoint with a recent_sequence. Cached per exact sequence."""
    body = {"recent_sequence": list(seq), "k": int(k)}
    r = requests.post(AML_ENDPOINT_URL, headers=aml_headers(), json=body, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False, ttl=300)
def qvc_product(sku: str) -> Dict:
    """Fetch product info from QVC API, with caching."""
    url = QVC_API_TMPL.format(sku=sku)
    r = requests.get(url, timeout=REQUEST_TIMEOUT)
    if r.status_code != 200:
        return {"sku": sku, "error": f"HTTP {r.status_code}"}
    try:
        data = r.json()
    except Exception:
        return {"sku": sku, "error": "Invalid JSON"}
    # Normalize fields we care about
    title = data.get("shortDescription") or data.get("shortDubner") or ""
    brand = data.get("brandName") or ""
    price = None
    reviews = data.get("reviews") or {}
    avg_rating = reviews.get("averageRating")
    count = reviews.get("count")
    pricing = data.get("pricing") or {}
    if "currentMinimumSellingPrice" in pricing:
        price = pricing["currentMinimumSellingPrice"]
    base = (data.get("baseImageUrl") or "").replace("http://", "https://")
    assets = data.get("assets") or []
    img = None
    if assets:
        # e.g., base + "886650.001"
        img = f"{base}{assets[0].get('url')}"
    return {
        "sku": sku,
        "title": title,
        "brand": brand,
        "price": price,
        "avg_rating": avg_rating,
        "review_count": count,
        "image": img,
        "raw": data,
    }

def batched(iterable, n):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

def fetch_products_parallel(skus: List[str]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    with ThreadPoolExecutor(max_workers=min(8, len(skus) or 1)) as ex:
        futs = {ex.submit(qvc_product, s): s for s in skus}
        for f in as_completed(futs):
            s = futs[f]
            try:
                out[s] = f.result()
            except Exception as e:
                out[s] = {"sku": s, "error": str(e)}
    return out

def filter_suggestions(prefix: str, limit: int = 10) -> List[str]:
    if not TOP_SKUS or not prefix:
        return TOP_SKUS[:limit]
    p = prefix.strip().lower()
    hits = [s for s in TOP_SKUS if s.lower().startswith(p)]
    if len(hits) < limit:
        # fallback to contains
        more = [s for s in TOP_SKUS if p in s.lower() and s not in hits]
        hits.extend(more)
    return hits[:limit]

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Next-Product Demo (BERT4Rec)", page_icon="🛒", layout="wide")
st.title("🛒 Next-Product Recommendations — BERT4Rec")

with st.sidebar:
    st.caption("Azure ML endpoint (server-side):")
    st.write(AML_ENDPOINT_URL)
    st.caption("Sequence length cap")
    st.write(SEQ_LEN)
    st.markdown("---")
    st.markdown("**How it works**")
    st.write(
        "- Build a SKU sequence.\n"
        "- Get live next-item suggestions for positions 2..N.\n"
        "- Click **Recommend** to see the Top-10 with details from QVC."
    )

# Session state
if "seq" not in st.session_state:
    st.session_state.seq: List[str] = []
if "last_preds" not in st.session_state:
    st.session_state.last_preds = None  # cache last AML result for the displayed sequence

# --- Sequence builder ---
st.subheader("1) Build a SKU sequence")

colA, colB = st.columns([2, 1])
with colA:
    skuprefix = st.text_input("Type a SKU (or paste)", key="typed_sku", placeholder="e.g., 886650")
    sug = filter_suggestions(skuprefix, limit=10)
    if sug:
        pick = st.selectbox("Quick pick (suggestions)", ["—"] + sug, index=0, key="pick_sku")
    else:
        pick = "—"
with colB:
    add_clicked = st.button("➕ Add SKU")

# Add logic
to_add = None
if add_clicked:
    if st.session_state.pick_sku and st.session_state.pick_sku != "—":
        to_add = st.session_state.pick_sku.strip()
    elif st.session_state.typed_sku:
        to_add = st.session_state.typed_sku.strip()
    if to_add:
        # cap length (left-truncate)
        seq = st.session_state.seq + [to_add]
        if len(seq) > SEQ_LEN:
            seq = seq[-SEQ_LEN:]
        st.session_state.seq = seq
        st.session_state.typed_sku = ""
        st.session_state.pick_sku = "—"
        st.session_state.last_preds = None  # force refresh

# Show current sequence with remove buttons
if st.session_state.seq:
    st.write("Current sequence:")
    chips = st.container()
    cols = st.columns(5)
    for i, sku in enumerate(st.session_state.seq):
        c = cols[i % 5]
        with c:
            st.write(f"`{sku}`")
            if st.button("Remove", key=f"rem_{i}"):
                new_seq = st.session_state.seq[:i] + st.session_state.seq[i+1:]
                st.session_state.seq = new_seq
                st.session_state.last_preds = None
                st.experimental_rerun()
else:
    st.info("Add at least one SKU to begin.")

# --- Live next-sku suggestions for positions 2..N ---
if len(st.session_state.seq) >= 1:
    st.subheader("2) (Optional) Use live suggestions for the next SKU")
    try:
        with st.spinner("Fetching next-item suggestions…"):
            # ask for 20 so dropdown has room; cache on exact sequence tuple
            seq_tuple = tuple(st.session_state.seq)
            preds = aml_predict_recent_seq(seq_tuple, k=20)
            st.session_state.last_preds = preds  # reuse later
        options = preds.get("items", [])[:20]
        col1, col2 = st.columns([2, 1])
        with col1:
            next_pick = st.selectbox("Predicted next SKUs", ["—"] + options, key="next_pick")
        with col2:
            add_next = st.button("Append suggestion")
        if add_next and next_pick and next_pick != "—":
            new_seq = st.session_state.seq + [next_pick]
            if len(new_seq) > SEQ_LEN:
                new_seq = new_seq[-SEQ_LEN:]
            st.session_state.seq = new_seq
            st.session_state.last_preds = None
            st.experimental_rerun()
    except Exception as e:
        st.error(f"Could not fetch suggestions: {e}")

# --- Final recommend ---
st.subheader("3) Get Top-10 recommendations")
colL, colR = st.columns([1, 4])
with colL:
    k = st.slider("How many?", min_value=5, max_value=30, value=10, step=1)
    go = st.button("🎯 Recommend")
with colR:
    if go:
        if not st.session_state.seq:
            st.warning("Please add at least one SKU.")
        else:
            try:
                with st.spinner("Scoring…"):
                    seq_tuple = tuple(st.session_state.seq)
                    preds = aml_predict_recent_seq(seq_tuple, k=int(k))
                items = preds.get("items", [])[:k]
                scores = preds.get("scores", [])[:k]
                if not items:
                    st.warning(preds.get("note") or "No results.")
                else:
                    # Softmax over top-K only for relative bars (make it explicit)
                    probs = _softmax(np.array(scores)) if len(scores) == len(items) else np.ones(len(items))/len(items)
                    meta = fetch_products_parallel(items)

                    st.caption("Top-K are ranked by model score (higher is better). Bars below are **relative** softmax over Top-K, not calibrated probabilities.")
                    for chunk in batched(list(zip(items, scores, probs)), 5):  # 5 cards per row
                        cols = st.columns(len(chunk))
                        for (sku, sc, pb), c in zip(chunk, cols):
                            with c:
                                m = meta.get(sku) or {}
                                img = m.get("image")
                                title = m.get("title") or "(no title)"
                                brand = m.get("brand") or ""
                                price = m.get("price")
                                rating = m.get("avg_rating")
                                count = m.get("review_count")
                                if img:
                                    st.image(img, use_container_width=True)
                                st.markdown(f"**{title}**")
                                if brand:
                                    st.caption(brand)
                                sub = []
                                if price is not None:
                                    sub.append(f"€{price:,.2f}")
                                if rating is not None:
                                    sub.append(f"★ {rating:.2f} ({count or 0})")
                                if sub:
                                    st.write(" · ".join(sub))
                                st.progress(float(pb))
                                st.caption(f"SKU `{sku}` · score {sc:.3f}")
                                if st.button("Append to sequence", key=f"add_{sku}"):
                                    new_seq = st.session_state.seq + [sku]
                                    if len(new_seq) > SEQ_LEN:
                                        new_seq = new_seq[-SEQ_LEN:]
                                    st.session_state.seq = new_seq
                                    st.session_state.last_preds = None
                                    st.experimental_rerun()
            except requests.Timeout:
                st.error("Timed out calling the AML endpoint. Try again.")
            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")
st.caption("Demo: BERT4Rec on Azure ML · Streamlit frontend · QVC product enrichment")
