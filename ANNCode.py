"""
Bandwidth-first ANN refinement — artifact implementation

This file collects:
  1) Data loading for .vec / ASCII-header binary / .fvecs
  2) Reduced-precision transforms for FP16 (bit-accurate field ops)
  3) Cosine and Euclidean top-K refinement with early rejection
  4) Bandwidth accounting (reduced pass + exact fetches)
  5) Optional: lossless compression model (bit-plane pack + zlib)
  6) Optional: experiment sweeps + simple plots

Design notes:
  • “Reduced” vectors are produced by truncating FP16 mantissa bits
    (and optionally exponent bits); we never peek at full precision
    when constructing bounds. The “Δ step” comes from kept exponent bits.
  • Cosine runs in unit space; Euclidean runs in value space.
"""

from __future__ import annotations
import io, os, struct, math, argparse, zlib
from typing import Tuple, List, Optional, Sequence
import numpy as np

try:
    import matplotlib.pyplot as plt  # noqa
    _HAS_PLT = True
except Exception:
    _HAS_PLT = False


# FP16 constants (IEEE 754 half)
#   layout: [sign:1][exp:5][mant:10]
FP16_EXP_BITS  = 5
FP16_EXP_BIAS  = 15
FP16_MAN_BITS  = 10



# Small utilities
def _rng():
    """Fixed RNG for deterministic demos/tests."""
    return np.random.default_rng(42)


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Row-wise L2 normalization with a small floor.
    Cosine modes assume unit vectors; we keep Euclidean in value space.
    """
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


# Data loaders
#   Supported:
#     • FastText .vec (first line: N D; subsequent lines: token + D floats)
#     • Binary with ASCII header lines (L, D) followed by L*D float32
#     • .fvecs (FAISS/SIFT style): [dim:int32][dim*float32] repeated
def load_vec(path: str, max_rows: int = 100_000) -> np.ndarray:
    """Load FastText .vec; subsample uniformly if N > max_rows."""
    with io.open(path, "r", encoding="utf-8", newline="\n", errors="ignore") as f:
        hdr = f.readline().strip().split()
        if len(hdr) < 2 or not hdr[0].isdigit() or not hdr[1].isdigit():
            raise ValueError("invalid .vec header")
        N, D = int(hdr[0]), int(hdr[1])

        take = min(N, max_rows)
        sel = None
        if N > max_rows:
            sel = sorted(_rng().choice(N, size=take, replace=False).tolist())
        sel_set = set(sel) if sel is not None else None

        X = np.zeros((take, D), dtype=np.float32)
        w = 0
        for i, line in enumerate(f):
            if w >= take:
                break
            if sel_set is not None and i not in sel_set:
                continue
            parts = line.rstrip("\n").split()
            if len(parts) < D + 1:
                continue
            arr = np.fromstring(" ".join(parts[1:]), sep=" ", dtype=np.float32)
            if arr.size != D:  # last line in some files can be short
                arr = np.array([float(x) for x in parts[1:D+1]], dtype=np.float32)
            X[w] = arr
            w += 1
        if w != take:
            X = X[:w].copy()
    return X


def load_bin_header_body(path: str, max_rows: int = 100_000) -> np.ndarray:
    """
    Binary: first two lines are ASCII integers (L, D),
    followed by L*D float32 values.
    """
    with open(path, "rb") as f:
        l1 = f.readline().strip(); l2 = f.readline().strip()
        L = int(l1.decode("ascii")); D = int(l2.decode("ascii"))
        buf = np.fromfile(f, dtype=np.float32, count=L * D)
        if buf.size != L * D:
            raise ValueError("truncated payload")
    X = buf.reshape(L, D)
    if L > max_rows:
        idx = _rng().choice(L, size=max_rows, replace=False)
        X = X[idx]
    return X.astype(np.float32, copy=False)


def load_fvecs(path: str, max_rows: int = 100_000) -> np.ndarray:
    """
    FAISS/SIFT .fvecs: each record starts with int32 dim, then 'dim' float32s.
    This loader assumes a fixed dimension; common for SIFT/GIST dumps.
    """
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 4:
        raise ValueError("bad fvecs")
    D = struct.unpack_from("<i", data, 0)[0]
    rec = 4 + 4 * D
    if (len(data) % rec) != 0:
        raise ValueError("size mismatch")
    N = len(data) // rec

    out = np.empty((min(N, max_rows), D), dtype=np.float32)
    off, w = 0, 0
    for _ in range(N):
        if w >= out.shape[0]:
            break
        off += 4  # skip dim
        arr = np.frombuffer(data, dtype=np.float32, count=D, offset=off)
        out[w] = arr
        w += 1
        off += 4 * D
    return out


def load_matrix(path: str, max_rows: int = 100_000) -> np.ndarray:
    """Unified entry: route to .vec / .fvecs / ascii-header binary as needed."""
    pl = path.lower()
    if pl.endswith(".vec"):
        return load_vec(path, max_rows)
    if pl.endswith(".fvecs"):
        return load_fvecs(path, max_rows)
    try:
        return load_bin_header_body(path, max_rows)
    except Exception:
        with open(path, "rb") as f:
            L = struct.unpack("<i", f.read(4))[0]
            D = struct.unpack("<i", f.read(4))[0]
            buf = np.fromfile(f, dtype=np.float32, count=L * D)
        X = buf.reshape(L, D)
        if L > max_rows:
            idx = _rng().choice(L, size=max_rows, replace=False)
            X = X[idx]
        return X.astype(np.float32, copy=False)



# FP16 helpers (bit-accurate)
#   We operate on raw 16-bit words to control mantissa/exponent bits
#   without relying on a platform's rounding behavior.
def _f16_to_u16(x16: np.ndarray) -> np.ndarray:
    return x16.view(np.uint16)


def _u16_to_f16(u16: np.ndarray) -> np.ndarray:
    return u16.view(np.float16)


def _split_fields(u16: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (sign, exp, mantissa) arrays (dtype=uint16)."""
    s = (u16 >> 15) & 0x1
    e = (u16 >> 10) & 0x1F
    m = u16 & 0x03FF
    return s, e, m


def _combine_fields(s: np.ndarray, e: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Pack (sign, exp, mantissa) into 16-bit FP16 words."""
    return ((s & 1) << 15) | ((e & 0x1F) << 10) | (m & 0x03FF)


def _truncate_mantissa(mant: np.ndarray, keep_m: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Keep the top 'keep_m' mantissa bits, zero out the rest.
    We intentionally do *not* round here to keep conservative errors.
    The returned 'carry' is zeros (no rounding).
    """
    assert 0 <= keep_m <= FP16_MAN_BITS
    if keep_m == FP16_MAN_BITS:
        return mant.copy(), np.zeros_like(mant, dtype=np.uint16)
    drop = FP16_MAN_BITS - keep_m
    kept = mant >> drop
    mant2 = (kept << drop).astype(np.uint16)
    carry = np.zeros_like(mant2, dtype=np.uint16)
    return mant2, carry


def reduce_precision_keep_exp(x: np.ndarray, keep_m: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce precision by truncating mantissa; keep the full exponent.
    Returns:
      reduced -> float32 view of FP16 with truncated mantissa
      kept_exp_bits -> uint8 array with original exponent field (used to get Δ)
    """
    x16 = x.astype(np.float16)
    u = _f16_to_u16(x16)
    s, e, m = _split_fields(u)
    m2, carry = _truncate_mantissa(m, keep_m)
    e2 = (e + carry).astype(np.uint16) # carry is zero in truncation path
    u2 = _combine_fields(s, e2, m2)
    xr = _u16_to_f16(u2)
    return xr.astype(np.float32), e2.astype(np.uint8)


def reduce_precision_keep_both(x: np.ndarray, keep_e: int, keep_m: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Truncate both exponent and mantissa.
    Returns:
      reduced -> float32 view,
      kept_exp_bits -> uint8,
      abs_error_vs_fp16 -> float32 (for diagnostics only).
    """
    assert 0 <= keep_e <= FP16_EXP_BITS
    assert 0 <= keep_m <= FP16_MAN_BITS

    x16 = x.astype(np.float16)
    u16 = _f16_to_u16(x16)
    s, e, m = _split_fields(u16)
    m2, carry = _truncate_mantissa(m, keep_m)
    e2 = (e + carry).astype(np.uint16)
    if keep_e != FP16_EXP_BITS:
        drop = FP16_EXP_BITS - keep_e
        e2 = ((e2 >> drop) << drop).astype(np.uint16)

    u2 = _combine_fields(s, e2, m2)
    xr = _u16_to_f16(u2)
    abs_err = np.abs(x16.astype(np.float32) - xr.astype(np.float32)).astype(np.float32)
    return xr.astype(np.float32), e2.astype(np.uint8), abs_err


def delta_from_exp(kept_exp_bits: np.ndarray, keep_m: int) -> np.ndarray:
    """
    “No-peek” absolute step Δ used in conservative bounds.
    For normals: Δ = 2^(e - bias - keep_m)
    Subnormals:  2^(1 - bias - keep_m)
    """
    e = kept_exp_bits.astype(np.int32)
    is_sub = (e == 0)
    step = np.where(
        is_sub,
        2.0 ** (1 - FP16_EXP_BIAS - keep_m),
        2.0 ** (e - FP16_EXP_BIAS - keep_m),
    )
    return step.astype(np.float32)


# Exact top-K (ground truth)
#   • Cosine in unit space
#   • Euclidean in value space
def exact_topk_cosine(DB_unit: np.ndarray, Q_unit: np.ndarray, k: int) -> List[np.ndarray]:
    out = []
    for q in Q_unit:
        s = DB_unit @ q
        idx = np.argpartition(s, -k)[-k:]
        idx = idx[np.argsort(s[idx])[::-1]]
        out.append(idx)
    return out


def exact_topk_l2(DB: np.ndarray, Q: np.ndarray, k: int) -> List[np.ndarray]:
    out = []
    for q in Q:
        d2 = np.sum((DB - q[None, :]) ** 2, axis=1)
        idx = np.argpartition(d2, k)[:k]
        idx = idx[np.argsort(d2[idx])]
        out.append(idx)
    return out


# Small array-heap for top-K (avoids Python heap overhead)
def _heap_insert(scores: np.ndarray, ids: np.ndarray, s: float, i: int):
    """
    Maintain K best scores (descending) in two parallel arrays:
      scores: length K (init with -inf)
      ids:    length K (init with -1)
    """
    K = scores.size
    if np.isneginf(scores).any():
        pos = np.where(np.isneginf(scores))[0][0]
        scores[pos] = s; ids[pos] = i
    else:
        worst = np.argmin(scores)
        if s > scores[worst]:
            scores[worst] = s; ids[worst] = i


# Cosine refinement
#   ch  : Hoeffding-style early reject using delta-derived norms
#   cl1 : Deterministic l1 cushion
#   cl2 : Deterministic l2 cushion using ||unit(DB_full) - unit(DB_reduced)||
#   All account bandwidth: reduced pass per-candidate; full fetch on survivors.
#   If 'red_bytes_override' is provided (compression), it overrides reduced bytes.
def refine_cos_hoeff(DBu: np.ndarray, DBru: np.ndarray, kept_exp: np.ndarray, Q: np.ndarray, k: int,
                     delta: float, keep_m: int, *, keep_e: int,
                     red_bytes_override: Optional[float] = None):
    assert 0.0 < delta < 1.0
    N, D = DBu.shape
    Qn = l2_normalize_rows(Q)
    Delta = delta_from_exp(kept_exp, keep_m)

    # bytes per vector for accounting (full vs reduced)
    bits_full, bits_red = 16, (1 + keep_e + keep_m)
    b_full = (bits_full * D) / 8.0
    b_red  = red_bytes_override if red_bytes_override is not None else (bits_red * D) / 8.0
    baseline = b_full * N

    const = math.sqrt(2.0 * math.log(1.0 / float(delta)))

    preds: List[np.ndarray] = []
    total = 0.0 # total bytes consumed
    cand_total = 0   # candidates seen (for FPR denominator)
    unnecessary = 0 # fetched full precision but didn't make final top-K

    for q in Qn:
        cand_total += N
        # K-heap in score space (cosine similarity)
        heap_s = np.full(k, -np.inf, dtype=np.float32)
        heap_i = np.full(k, -1, dtype=np.int32)

        # reduced pass for all candidates
        total += b_red * N
        s_hat = DBru @ q

        # seed with best K under reduced score
        seed = np.argpartition(s_hat, -k)[-k:]
        fetched = set(int(i) for i in seed)
        for i in seed:
            sx = float(np.dot(DBu[i], q))
            _heap_insert(heap_s, heap_i, sx, int(i))
        worst_s = float(np.min(heap_s))
        total  += b_full * len(seed)

        # process remainder in descending reduced score
        mask = np.ones(N, dtype=bool); mask[seed] = False
        order = np.argsort(s_hat[mask])[::-1]
        rest  = np.where(mask)[0][order]

        for i in rest:
            # probabilistic cushion
            t = const * float(np.linalg.norm(q * Delta[i]))
            if float(s_hat[i]) + t < worst_s:
                continue  # reject without full fetch
            sx = float(np.dot(DBu[i], q))
            fetched.add(int(i))
            if (sx > worst_s) or np.isneginf(heap_s).any():
                _heap_insert(heap_s, heap_i, sx, int(i))
                worst_s = float(np.min(heap_s))
            total += b_full

        final = set(int(x) for x in heap_i.tolist())
        unnecessary += len(fetched - final)
        preds.append(heap_i[np.argsort(heap_s)[::-1]])

    saving = 1.0 - (total / (baseline * len(Qn)))
    fpr = (unnecessary / cand_total) if cand_total else 0.0
    return preds, saving, fpr


def refine_cos_l1(DBu: np.ndarray, DBru: np.ndarray, kept_exp: np.ndarray, Q: np.ndarray, k: int,
                  keep_m: int, *, alpha: float = 1.0, red_bytes_override: Optional[float] = None):
    N, D = DBu.shape
    Qn = l2_normalize_rows(Q)
    Delta = delta_from_exp(kept_exp, keep_m)

    bits_full, bits_red = 16, (1 + FP16_EXP_BITS + keep_m)
    b_full = (bits_full * D) / 8.0
    b_red  = red_bytes_override if red_bytes_override is not None else (bits_red * D) / 8.0
    baseline = b_full * N

    preds: List[np.ndarray] = []
    total = 0.0
    cand_total = 0
    unnecessary = 0

    for q in Qn:
        cand_total += N
        aq = np.abs(q).astype(np.float32)
        heap_s = np.full(k, -np.inf, dtype=np.float32)
        heap_i = np.full(k, -1, dtype=np.int32)

        total += b_red * N
        s_hat = DBru @ q

        seed = np.argpartition(s_hat, -k)[-k:]
        fetched = set(int(i) for i in seed)
        for i in seed:
            sx = float(np.dot(DBu[i], q))
            _heap_insert(heap_s, heap_i, sx, int(i))
        worst_s = float(np.min(heap_s))
        total  += b_full * len(seed)

        mask = np.ones(N, dtype=bool); mask[seed] = False
        order = np.argsort(s_hat[mask])[::-1]
        rest  = np.where(mask)[0][order]

        for i in rest:
            # deterministic ℓ1 cushion
            b = float(np.dot(aq, Delta[i]))
            if float(s_hat[i]) + alpha * b < worst_s:
                continue
            sx = float(np.dot(DBu[i], q))
            fetched.add(int(i))
            if (sx > worst_s) or np.isneginf(heap_s).any():
                _heap_insert(heap_s, heap_i, sx, int(i))
                worst_s = float(np.min(heap_s))
            total += b_full

        final = set(int(x) for x in heap_i.tolist())
        unnecessary += len(fetched - final)
        preds.append(heap_i[np.argsort(heap_s)[::-1]])

    saving = 1.0 - (total / (baseline * len(Qn)))
    fpr = (unnecessary / cand_total) if cand_total else 0.0
    return preds, saving, fpr


def refine_cos_l2(DB_full: np.ndarray, DB_red: np.ndarray, kept_exp: np.ndarray, Q: np.ndarray, k: int,
                  keep_m: int, *, keep_e: int = FP16_EXP_BITS, alpha: float = 1.0,
                  red_bytes_override: Optional[float] = None):
    """
    ℓ2 cushion in unit space: ||unit(DB_full) - unit(DB_reduced)||
    This produces a single per-row constant bound that can be multiplied by α.
    """
    N, D = DB_full.shape
    DBu  = l2_normalize_rows(DB_full)
    DBru = l2_normalize_rows(DB_red)
    Qn   = l2_normalize_rows(Q)

    Delta_u = np.abs(DBu - DBru).astype(np.float32)
    B = np.sqrt(np.sum(Delta_u * Delta_u, axis=1)).astype(np.float32)

    bits_full, bits_red = 16, (1 + keep_e + keep_m)
    b_full = (bits_full * D) / 8.0
    b_red  = red_bytes_override if red_bytes_override is not None else (bits_red * D) / 8.0
    baseline = b_full * N

    preds: List[np.ndarray] = []
    total = 0.0
    cand_total = 0
    unnecessary = 0

    for q in Qn:
        cand_total += N
        heap_s = np.full(k, -np.inf, dtype=np.float32)
        heap_i = np.full(k, -1, dtype=np.int32)

        total += b_red * N
        s_hat = DBru @ q

        seed = np.argpartition(s_hat, -k)[-k:]
        fetched = set(int(i) for i in seed)
        for i in seed:
            sx = float(np.dot(DBu[i], q))
            _heap_insert(heap_s, heap_i, sx, int(i))
        worst_s = float(np.min(heap_s))
        total  += b_full * len(seed)

        mask = np.ones(N, dtype=bool); mask[seed] = False
        order = np.argsort(s_hat[mask])[::-1]
        rest  = np.where(mask)[0][order]

        for i in rest:
            if float(s_hat[i]) + alpha * float(B[i]) < worst_s:
                continue
            sx = float(np.dot(DBu[i], q))
            fetched.add(int(i))
            if (sx > worst_s) or np.isneginf(heap_s).any():
                _heap_insert(heap_s, heap_i, sx, int(i))
                worst_s = float(np.min(heap_s))
            total += b_full

        final = set(int(x) for x in heap_i.tolist())
        unnecessary += len(fetched - final)
        preds.append(heap_i[np.argsort(heap_s)[::-1]])

    saving = 1.0 - (total / (baseline * len(Qn)))
    fpr = (unnecessary / cand_total) if cand_total else 0.0
    return preds, saving, fpr


# Euclidean refinement
#   e1: coordinate-wise deterministic lower bounds
#       • mode="tz": sign-aware
#       • mode="sym": symmetric interval
#   e2: Hoeffding lower bound for L2
def _lb_sym(a: np.ndarray, Delta: np.ndarray) -> float:
    """
    Symmetric interval bound on ||a + ε||^2 where |ε_j| ≤ Δ_j.
    """
    lower = np.maximum(np.abs(a) - Delta, 0.0).astype(np.float32)
    return float(np.sum(lower * lower, dtype=np.float32))


def _lb_tz(a: np.ndarray, Delta: np.ndarray, x_tilde: np.ndarray) -> float:
    """
    Sign-aware bound: if the reduced coordinate suggests a direction,
    exploit that to tighten the lower bound (matches tz variant in paper).
    """
    s = np.where(np.signbit(x_tilde), -1.0, 1.0).astype(np.float32)
    b = a * s
    m1 = (b <= 0.0)
    m2 = (b > 0.0) & (b < Delta)
    m3 = ~(m1 | m2)
    out = np.empty_like(a, dtype=np.float32)
    out[m1] = a[m1] * a[m1]
    out[m2] = 0.0
    out[m3] = (np.abs(a[m3]) - Delta[m3]) ** 2
    return float(np.sum(out, dtype=np.float32))


def refine_l2_det(DB_full: np.ndarray, DB_red: np.ndarray, kept_exp: np.ndarray, Q: np.ndarray, k: int,
                  keep_m: int, *, keep_e: int, mode: str = "tz",
                  red_bytes_override: Optional[float] = None):
    assert mode in ("tz", "sym")
    N, D = DB_full.shape
    Delta = delta_from_exp(kept_exp, keep_m).astype(np.float32)

    bits_full, bits_red = 16, (1 + keep_e + keep_m)
    b_full = (bits_full * D) / 8.0
    b_red  = red_bytes_override if red_bytes_override is not None else (bits_red * D) / 8.0
    baseline = b_full * N

    preds: List[np.ndarray] = []
    total = 0.0
    cand_total = 0
    unnecessary = 0

    for q in Q:
        cand_total += N
        total += b_red * N
        # Reduced-space distances
        A = (q[None, :] - DB_red).astype(np.float32)
        d2_hat = np.einsum("ij,ij->i", A, A)

        # Seed top-K in reduced metric
        seed = np.argpartition(d2_hat, k)[:k]
        fetched = set(int(i) for i in seed)
        heap_d = np.full(k, np.inf, dtype=np.float32)
        heap_i = np.full(k, -1, dtype=np.int32)
        for i in seed:
            diff = (q - DB_full[i]); d2 = float(np.dot(diff, diff))
            worst = np.argmax(heap_d)
            if d2 < heap_d[worst] or np.isinf(heap_d).any():
                heap_d[worst] = d2; heap_i[worst] = int(i)
        worst_d = float(np.max(heap_d))
        total  += b_full * len(seed)

        # Remaining candidates in ascending reduced distance
        mask = np.ones(N, dtype=bool); mask[seed] = False
        order = np.argsort(d2_hat[mask])
        rest  = np.where(mask)[0][order]

        for i in rest:
            a = A[i]
            lower = _lb_tz(a, Delta[i], DB_red[i]) if mode == "tz" else _lb_sym(a, Delta[i])
            if lower > worst_d:
                continue
            diff = (q - DB_full[i]); d2 = float(np.dot(diff, diff))
            fetched.add(int(i))
            worst = np.argmax(heap_d)
            if d2 < heap_d[worst] or np.isinf(heap_d).any():
                heap_d[worst] = d2; heap_i[worst] = int(i); worst_d = float(np.max(heap_d))
            total += b_full

        final = set(int(x) for x in heap_i.tolist())
        unnecessary += len(fetched - final)
        preds.append(heap_i[np.argsort(heap_d)])

    saving = 1.0 - (total / (baseline * len(Q)))
    fpr = (unnecessary / cand_total) if cand_total else 0.0
    return preds, saving, fpr


def refine_l2_hoeff(DB_full: np.ndarray, DB_red: np.ndarray, kept_exp: np.ndarray, Q: np.ndarray, k: int,
                    delta: float, keep_m: int, *, keep_e: int,
                    red_bytes_override: Optional[float] = None):
    assert 0.0 < delta < 1.0
    N, D = DB_full.shape
    Delta = delta_from_exp(kept_exp, keep_m).astype(np.float32)

    bits_full, bits_red = 16, (1 + keep_e + keep_m)
    b_full = (bits_full * D) / 8.0
    b_red  = red_bytes_override if red_bytes_override is not None else (bits_red * D) / 8.0
    baseline = b_full * N

    preds: List[np.ndarray] = []
    total = 0.0
    cand_total = 0
    unnecessary = 0
    const = math.sqrt(2.0 * math.log(1.0 / float(delta)))

    for q in Q:
        cand_total += N
        total += b_red * N
        A = (q[None, :] - DB_red).astype(np.float32)
        d2_hat = np.einsum("ij,ij->i", A, A)

        seed = np.argpartition(d2_hat, k)[:k]
        fetched = set(int(i) for i in seed)
        heap_d = np.full(k, np.inf, dtype=np.float32)
        heap_i = np.full(k, -1, dtype=np.int32)
        for i in seed:
            diff = (q - DB_full[i]); d2 = float(np.dot(diff, diff))
            worst = np.argmax(heap_d)
            if d2 < heap_d[worst] or np.isinf(heap_d).any():
                heap_d[worst] = d2; heap_i[worst] = int(i)
        worst_d = float(np.max(heap_d))
        total  += b_full * len(seed)

        mask = np.ones(N, dtype=bool); mask[seed] = False
        order = np.argsort(d2_hat[mask])
        rest  = np.where(mask)[0][order]

        for i in rest:
            a = A[i]
            # Hoeffding-style lower bound in L2
            lower = float(np.dot(a, a)) - 2.0 * const * float(np.linalg.norm(a * Delta[i]))
            if lower > worst_d:
                continue
            diff = (q - DB_full[i]); d2 = float(np.dot(diff, diff))
            fetched.add(int(i))
            worst = np.argmax(heap_d)
            if d2 < heap_d[worst] or np.isinf(heap_d).any():
                heap_d[worst] = d2; heap_i[worst] = int(i); worst_d = float(np.max(heap_d))
            total += b_full

        final = set(int(x) for x in heap_i.tolist())
        unnecessary += len(fetched - final)
        preds.append(heap_i[np.argsort(heap_d)])

    saving = 1.0 - (total / (baseline * len(Q)))
    fpr = (unnecessary / cand_total) if cand_total else 0.0
    return preds, saving, fpr


# Evaluation helpers
def recall_at_k(ground: List[np.ndarray], preds: List[np.ndarray]) -> float:
    """
    Mean Recall@K over queries:
      |GT ∩ Pred| / K, averaged across the evaluation set.
    """
    K = ground[0].size
    s = 0.0
    for g, p in zip(ground, preds):
        s += len(set(g.tolist()) & set(p.tolist())) / K
    return s / len(ground)


# Bit-plane packing + lossless compression model
#   We bit-slice the kept planes (sign + top exponent planes + top mantissa
#   planes) across all scalars, pack into bytes, and run zlib.
#   The returned value is the average compressed bytes per vector, which can
#   be plugged into bandwidth accounting in lieu of the naive bit count.
def _pack_bitplanes(u16: np.ndarray, keep_e: int, keep_m: int) -> bytes:
    s, e, m = _split_fields(u16)
    scalars = u16.size
    planes: List[np.ndarray] = []

    # sign plane
    planes.append((s.reshape(-1) & 1).astype(np.uint8))

    # exponent MSB→LSB (keep_e planes)
    for b in range(keep_e):
        shift = FP16_EXP_BITS - 1 - b
        planes.append(((e.reshape(-1) >> shift) & 1).astype(np.uint8))

    # mantissa MSB→LSB (keep_m planes)
    for b in range(keep_m):
        shift = FP16_MAN_BITS - 1 - b
        planes.append(((m.reshape(-1) >> shift) & 1).astype(np.uint8))

    if not planes:
        return b""
    M = np.vstack(planes)   # [num_planes, scalars]
    packed = np.packbits(M, axis=1) # pack along scalar axis
    return packed.tobytes()


def avg_reduced_bytes_per_vec(DB_full: np.ndarray, keep_e: int, keep_m: int, level: int = 6) -> float:
    """
    Average compressed bytes per vector for the reduced representation
    at the given zlib level (0..9). The higher the level, the slower but
    usually smaller. For apples-to-apples, keep a fixed level across runs.
    """
    u16 = _f16_to_u16(DB_full.astype(np.float16))
    blob = _pack_bitplanes(u16, keep_e, keep_m)
    if not blob:
        return 0.0
    comp = zlib.compress(blob, level=max(0, min(9, level)))
    return len(comp) / float(DB_full.shape[0])


# Plotting / sweeps
#   Produces two simple figures:
#     • recall vs bandwidth savings
#     • FPR vs mantissa bits removed
#   and a CSV with (mode, m_keep, delta, recall, save, fpr).
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _plot_xy(path: str, xs: Sequence[float], ys: Sequence[float],
             xlabel: str, ylabel: str, title: str):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def sweep_and_plot(DB: np.ndarray, Q: np.ndarray, mode: str, k: int,
                   keep_e: int, m_keeps: Sequence[int], deltas: Sequence[float],
                   outdir: str, compress_level: Optional[int] = None):
    _ensure_dir(outdir)
    csv_path = os.path.join(outdir, f"{mode}_sweep.csv")
    with open(csv_path, "w") as f:
        f.write("mode,m_keep,delta,recall,save,fpr\n")

    DBf = DB.astype(np.float16).astype(np.float32)
    DBu = l2_normalize_rows(DBf)

    recall_pts, save_pts, fpr_pts, removed_bits = [], [], [], []

    for m in m_keeps:
        DBr, eb = reduce_precision_keep_exp(DB, m)
        DBru = l2_normalize_rows(DBr)

        red_override = None
        if compress_level is not None:
            red_override = avg_reduced_bytes_per_vec(DB, keep_e, m, level=compress_level)

        # ground truth depends on metric
        gt = exact_topk_cosine(DBu, l2_normalize_rows(Q), k) if mode.startswith("cos-") \
             else exact_topk_l2(DBf, Q, k)

        # For Hoeffding modes we sweep delta; deterministic run once
        deltas_in = deltas if ("hoeff" in mode) else [None]
        for d in deltas_in:
            if mode == "cos-hoeff":
                pred, sv, fp = refine_cos_hoeff(DBu, DBru, eb, Q, k, d, m, keep_e=keep_e,
                                                red_bytes_override=red_override)
            elif mode == "cos-l1":
                pred, sv, fp = refine_cos_l1(DBu, DBru, eb, Q, k, m,
                                             red_bytes_override=red_override)
            elif mode == "cos-l2":
                pred, sv, fp = refine_cos_l2(DB, DBr, eb, Q, k, m, keep_e=keep_e,
                                             red_bytes_override=red_override)
            elif mode == "l2-tz":
                pred, sv, fp = refine_l2_det(DBf, DBr, eb, Q, k, m, keep_e=keep_e, mode="tz",
                                             red_bytes_override=red_override)
            elif mode == "l2-sym":
                pred, sv, fp = refine_l2_det(DBf, DBr, eb, Q, k, m, keep_e=keep_e, mode="sym",
                                             red_bytes_override=red_override)
            elif mode == "l2-hoeff":
                pred, sv, fp = refine_l2_hoeff(DBf, DBr, eb, Q, k, d, m, keep_e=keep_e,
                                               red_bytes_override=red_override)
            else:
                raise ValueError(f"unknown mode {mode}")

            R = recall_at_k(gt, pred)
            with open(csv_path, "a") as f:
                f.write(f"{mode},{m},{0 if d is None else d},{R:.6f},{sv:.6f},{fp:.6f}\n")

            recall_pts.append(R); save_pts.append(sv); fpr_pts.append(fp); removed_bits.append(FP16_MAN_BITS - m)

    _plot_xy(os.path.join(outdir, f"{mode}_recall_vs_save.png"),
             save_pts, recall_pts, "Bandwidth saving", f"Recall@{k}", f"{mode}: recall vs saving")

    _plot_xy(os.path.join(outdir, f"{mode}_fpr_vs_mbits.png"),
             removed_bits, fpr_pts, "Mantissa bits removed", "FPR per candidate",
             f"{mode}: FPR vs mantissa removal")


# Command line
def main():
    ap = argparse.ArgumentParser(description="Bandwidth-first ANN refinement (artifact)")
    ap.add_argument("--mode", choices=["cos-hoeff","cos-l1","cos-l2","l2-tz","l2-sym","l2-hoeff"], default="cos-hoeff")
    ap.add_argument("--db", required=False, help="Path to DB matrix (.fvecs, .vec, or ASCII-header binary).")
    ap.add_argument("--q",  required=False, help="Path to Query matrix (same formats).")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--keep", type=int, default=6, help="Mantissa bits to keep (0..10).")
    ap.add_argument("--ekeep", type=int, default=FP16_EXP_BITS, help="Exponent bits to keep (0..5).")
    ap.add_argument("--delta", type=float, default=0.7, help="Hoeffding δ for *-hoeff modes.")
    ap.add_argument("--maxrows", type=int, default=100000)

    # Compression model: override reduced-pass bytes with compressed bytes/vec
    ap.add_argument("--compress_level", type=int, default=-1,
                    help="zlib level (0..9) for reduced-pass bit-planes; <0 disables.")

    # Optional plots/sweeps
    ap.add_argument("--plot_out", type=str, default=None, help="Directory to write CSV and PNGs.")
    ap.add_argument("--sweep_m", type=str, default=None, help="Comma list of mantissa bits, e.g. 10,8,6,4")
    ap.add_argument("--sweep_delta", type=str, default=None, help="Comma list of deltas for Hoeffding, e.g. 0.7,0.5,0.3")

    args = ap.parse_args()

    # Load data or synthesize a small demo if paths are not provided
    if args.db is None or args.q is None:
        N, D, Qn = 2000, 128, 30
        DB = _rng().normal(size=(N, D)).astype(np.float32)
        Q  = _rng().normal(size=(Qn, D)).astype(np.float32)
    else:
        DB = load_matrix(args.db, args.maxrows)
        Q  = load_matrix(args.q,  args.maxrows)

    # Reduced copy + exp bits used by most paths
    DBf = DB.astype(np.float16).astype(np.float32)
    DBu = l2_normalize_rows(DBf)
    DBr, kept_exp = reduce_precision_keep_exp(DB, args.keep)
    DBru = l2_normalize_rows(DBr)

    # Optional compression model for reduced pass
    red_bytes_override = None
    if args.compress_level is not None and args.compress_level >= 0:
        red_bytes_override = avg_reduced_bytes_per_vec(DB, args.ekeep, args.keep, level=args.compress_level)

    # Ground truth (for reporting recall)
    if args.mode.startswith("cos-"):
        gt = exact_topk_cosine(DBu, l2_normalize_rows(Q), args.k)
    else:
        gt = exact_topk_l2(DBf, Q, args.k)

    # Run selected refinement
    if args.mode == "cos-hoeff":
        preds, saving, fpr = refine_cos_hoeff(DBu, DBru, kept_exp, Q, args.k, args.delta, args.keep,
                                              keep_e=args.ekeep, red_bytes_override=red_bytes_override)
    elif args.mode == "cos-l1":
        preds, saving, fpr = refine_cos_l1(DBu, DBru, kept_exp, Q, args.k, args.keep,
                                           red_bytes_override=red_bytes_override)
    elif args.mode == "cos-l2":
        preds, saving, fpr = refine_cos_l2(DB, DBr, kept_exp, Q, args.k, args.keep, keep_e=args.ekeep,
                                           red_bytes_override=red_bytes_override)
    elif args.mode == "l2-tz":
        preds, saving, fpr = refine_l2_det(DBf, DBr, kept_exp, Q, args.k, args.keep, keep_e=args.ekeep, mode="tz",
                                           red_bytes_override=red_bytes_override)
    elif args.mode == "l2-sym":
        preds, saving, fpr = refine_l2_det(DBf, DBr, kept_exp, Q, args.k, args.keep, keep_e=args.ekeep, mode="sym",
                                           red_bytes_override=red_bytes_override)
    elif args.mode == "l2-hoeff":
        preds, saving, fpr = refine_l2_hoeff(DBf, DBr, kept_exp, Q, args.k, args.delta, args.keep, keep_e=args.ekeep,
                                            red_bytes_override=red_bytes_override)
    else:
        raise ValueError("unknown mode")

    R = recall_at_k(gt, preds)
    print(f"mode={args.mode}  k={args.k}  m_keep={args.keep}  e_keep={args.ekeep}  delta={args.delta}")
    print(f"recall@{args.k}={R:.4f}  save={saving*100:.2f}%  fpr={fpr*100:.3f}%")
    if red_bytes_override is not None:
        print(f"compressed reduced-pass bytes/vec ≈ {red_bytes_override:.2f}")

    # If plots were requested, require matplotlib and run a sweep
    if args.plot_out is not None:
        if not _HAS_PLT:
            raise RuntimeError("plotting requested but matplotlib is not available")
        m_grid = [int(x) for x in args.sweep_m.split(",")] if args.sweep_m else [args.keep]
        d_grid = [float(x) for x in args.sweep_delta.split(",")] if args.sweep_delta else [args.delta]
        sweep_and_plot(DB, Q, args.mode, args.k, args.ekeep, m_grid, d_grid,
                       args.plot_out,
                       compress_level=(None if args.compress_level is None or args.compress_level < 0
                                       else args.compress_level))


if __name__ == "__main__":
    main()
