#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import io, os, struct, math, argparse
from typing import Tuple, List
import numpy as np

FP16_EXP_BITS = 5
FP16_EXP_BIAS = 15
FP16_MAN_BITS = 10

def rg():
    return np.random.default_rng(42)

def nz(a: np.ndarray, e: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n = np.maximum(n, e)
    return a / n

def ldv(p: str, m: int = 100_000) -> np.ndarray:
    with io.open(p, "r", encoding="utf-8", newline="\n", errors="ignore") as f:
        h = f.readline().strip().split()
        if len(h) < 2 or not h[0].isdigit() or not h[1].isdigit():
            raise ValueError("bad vec header")
        n, d = int(h[0]), int(h[1])
        t = min(n, m)
        s = None
        if n > m:
            s = sorted(rg().choice(n, size=t, replace=False).tolist())
        x = np.zeros((t, d), dtype=np.float32)
        w = 0
        S = set(s) if s is not None else None
        for i, line in enumerate(f):
            if w >= t: break
            if S is not None and i not in S: continue
            z = line.rstrip("\n").split()
            if len(z) < d + 1: continue
            v = np.fromstring(" ".join(z[1:]), sep=" ", dtype=np.float32)
            if v.size != d:
                v = np.array([float(q) for q in z[1:d+1]], dtype=np.float32)
            x[w] = v; w += 1
        if w != t: x = x[:w].copy()
    return x

def ldb(p: str, m: int = 100_000) -> np.ndarray:
    if p.lower().endswith(".vec"):
        return ldv(p, m)
    try:
        with open(p, "rb") as f:
            a = f.readline().strip(); b = f.readline().strip()
            L = int(a.decode("ascii")); D = int(b.decode("ascii"))
            v = np.fromfile(f, dtype=np.float32, count=L*D)
            if v.size != L*D: raise ValueError
            x = v.reshape(L, D)
            if L > m:
                idx = rg().choice(L, size=m, replace=False)
                x = x[idx]
            return x.astype(np.float32, copy=False)
    except Exception:
        pass
    with open(p, "rb") as f:
        L = struct.unpack("<i", f.read(4))[0]
        D = struct.unpack("<i", f.read(4))[0]
        v = np.fromfile(f, dtype=np.float32, count=L*D)
    x = v.reshape(L, D)
    if L > m:
        idx = rg().choice(L, size=m, replace=False)
        x = x[idx]
    return x.astype(np.float32, copy=False)

def ldf(p: str, m: int = 100_000) -> np.ndarray:
    with open(p, "rb") as f:
        d = f.read()
    if len(d) < 4: raise ValueError("bad fvecs")
    D = struct.unpack_from("<i", d, 0)[0]
    r = 4 + 4*D
    if (len(d) % r) != 0: raise ValueError("size mismatch")
    N = len(d)//r
    out = np.empty((min(N,m), D), dtype=np.float32)
    off = 0; w = 0
    for i in range(N):
        if w >= out.shape[0]: break
        off += 4
        v = np.frombuffer(d, dtype=np.float32, count=D, offset=off)
        out[w] = v; w += 1
        off += 4*D
    return out

def f2u(a: np.ndarray) -> np.ndarray:
    return a.view(np.uint16)

def u2f(a: np.ndarray) -> np.ndarray:
    return a.view(np.float16)

def sp(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = (a >> 15) & 0x1
    e = (a >> 10) & 0x1F
    m = a & 0x03FF
    return s, e, m

def cb(s: np.ndarray, e: np.ndarray, m: np.ndarray) -> np.ndarray:
    return ((s & 1) << 15) | ((e & 0x1F) << 10) | (m & 0x03FF)

def tm(mn: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    assert 0 <= k <= FP16_MAN_BITS
    if k == FP16_MAN_BITS:
        return mn.copy(), np.zeros_like(mn, dtype=np.uint16)
    d = FP16_MAN_BITS - k
    t = mn >> d
    mo = (t << d).astype(np.uint16)
    c = np.zeros_like(mo, dtype=np.uint16)
    return mo, c

def rp(x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    x16 = x.astype(np.float16)
    u = f2u(x16)
    s,e,m = sp(u)
    if k == FP16_MAN_BITS:
        mo, c = m, np.zeros_like(m, dtype=np.uint16)
    else:
        mo, c = tm(m, k)
    eo = (e + c).astype(np.uint16)
    uo = cb(s, eo, mo)
    xr = u2f(uo)
    return xr.astype(np.float32), eo.astype(np.uint8)

def rpe(x: np.ndarray, ek: int, mk: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert 0 <= ek <= FP16_EXP_BITS
    assert 0 <= mk <= FP16_MAN_BITS
    xf = x.astype(np.float16)
    uf = f2u(xf)
    sf, ef, mf = sp(uf)
    if mk == FP16_MAN_BITS:
        mt, cm = mf.copy(), np.zeros_like(mf, dtype=np.uint16)
    else:
        mt, cm = tm(mf, mk)
    et = (ef + cm).astype(np.uint16)
    if ek == FP16_EXP_BITS:
        er = et
    else:
        de = FP16_EXP_BITS - ek
        er = ((et >> de) << de).astype(np.uint16)
    ur = cb(sf, er, mt)
    xr = u2f(ur)
    ae = np.abs(xf.astype(np.float32) - xr.astype(np.float32)).astype(np.float32)
    return xr.astype(np.float32), er.astype(np.uint8), ae

def st(e: np.ndarray, mk: int) -> np.ndarray:
    ei = e.astype(np.int32)
    sub = (ei == 0)
    z = np.where(sub, 2.0 ** (1 - FP16_EXP_BIAS - mk),
                      2.0 ** (ei - FP16_EXP_BIAS - mk))
    return z.astype(np.float32)

def tkc(DB: np.ndarray, Q: np.ndarray, K: int) -> List[np.ndarray]:
    o = []
    for q in Q:
        s = DB @ q
        idx = np.argpartition(s, -K)[-K:]
        idx = idx[np.argsort(s[idx])[::-1]]
        o.append(idx)
    return o

def tkl(DB: np.ndarray, Q: np.ndarray, K: int) -> List[np.ndarray]:
    o = []
    for q in Q:
        d2 = np.sum((DB - q[None,:])**2, axis=1)
        idx = np.argpartition(d2, K)[:K]
        idx = idx[np.argsort(d2[idx])]
        o.append(idx)
    return o

def ins(hs: np.ndarray, hi: np.ndarray, s: float, i: int):
    K = hs.size
    if np.isneginf(hs).any():
        p = np.where(np.isneginf(hs))[0][0]
        hs[p] = s; hi[p] = i
    else:
        w = np.argmin(hs)
        if s > hs[w]:
            hs[w] = s; hi[w] = i

def ch(DBu: np.ndarray, DBru: np.ndarray, eb: np.ndarray, Q: np.ndarray, K: int, d: float, mk: int, *, ek: int):
    N, D = DBu.shape
    Qn = nz(Q)
    Dl = st(eb, mk)
    bf, br = 16, (1 + ek + mk)
    bF, bR = (bf*D)/8.0, (br*D)/8.0
    base = bF * N
    tot = 0.0
    out = []
    C = math.sqrt(2.0 * math.log(1.0 / float(d)))
    tc, fu = 0, 0
    for q in Qn:
        tc += N
        hs = np.full(K, -np.inf, dtype=np.float32)
        hi = np.full(K, -1, dtype=np.int32)
        tot += bR * N
        sh = DBru @ q
        se = np.argpartition(sh, -K)[-K:]
        F = set(int(i) for i in se)
        for i in se:
            sx = float(np.dot(DBu[i], q))
            ins(hs, hi, sx, int(i))
        sw = float(np.min(hs))
        tot += bF * len(se)
        m = np.ones(N, dtype=bool); m[se] = False
        ord = np.argsort(sh[m])[::-1]
        ci = np.where(m)[0][ord]
        for i in ci:
            t = C * float(np.linalg.norm(q * Dl[i]))
            if float(sh[i]) + t < sw: continue
            sx = float(np.dot(DBu[i], q))
            F.add(int(i))
            if (sx > sw) or np.isneginf(hs).any():
                ins(hs, hi, sx, int(i)); sw = float(np.min(hs))
            tot += bF
        fin = set(int(x) for x in hi.tolist())
        fu += len(F - fin)
        o = np.argsort(hs)[::-1]; out.append(hi[o])
    sv = 1.0 - (tot / (base * len(Qn)))
    fp = (fu / tc) if tc else 0.0
    return out, sv, fp

def cl1(DBu: np.ndarray, DBru: np.ndarray, eb: np.ndarray, Q: np.ndarray, K: int, mk: int, *, a: float = 1.0):
    N, D = DBu.shape
    Qn = nz(Q)
    Dl = st(eb, mk)
    bf, br = 16, (1 + FP16_EXP_BITS + mk)
    bF, bR = (bf*D)/8.0, (br*D)/8.0
    base = bF * N
    tot = 0.0
    out = []
    tc, fu = 0, 0
    for q in Qn:
        tc += N
        aq = np.abs(q).astype(np.float32)
        hs = np.full(K, -np.inf, dtype=np.float32)
        hi = np.full(K, -1, dtype=np.int32)
        tot += bR * N
        sh = DBru @ q
        se = np.argpartition(sh, -K)[-K:]
        F = set(int(i) for i in se)
        for i in se:
            sx = float(np.dot(DBu[i], q))
            ins(hs, hi, sx, int(i))
        sw = float(np.min(hs))
        tot += bF * len(se)
        m = np.ones(N, dtype=bool); m[se] = False
        ord = np.argsort(sh[m])[::-1]
        ci = np.where(m)[0][ord]
        for i in ci:
            b = float(np.dot(aq, Dl[i]))
            if float(sh[i]) + a*b < sw: continue
            sx = float(np.dot(DBu[i], q))
            F.add(int(i))
            if (sx > sw) or np.isneginf(hs).any():
                ins(hs, hi, sx, int(i)); sw = float(np.min(hs))
            tot += bF
        fin = set(int(x) for x in hi.tolist())
        fu += len(F - fin)
        o = np.argsort(hs)[::-1]; out.append(hi[o])
    sv = 1.0 - (tot / (base * len(Qn)))
    fp = (fu / tc) if tc else 0.0
    return out, sv, fp

def cl2(DB: np.ndarray, DBr: np.ndarray, eb: np.ndarray, Q: np.ndarray, K: int, mk: int, *, ek: int = FP16_EXP_BITS, a: float = 1.0):
    N, D = DB.shape
    DBu = nz(DB)
    DBru = nz(DBr)
    Qn = nz(Q)
    Dl = np.abs(DBu - DBru).astype(np.float32)
    B = np.sqrt(np.sum(Dl*Dl, axis=1)).astype(np.float32)
    bf, br = 16, (1 + ek + mk)
    bF, bR = (bf*D)/8.0, (br*D)/8.0
    base = bF * N
    tot = 0.0
    out = []
    tc, fu = 0, 0
    for q in Qn:
        tc += N
        hs = np.full(K, -np.inf, dtype=np.float32)
        hi = np.full(K, -1, dtype=np.int32)
        tot += bR * N
        sh = DBru @ q
        se = np.argpartition(sh, -K)[-K:]
        F = set(int(i) for i in se)
        for i in se:
            sx = float(np.dot(DBu[i], q))
            ins(hs, hi, sx, int(i))
        sw = float(np.min(hs))
        tot += bF * len(se)
        m = np.ones(N, dtype=bool); m[se] = False
        ord = np.argsort(sh[m])[::-1]
        ci = np.where(m)[0][ord]
        for i in ci:
            if float(sh[i]) + a*float(B[i]) < sw: continue
            sx = float(np.dot(DBu[i], q))
            F.add(int(i))
            if (sx > sw) or np.isneginf(hs).any():
                ins(hs, hi, sx, int(i)); sw = float(np.min(hs))
            tot += bF
        fin = set(int(x) for x in hi.tolist())
        fu += len(F - fin)
        o = np.argsort(hs)[::-1]; out.append(hi[o])
    sv = 1.0 - (tot / (base * len(Qn)))
    fp = (fu / tc) if tc else 0.0
    return out, sv, fp

def lb1(a: np.ndarray, Dv: np.ndarray) -> float:
    t = np.maximum(np.abs(a) - Dv, 0.0).astype(np.float32)
    return float(np.sum(t*t, dtype=np.float32))

def lb2(a: np.ndarray, Dv: np.ndarray, xt: np.ndarray) -> float:
    s = np.where(np.signbit(xt), -1.0, 1.0).astype(np.float32)
    b = a * s
    m1 = (b <= 0.0)
    m2 = (b > 0.0) & (b < Dv)
    m3 = ~(m1 | m2)
    t = np.empty_like(a, dtype=np.float32)
    t[m1] = a[m1]*a[m1]
    t[m2] = 0.0
    t[m3] = (np.abs(a[m3]) - Dv[m3])**2
    return float(np.sum(t, dtype=np.float32))

def e1(DBf: np.ndarray, DBr: np.ndarray, eb: np.ndarray, Q: np.ndarray, K: int, mk: int, *, ek: int, mode: str = "tz"):
    N, D = DBf.shape
    Dl = st(eb, mk).astype(np.float32)
    bf, br = 16, (1 + ek + mk)
    bF, bR = (bf*D)/8.0, (br*D)/8.0
    base = bF * N
    tot = 0.0
    out = []
    tc, fu = 0, 0
    for q in Q:
        tc += N
        tot += bR * N
        A = (q[None,:] - DBr).astype(np.float32)
        dh = np.einsum("ij,ij->i", A, A)
        se = np.argpartition(dh, K)[:K]
        F = set(int(i) for i in se)
        hd = np.full(K, np.inf, dtype=np.float32)
        hi = np.full(K, -1, dtype=np.int32)
        for i in se:
            dif = (q - DBf[i]); d2 = float(np.dot(dif, dif))
            w = np.argmax(hd)
            if d2 < hd[w] or np.isinf(hd).any():
                hd[w] = d2; hi[w] = int(i)
        dw = float(np.max(hd))
        tot += bF * len(se)
        m = np.ones(N, dtype=bool); m[se] = False
        ord = np.argsort(dh[m]); ci = np.where(m)[0][ord]
        for i in ci:
            a = A[i]
            L = lb2(a, Dl[i], DBr[i]) if mode == "tz" else lb1(a, Dl[i])
            if L > dw: continue
            dif = (q - DBf[i]); d2 = float(np.dot(dif, dif))
            F.add(int(i))
            w = np.argmax(hd)
            if d2 < hd[w] or np.isinf(hd).any():
                hd[w] = d2; hi[w] = int(i); dw = float(np.max(hd))
            tot += bF
        fin = set(int(x) for x in hi.tolist())
        fu += len(F - fin)
        o = np.argsort(hd); out.append(hi[o])
    sv = 1.0 - (tot / (base * len(Q)))
    fp = (fu / tc) if tc else 0.0
    return out, sv, fp

def e2(DBf: np.ndarray, DBr: np.ndarray, eb: np.ndarray, Q: np.ndarray, K: int, d: float, mk: int, *, ek: int):
    N, D = DBf.shape
    Dl = st(eb, mk).astype(np.float32)
    bf, br = 16, (1 + ek + mk)
    bF, bR = (bf*D)/8.0, (br*D)/8.0
    base = bF * N
    tot = 0.0
    out = []
    tc, fu = 0, 0
    C = math.sqrt(2.0 * math.log(1.0 / float(d)))
    for q in Q:
        tc += N
        tot += bR * N
        A = (q[None,:] - DBr).astype(np.float32)
        dh = np.einsum("ij,ij->i", A, A)
        se = np.argpartition(dh, K)[:K]
        F = set(int(i) for i in se)
        hd = np.full(K, np.inf, dtype=np.float32)
        hi = np.full(K, -1, dtype=np.int32)
        for i in se:
            dif = (q - DBf[i]); d2 = float(np.dot(dif, dif))
            w = np.argmax(hd)
            if d2 < hd[w] or np.isinf(hd).any():
                hd[w] = d2; hi[w] = int(i)
        dw = float(np.max(hd))
        tot += bF * len(se)
        m = np.ones(N, dtype=bool); m[se] = False
        ord = np.argsort(dh[m]); ci = np.where(m)[0][ord]
        for i in ci:
            a = A[i]
            L = float(np.dot(a, a)) - 2.0 * C * float(np.linalg.norm(a * Dl[i]))
            if L > dw: continue
            dif = (q - DBf[i]); d2 = float(np.dot(dif, dif))
            F.add(int(i))
            w = np.argmax(hd)
            if d2 < hd[w] or np.isinf(hd).any():
                hd[w] = d2; hi[w] = int(i); dw = float(np.max(hd))
            tot += bF
        fin = set(int(x) for x in hi.tolist())
        fu += len(F - fin)
        o = np.argsort(hd); out.append(hi[o])
    sv = 1.0 - (tot / (base * len(Q)))
    fp = (fu / tc) if tc else 0.0
    return out, sv, fp

def rk(g: List[np.ndarray], p: List[np.ndarray]) -> float:
    K = g[0].size
    s = 0.0
    for a, b in zip(g, p):
        s += len(set(a.tolist()) & set(b.tolist())) / K
    return s / len(g)

def mp(t: int) -> Tuple[int, int]:
    if t >= 0:
        return FP16_EXP_BITS, int(t)
    else:
        return int(min(FP16_EXP_BITS, max(0, -t))), 0

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--mode", choices=["cos-hoeff","cos-l1","cos-l2","l2-tz","l2-sym","l2-hoeff"], default="cos-hoeff")
    pa.add_argument("--db", required=False)
    pa.add_argument("--q", required=False)
    pa.add_argument("--k", type=int, default=20)
    pa.add_argument("--keep", type=int, default=6)
    pa.add_argument("--ekeep", type=int, default=FP16_EXP_BITS)
    pa.add_argument("--delta", type=float, default=0.7)
    pa.add_argument("--maxrows", type=int, default=100000)
    a = pa.parse_args()

    if a.db is None or a.q is None:
        N, D, QQ = 2000, 128, 30
        DB = rg().normal(size=(N,D)).astype(np.float32)
        Q = rg().normal(size=(QQ,D)).astype(np.float32)
    else:
        DB = ldf(a.db, a.maxrows) if a.db.endswith(".fvecs") else ldb(a.db, a.maxrows)
        Q = ldb(a.q, 10_000)

    DBf = DB.astype(np.float16).astype(np.float32)
    DBu = nz(DBf)
    DBr, eb = rp(DB, a.keep)
    DBru = nz(DBr)

    if a.mode.startswith("cos-"):
        gt = tkc(DBu, nz(Q), a.k)
    else:
        gt = tkl(DBf, Q, a.k)

    if a.mode == "cos-hoeff":
        pr, sv, fp = ch(DBu, DBru, eb, Q, a.k, a.delta, a.keep, ek=a.ekeep)
    elif a.mode == "cos-l1":
        pr, sv, fp = cl1(DBu, DBru, eb, Q, a.k, a.keep, a=1.0)
    elif a.mode == "cos-l2":
        pr, sv, fp = cl2(DB, DBr, eb, Q, a.k, a.keep, ek=a.ekeep, a=1.0)
    elif a.mode == "l2-tz":
        pr, sv, fp = e1(DBf, DBr, eb, Q, a.k, a.keep, ek=a.ekeep, mode="tz")
    elif a.mode == "l2-sym":
        pr, sv, fp = e1(DBf, DBr, eb, Q, a.k, a.keep, ek=a.ekeep, mode="sym")
    elif a.mode == "l2-hoeff":
        pr, sv, fp = e2(DBf, DBr, eb, Q, a.k, a.delta, a.keep, ek=a.ekeep)
    else:
        raise ValueError

    R = rk(gt, pr)
    print(f"mode={a.mode} k={a.k} m_keep={a.keep} e_keep={a.ekeep} delta={a.delta}")
    print(f"recall@{a.k}={R:.4f}  save={sv*100:.2f}%  fpr={fp*100:.3f}%")

if __name__ == "__main__":
    main()
