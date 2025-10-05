# Bandwidth-First ANN Refinement: Precision-on-Demand in Vector Databases

This repository provides the implementation and evaluation code for the paper:  
**“Bandwidth-First ANN Refinement: Precision-on-Demand in Vector Databases”**

---

## Overview

Approximate Nearest Neighbor (ANN) search is fundamental to modern retrieval systems but is increasingly **bottlenecked by memory bandwidth** rather than computation.  
This project introduces a **bandwidth-first, representation-preserving refinement framework** that minimizes data movement during ANN search while maintaining accuracy — without modifying vector representations or index structures.

The paper proposes and evaluates:

- **Precision-on-demand refinement** with early rejection to avoid unnecessary full-precision fetches  
- **Mathematically grounded cushions** (ℓ₁, ℓ₂, sign-aware, Hoeffding) for cosine similarity and Euclidean distance  
- **Disaggregated bit-plane memory layout** enabling sub-word precision access under DRAM bursts  
- **Bit-wise shuffling and hardware-assisted lossless compression** to further reduce effective bandwidth usage  

Together, these techniques reduce ANN refinement bandwidth by **up to 60%**, with an additional **1.8× reduction** from lossless compression — all while maintaining **≥99% recall** across diverse real-world datasets.

---

## Code

The codebase includes:

- Logic for **reduced-precision distance computation** and adaptive early-rejection thresholds  
- Implementations of **ℓ₁, ℓ₂, sign-aware, and Hoeffding cushions** for both cosine and Euclidean metrics  
- Simulation of **disaggregated in-memory placement** and bandwidth accounting  
- Tools for **bit-wise shuffling, compression evaluation**, and plotting recall vs. bandwidth trade-offs  
- Scripts for reproducing all data from the paper (Recall@20, False Positives, Bandwidth Saving)

---

## Datasets

The datasets used in our experiments include:

- **Cosine similarity:** GloVe, WikiNews, FineWeb, MS MARCO, 20 Newsgroups, DBPEDIA  
- **Euclidean distance:** SIFT, GIST, GloVe  

All datasets are publicly available through FAISS, HuggingFace, or other open IR archives.  
Due to size constraints, they are **not included** in this repository; however, they are all publicly available and can be easily found online.
