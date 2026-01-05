#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sydney Urban Objects (SUO) → NPZ
- objects/<class> 또는 flat(objects/*.{csv,npy,npz,ply,pcd,bin})에서 'stem'(예: pedestrian.80.4093) 단위 집계
  동일 stem 다중 확장자: CSV > NPY > NPZ > PLY > PCD > BIN 우선순위로 1개만 채택
  *.bin.meta 완전 무시
- 클래스별 '객체(파일) 단위' 7:1:2 분할(교집합 0, 완전 분리)
- 각 객체: (객체 전체 기준) 정규화(centroid→0, max radius=1) → FPS로 num_points 샘플, no-reuse 1차 충전
  → 부족분은 **무조건** 랜덤(객체 내 used 무시)으로 채움. 단, 샘플 내부는 가능한 한 중복 없는 인덱스로 구성
- CSV 자동 구분자 + 헤더명(x,y,z) 인식 + 정규식 파서(괄호/혼합 구분자 허용), BIN은 여러 레이아웃 시도
- CSV 실패 시 동일 stem의 하위 우선순위 확장자로 자동 폴백
- 스플릿 합친 뒤: (1) 샘플 내부 포인트 순열, (2) split별 (X,Y) 페어 셔플
- rotation==1이면 하나의 공통 SO(3) 회전을 train/val/test 전부에 적용 (ShapeNet 코드와 동일)
- 결과는 항상 /Users/semin_love_u/Desktop/1st_paper 에 저장
"""

import os, sys, glob, json, argparse, importlib.util, re
import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.stats import special_ortho_group

# ---------------- Optional user BIN reader (read-bin.py) ----------------
def _maybe_import_read_bin(root_dir: str):
    path = os.path.join(root_dir, "read-bin.py")
    if not os.path.isfile(path):
        return None
    spec = importlib.util.spec_from_file_location("suo_read_bin", path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore
        fn = getattr(mod, "read_bin", None)
        return fn if callable(fn) else None
    except Exception:
        return None

# ---------------- Discovery: CSV first, ignore .bin.meta ----------------
_PRIORITIES = [
    (".csv", ".CSV"),
    (".npy", ".NPY"),
    (".npz", ".NPZ"),
    (".ply", ".PLY"),
    (".pcd", ".PCD"),
    (".bin", ".BIN"),
]

def _stem_for_pick(base: str) -> Optional[str]:
    if base.endswith(".bin.meta") or base.endswith(".BIN.META"):
        return None
    if base.lower().endswith(".bin"):
        return base[:-4]
    name, _ = os.path.splitext(base)
    return name

def _gather_by_priority(dirpath: str) -> Dict[str, str]:
    chosen: Dict[str, str] = {}
    for group in _PRIORITIES:
        for ext in group:
            for p in glob.glob(os.path.join(dirpath, f"*{ext}")):
                base = os.path.basename(p)
                sk = _stem_for_pick(base)
                if sk is None:
                    continue
                if sk not in chosen:
                    chosen[sk] = p  # CSV 우선
    return chosen

def discover_objects(root_dir: str, classes: List[str]) -> Dict[str, List[str]]:
    obj_root = os.path.join(root_dir, "objects")
    if not os.path.isdir(obj_root):
        raise FileNotFoundError(f"Missing folder: {obj_root}")

    classes_lc = [c.lower() for c in classes]
    out: Dict[str, List[str]] = {cls: [] for cls in classes}

    # 폴더 기반 분류가 있으면 그대로, 없으면 flat 파일명 prefix로 자동 분류
    has_any_class_dir = any(os.path.isdir(os.path.join(obj_root, cls)) for cls in classes)
    if has_any_class_dir:
        for cls in classes:
            cls_dir = os.path.join(obj_root, cls)
            files: List[str] = []
            if os.path.isdir(cls_dir):
                chosen = _gather_by_priority(cls_dir)
                files = sorted(chosen.values())
            out[cls] = files
            examples = ", ".join(os.path.basename(x) for x in files[:3])
            print(f"[DISCOVER] {cls}: {len(files)} files (CSV prioritized). e.g., {examples}")
        return out

    print(f"[INFO] No class subfolders in '{obj_root}'. Falling back to flat discovery by filename prefix.")
    chosen_all = _gather_by_priority(obj_root)
    for sk, path in chosen_all.items():
        prefix = sk.split('.', 1)[0].lower()
        if prefix in classes_lc:
            cls = classes[classes_lc.index(prefix)]
            out[cls].append(path)
    for cls in classes:
        files = sorted(out[cls])
        examples = ", ".join(os.path.basename(x) for x in files[:3])
        print(f"[DISCOVER] {cls}: {len(files)} files (CSV prioritized). e.g., {examples}")
    return out

# ---------------- Robust loaders ----------------
_FLOAT_RE = re.compile(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?')

def _detect_header_and_cols(first_line: str):
    """
    첫 줄에서 헤더명 'x','y','z'가 보이면 그 인덱스를 반환.
    보이지 않으면 SUO 기본 포맷(t, intensity, id, x, y, z, ...)에 맞춰 (3,4,5) 반환.
    """
    # 후보 구분자
    for d in [",", "\t", ";", " ", "|"]:
        parts = [p.strip().lower() for p in first_line.strip().split(d)]
        if len(parts) >= 3 and {"x", "y", "z"}.issubset(set(parts)):
            return True, (parts.index("x"), parts.index("y"), parts.index("z"))
    # 헤더가 없거나 헤더에 xyz가 명시되지 않은 경우: SUO 기본 (3,4,5)
    # 단, 첫 줄이 과학적 표기(1.2e-3 등) 숫자여도 헤더로 오인하지 않도록 함
    has_alpha = any(ch.isalpha() for ch in first_line if ch not in "eE+-.,;|\t ")
    return (True if has_alpha else False), (3, 4, 5)

def _load_csv_xyz(path: str) -> np.ndarray:
    # 1) 첫 줄 검사로 헤더/타겟 컬럼 결정
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            first = f.readline()
    except Exception:
        first = ""
    has_header_hint, target_cols = _detect_header_and_cols(first)

    cand_delims = [",", "\t", ";", " ", "|"]
    # 2) 여러 구분자 × (헤더유무 0/1) 조합 시도
    for delim in cand_delims:
        for skip in ((1 if has_header_hint else 0), 0):
            try:
                arr = np.genfromtxt(
                    path,
                    delimiter=delim,
                    dtype=np.float32,
                    usecols=target_cols,
                    skip_header=skip,
                    autostrip=True,
                    invalid_raise=False,
                )
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 3)
                if arr.size > 0 and arr.shape[1] == 3 and np.isfinite(arr).all():
                    return arr.astype(np.float32)
            except Exception:
                pass

    # 3) 정규식 강제 파서(괄호/혼합 구분자 허용) — 행마다 최소 6개 수가 있으면 (3,4,5)로 취함
    xs, ys, zs = [], [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            nums = _FLOAT_RE.findall(line)
            if len(nums) >= 6:
                try:
                    x = float(nums[3]); y = float(nums[4]); z = float(nums[5])
                    xs.append(x); ys.append(y); zs.append(z)
                except Exception:
                    continue
            elif len(nums) >= 3 and not has_header_hint:
                # 정말로 3개씩만 있는 순수 XYZ 파일일 수도 있음
                try:
                    x = float(nums[0]); y = float(nums[1]); z = float(nums[2])
                    xs.append(x); ys.append(y); zs.append(z)
                except Exception:
                    continue
    if not xs:
        raise ValueError("CSV parse failed: no XYZ columns recognized")
    return np.stack([xs, ys, zs], axis=1).astype(np.float32)

def _load_points_generic(path: str, read_bin_fn=None) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".txt", ".tsv"):
        return _load_csv_xyz(path)
    if ext == ".npy":
        arr = np.asarray(np.load(path))
        if arr.ndim == 2 and arr.shape[1] >= 3: return arr[:, :3].astype(np.float32)
        raise ValueError(f".npy not (N,>=3): {arr.shape}")
    if ext == ".npz":
        with np.load(path) as z:
            for key in ("points", "xyz", "XYZ", "data"):
                if key in z: arr = np.asarray(z[key]); break
            else:
                ks = [k for k in z.files if hasattr(z[k], "shape")]
                if not ks: raise ValueError(".npz has no arrays")
                arr = np.asarray(z[ks[0]])
        if arr.ndim == 2 and arr.shape[1] >= 3: return arr[:, :3].astype(np.float32)
        raise ValueError(f".npz not (N,>=3): {arr.shape}")
    if ext in (".ply", ".pcd"):
        try:
            import open3d as o3d
        except Exception as e:
            raise ValueError("open3d not installed for ply/pcd") from e
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 3: raise ValueError("pcd/ply has <3 dims")
        return pts[:, :3].astype(np.float32)
    if ext == ".bin":
        if read_bin_fn is not None:
            pts = np.asarray(read_bin_fn(path))
            if pts.ndim != 2 or pts.shape[1] < 3: raise ValueError(f"read_bin bad shape {pts.shape}")
            return pts[:, :3].astype(np.float32)
        # (A) headerless float32 NxD
        b32 = np.fromfile(path, dtype=np.float32)
        for d in (3,4,5,6,7,8,9,10,12,16):
            if b32.size % d == 0 and b32.size >= d:
                arr = b32.reshape(-1, d)[:, :3]
                if arr.shape[0] > 0: return arr.astype(np.float32)
        # (B) int32 N header + float32 payload
        try:
            with open(path, "rb") as f:
                n = np.fromfile(f, dtype=np.int32, count=1)
                if n.size == 1 and n[0] > 0:
                    rest = np.fromfile(f, dtype=np.float32)
                    if rest.size >= 3*n[0] and rest.size % n[0] == 0:
                        d = rest.size // n[0]
                        arr = rest.reshape(n[0], d)[:, :3]
                        return arr.astype(np.float32)
        except Exception:
            pass
        # (C) float64 fallback
        b64 = np.fromfile(path, dtype=np.float64)
        for d in (3,4,5,6,8,10,12,16):
            if b64.size % d == 0 and b64.size >= d:
                arr = b64.reshape(-1, d)[:, :3]
                if arr.shape[0] > 0: return arr.astype(np.float32)
        raise ValueError("Unrecognized .bin layout (prefer CSV or provide read-bin.py)")
    raise ValueError(f"Unsupported extension: {ext}")

# ---- Same-stem fallback helpers ---------------------------------------
def _stem_from_path(path: str) -> str:
    base = os.path.basename(path); sk = _stem_for_pick(base)
    if sk is None: raise ValueError("bad stem")
    return sk

def _candidate_paths_for_same_stem(path: str) -> List[str]:
    d = os.path.dirname(path); stem = _stem_from_path(path); cands: List[str] = []
    for group in _PRIORITIES:
        for ext in group:
            p = os.path.join(d, stem + ext)
            if os.path.isfile(p): cands.append(p)
    cands = [p for p in cands if not p.lower().endswith(".bin.meta")]
    seen=set(); uniq=[]
    for p in cands:
        if p not in seen: uniq.append(p); seen.add(p)
    return uniq

# ---------------- Geometry & FPS ----------------
def normalize_unit_sphere(points: np.ndarray):
    c = points.mean(axis=0, keepdims=True)
    centered = points - c
    scale = float(np.linalg.norm(centered, axis=1).max() + 1e-12)
    return (centered / scale).astype(np.float32), c.squeeze(0).astype(np.float32), scale

def farthest_point_sampling(points: np.ndarray, k: int, candidate_idx=None, rng=None) -> np.ndarray:
    if rng is None: rng = np.random.default_rng()
    P = points.shape[0]
    if candidate_idx is None: candidate_idx = np.arange(P, dtype=np.int32)
    if candidate_idx.size < k: raise ValueError("insufficient_candidates")
    cand = points[candidate_idx]; C = cand.shape[0]
    sel_local = np.empty(k, dtype=np.int32); cur = int(rng.integers(0, C)); sel_local[0] = cur
    d2 = np.full(C, np.inf, dtype=np.float64)
    for i in range(1, k):
        diff = cand - cand[cur]
        d2 = np.minimum(d2, np.einsum('ij,ij->i', diff, diff))
        cur = int(np.argmax(d2)); sel_local[i] = cur
    return candidate_idx[sel_local]

def always_permute_points(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    N, k, _ = X.shape; out = np.empty_like(X)
    for i in range(N): out[i] = X[i, rng.permutation(k)]
    return out

# ---------------- Split & Build ----------------
def _tqdm(it, desc):
    try:
        from tqdm import tqdm
        return tqdm(it, desc=desc, leave=False)
    except Exception:
        return it

def split_objects_disjoint(objs: List[str], rng, ratio=(7,1,2)) -> Tuple[List[str], List[str], List[str]]:
    n = len(objs)
    if n == 0: return [], [], []
    a,b,c = ratio; s = a+b+c
    frac = np.array([a,b,c], dtype=np.float64) / float(s)
    counts = np.floor(frac * n).astype(int); need = n - counts.sum(); order = np.argsort(-frac)
    for i in range(need): counts[order[i % 3]] += 1
    rng.shuffle(objs); x, y = counts[0], counts[1]
    return objs[:x], objs[x:x+y], objs[x+y:]

def build_pools_for_split(file_list: List[str], oversample_P: int, loader_read_bin, rng, verbose=False):
    cache, used_map, errors = {}, {}, {}; ok = fail = 0
    iterator = file_list if not verbose else _tqdm(file_list, f"pre-sample [{len(file_list)} objs]")
    for chosen_path in iterator:
        tried = []; success = False; last_err = "unknown"
        candidates = _candidate_paths_for_same_stem(chosen_path)
        if chosen_path in candidates:
            candidates.remove(chosen_path)
        candidates = [chosen_path] + candidates
        for path in candidates:
            tried.append(os.path.basename(path))
            try:
                pts_all = _load_points_generic(path, read_bin_fn=loader_read_bin)
                # 객체 전체 기준으로 정규화 먼저, 그 다음 oversample_P로 줄이기
                pts_norm, _, _ = normalize_unit_sphere(pts_all)
                if pts_norm.shape[0] <= oversample_P:
                    pool = pts_norm
                else:
                    pool = pts_norm[rng.choice(pts_norm.shape[0], size=oversample_P, replace=False)]
                cache[path] = pool
                used_map[path] = np.zeros(pool.shape[0], dtype=bool)
                ok += 1; success = True
                if verbose and os.path.basename(path) != os.path.basename(chosen_path):
                    print(f"[FALLBACK] used {os.path.basename(path)} instead of {os.path.basename(chosen_path)}")
                break
            except Exception as e:
                last_err = str(e); continue
        if not success:
            fail += 1; errors[chosen_path] = f"all candidates failed; tried={tried} | last_err={last_err}"
    print(f"[POOL] built: ok={ok}, fail={fail}")
    for p, msg in list(errors.items())[:10]:
        print(f"   - {os.path.basename(p)} → {msg}")
    return cache, used_map

def capacity(cache: Dict[str, np.ndarray], num_points: int) -> int:
    return sum(int(cache[p].shape[0] // num_points) for p in cache)

# --- NEW: 1차 FPS 무재사용 + 2차 랜덤 재사용(샘플 내부 중복 최소화) ---
def fill_quota_with_fps_then_random(num_points: int, target: int,
                                    cache: Dict[str, np.ndarray], used_map: Dict[str, np.ndarray],
                                    label_value: int, rng,
                                    bucket_X: List[np.ndarray], bucket_Y: List[int],
                                    verbose=False, tag="") -> Tuple[int, int]:
    if target <= 0 or len(cache) == 0:
        if verbose: print(f"[INFO] {tag}: nothing to fill")
        return 0, 0

    keys = list(cache.keys())
    ptr = 0
    cnt = 0

    # ---- Phase 1: FPS + no-reuse ----
    stagnation = 0
    while cnt < target:
        prev = cnt
        path = keys[ptr]
        ptr = (ptr + 1) % len(keys)
        pool = cache[path]
        used = used_map[path]
        cand = np.where(~used)[0]
        if cand.size >= num_points:
            sel = farthest_point_sampling(pool, num_points, candidate_idx=cand, rng=rng)
            used[sel] = True
            bucket_X.append(pool[sel][None, ...])
            bucket_Y.append(label_value)
            cnt += 1
        stagnation = stagnation + 1 if cnt == prev else 0
        if stagnation > len(keys) * 3:
            break

    first_pass = cnt

    # ---- Phase 2: Random + reuse (within-sample unique if possible) ----
    fallback = 0
    while cnt < target and len(keys) > 0:
        path = keys[int(rng.integers(0, len(keys)))]
        pool = cache[path]
        C = pool.shape[0]
        if C == 0:
            continue
        if C >= num_points:
            sel = rng.choice(C, size=num_points, replace=False)  # 샘플 내부 중복 없음
        else:
            # 가능한 한 유니크하게 채우고, 남는 건 순환으로 채움
            order = rng.permutation(C)
            sel = np.resize(order, num_points)
        bucket_X.append(pool[sel][None, ...])
        bucket_Y.append(label_value)
        cnt += 1
        fallback += 1

    if verbose:
        print(f"[INFO] {tag}: FPS(no-reuse) filled {first_pass}/{target}; fallback(random) +{fallback}")
    return cnt, fallback

# ---------------- Build pipeline ----------------
def build_suo(
    root_dir: str,
    classes: List[str],
    num_points: int,
    train_per_class: int,
    val_per_class: int,
    test_per_class: int,
    oversample_P: int = 4096,
    rotation: int = 0,
    seed: int = 1557,
    verbose: bool = True,
):
    np.random.seed(seed); rng = np.random.default_rng(seed)

    class_to_files = discover_objects(root_dir, classes)

    # class별 객체 리스트를 먼저 7:1:2로 분할(완전 분리)
    splits = {}
    for cls in classes:
        files = sorted(list(class_to_files.get(cls, [])))
        tr, va, te = split_objects_disjoint(files, rng, ratio=(7,1,2))
        splits[cls] = (tr, va, te)
        if verbose: print(f"[SPLIT] {cls}: total={len(files)} → train={len(tr)}, val={len(va)}, test={len(te)}")

    read_bin_fn = _maybe_import_read_bin(root_dir)

    # split별 cache
    tr_cache = {}; tr_used = {}
    va_cache = {}; va_used = {}
    te_cache = {}; te_used = {}

    for cls in classes:
        tr, va, te = splits[cls]
        c1,u1 = build_pools_for_split(tr, oversample_P, read_bin_fn, rng, verbose)
        c2,u2 = build_pools_for_split(va, oversample_P, read_bin_fn, rng, verbose)
        c3,u3 = build_pools_for_split(te, oversample_P, read_bin_fn, rng, verbose)
        tr_cache.update(c1); tr_used.update(u1)
        va_cache.update(c2); va_used.update(u2)
        te_cache.update(c3); te_used.update(u3)

    # 로드 통계(stem으로 정확 매칭)
    def _stem_name(p):
        return os.path.splitext(os.path.basename(p))[0]
    for cls in classes:
        tr, va, te = splits[cls]
        tr_stems = {_stem_name(p) for p in tr}
        va_stems = {_stem_name(p) for p in va}
        te_stems = {_stem_name(p) for p in te}
        n_tr_ok = sum(1 for k in tr_cache if _stem_name(k) in tr_stems)
        n_va_ok = sum(1 for k in va_cache if _stem_name(k) in va_stems)
        n_te_ok = sum(1 for k in te_cache if _stem_name(k) in te_stems)
        print(f"[LOAD] {cls}: loaded train/val/test objs = {n_tr_ok}/{n_va_ok}/{n_te_ok}")

    # 클래스별 용량 체크(각 클래스의 split subset만 사용) — 부족해도 에러 없이 경고 후 진행
    def cap_subset(split_files, cache_dict):
        stems = {os.path.splitext(os.path.basename(p))[0] for p in split_files}
        sub = {k:v for k,v in cache_dict.items() if os.path.splitext(os.path.basename(k))[0] in stems}
        return capacity(sub, num_points)

    for cls in classes:
        tr, va, te = splits[cls]
        cap_tr = cap_subset(tr, tr_cache); cap_va = cap_subset(va, va_cache); cap_te = cap_subset(te, te_cache)
        print(f"[CAP ] {cls}: train/val/test = {cap_tr}/{cap_va}/{cap_te}  (quota {train_per_class}/{val_per_class}/{test_per_class})")
        if cap_tr < train_per_class or cap_va < val_per_class or cap_te < test_per_class:
            print(f"[WARN] {cls}: capacity shortage; will fill remaining with random (reuse allowed per-sample policy).")

    # 샘플 채우기
    label_map = {cls: i for i, cls in enumerate(classes)}
    trX, trY, vaX, vaY, teX, teY = [], [], [], [], [], []

    # fallback 통계
    fb_stats = {"train": 0, "val": 0, "test": 0}

    for cls in classes:
        lt = label_map[cls]
        tr_files, va_files, te_files = splits[cls]

        def subset(cache_dict, split_files):
            stems = {os.path.splitext(os.path.basename(p))[0] for p in split_files}
            return {k:v for k,v in cache_dict.items() if os.path.splitext(os.path.basename(k))[0] in stems}

        tr_sub = subset(tr_cache, tr_files)
        va_sub = subset(va_cache, va_files)
        te_sub = subset(te_cache, te_files)

        tr_used_sub = {k:tr_used[k] for k in tr_sub}
        va_used_sub = {k:va_used[k] for k in va_sub}
        te_used_sub = {k:te_used[k] for k in te_sub}

        filled, fb = fill_quota_with_fps_then_random(num_points, train_per_class, tr_sub, tr_used_sub, lt, rng, trX, trY, verbose, f"{cls} train")
        fb_stats["train"] += fb
        filled, fb = fill_quota_with_fps_then_random(num_points, val_per_class,   va_sub, va_used_sub, lt, rng, vaX, vaY, verbose, f"{cls} val")
        fb_stats["val"]   += fb
        filled, fb = fill_quota_with_fps_then_random(num_points, test_per_class,  te_sub, te_used_sub, lt, rng, teX, teY, verbose, f"{cls} test")
        fb_stats["test"]  += fb

    # stack & dtype
    trX = np.concatenate(trX, axis=0).astype(np.float32)
    vaX = np.concatenate(vaX, axis=0).astype(np.float32)
    teX = np.concatenate(teX, axis=0).astype(np.float32)
    trY = np.asarray(trY, dtype=np.int64)
    vaY = np.asarray(vaY, dtype=np.int64)
    teY = np.asarray(teY, dtype=np.int64)

    # 1) 샘플 내부 포인트 순열 (split별 독립)
    trX = always_permute_points(trX, rng)
    vaX = always_permute_points(vaX, rng)
    teX = always_permute_points(teX, rng)

    # 2) split별 (X,Y) 페어 셔플
    idx = rng.permutation(len(trX)); trX, trY = trX[idx], trY[idx]
    idx = rng.permutation(len(vaX)); vaX, vaY = vaX[idx], vaY[idx]
    idx = rng.permutation(len(teX)); teX, teY = teX[idx], teY[idx]

    # 3) 공통 회전(옵션)
    if rotation == 1:
        R = special_ortho_group.rvs(3, random_state=int(rng.integers(0, 1<<31)))
        trX = trX @ R.T; vaX = vaX @ R.T; teX = teX @ R.T
        print("[AUG ] Applied one common SO(3) rotation to all splits.")

    print(f"[DONE] totals → train={len(trX)} val={len(vaX)} test={len(teX)}")
    print(f"[FALLBACK] random-with-reuse counts → train:{fb_stats['train']} val:{fb_stats['val']} test:{fb_stats['test']}")
    return trX, trY, vaX, vaY, teX, teY, fb_stats

# ---------------- Save helper (fixed dir) ----------------
def save_npz_fixed(out_arrays, classes, num_points, train_per_class, val_per_class, test_per_class, seed, rotation, oversample_P):
    FIXED_OUT_DIR = "/Users/semin_love_u/Desktop/1st_paper"
    os.makedirs(FIXED_OUT_DIR, exist_ok=True)
    K = len(classes)
    # 파일명은 기존 패턴 유지하되, 버전/메타에 정책을 명시
    fname = f"SUO_{K}classes_{num_points}_1_fps_train{train_per_class}_val{val_per_class}_test{test_per_class}_new.npz"
    out_npz = os.path.join(FIXED_OUT_DIR, fname)

    trX, trY, vaX, vaY, teX, teY, fb_stats = out_arrays
    meta = {
        "version": "suo_quota_fps_then_random_v1",
        "classes": classes,
        "num_classes": int(K),
        "num_points": int(num_points),
        "train_per_class": int(train_per_class),
        "val_per_class":   int(val_per_class),
        "test_per_class":  int(test_per_class),
        "oversample_P": int(oversample_P),
        "rotation": int(rotation),
        "seed": int(seed),
        "shuffled": True,
        "disjoint_objects": True,
        "split_ratio": [7,1,2],
        "xyz_source": "csv(x,y,z)=cols(3,4,5) or header-detected; npy/npz/ply/pcd/bin first 3 columns",
        "normalize_policy": "per-object, full points → unit sphere",
        "sampling_policy": "Phase-1: FPS with no reuse per object; Phase-2: random with reuse (within-sample unique if possible) to meet quotas",
        "fallback_counts": fb_stats,
    }

    np.savez_compressed(
        out_npz,
        train_dataset_x=trX, train_dataset_y=trY,
        val_dataset_x=vaX,   val_dataset_y=vaY,
        test_dataset_x=teX,  test_dataset_y=teY,
        classes=np.array(classes),
        num_classes=K,
        meta=json.dumps(meta, ensure_ascii=False),
    )
    print(f"\n✅ Saved: {out_npz}")
    print("Train:", trX.shape, "| Val:", vaX.shape, "| Test:", teX.shape)

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="SUO → NPZ (robust loaders + same-stem fallback; disjoint 7:1:2; FPS no-reuse then random reuse to fill; shuffle & common rotation)")
    ap.add_argument("--root_dir", required=False, default="/Users/semin_love_u/Desktop/1st_paper/sydney-urban-objects-dataset")
    # ap.add_argument("--classes", nargs="+", required=False, default=["car", "traffic_sign", "pole"]) -> 한 번 해볼 가치는 있음
    ap.add_argument("--classes", nargs="+", required=False, default=["car", "traffic_sign", "pedestrian"]) # 가장 좋음 우선 
    ap.add_argument("--num_points", type=int, default=3)
    ap.add_argument("--train_per_class", type=int, default=700)
    ap.add_argument("--val_per_class",   type=int, default=100)
    ap.add_argument("--test_per_class",  type=int, default=200)
    ap.add_argument("--oversample_P", type=int, default=2048)
    ap.add_argument("--rotation", type=int, default=1)  # 1 → 공통 회전 적용
    ap.add_argument("--seed", type=int, default=1557)
    args = ap.parse_args()

    trX, trY, vaX, vaY, teX, teY, fb_stats = build_suo(
        root_dir=args.root_dir,
        classes=args.classes,
        num_points=args.num_points,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        test_per_class=args.test_per_class,
        oversample_P=args.oversample_P,
        rotation=args.rotation,
        seed=args.seed,
        verbose=True,
    )
    save_npz_fixed(
        (trX, trY, vaX, vaY, teX, teY, fb_stats),
        classes=args.classes,
        num_points=args.num_points,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        test_per_class=args.test_per_class,
        seed=args.seed,
        rotation=args.rotation,
        oversample_P=args.oversample_P,
    )

if __name__ == "__main__":
    main()
