#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, glob, json, argparse, importlib.util, re
from pathlib import Path
import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.stats import special_ortho_group

_PRIORITIES = [
    (".csv", ".CSV"),
    (".npy", ".NPY"),
    (".npz", ".NPZ"),
    (".ply", ".PLY"),
    (".pcd", ".PCD"),
    (".bin", ".BIN"),
]

_FLOAT_RE = re.compile(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?')

def _maybe_import_read_bin(root_dir: str):
    path = os.path.join(root_dir, "read-bin.py")
    if not os.path.isfile(path):
        return None
    spec = importlib.util.spec_from_file_location("suo_read_bin", path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        fn = getattr(mod, "read_bin", None)
        return fn if callable(fn) else None
    except Exception:
        return None

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
                    chosen[sk] = p
    return chosen

def discover_objects(root_dir: str, classes: List[str]) -> Dict[str, List[str]]:
    obj_root = os.path.join(root_dir, "objects")
    if not os.path.isdir(obj_root):
        raise FileNotFoundError(f"Missing folder: {obj_root}")

    classes_lc = [c.lower() for c in classes]
    out: Dict[str, List[str]] = {cls: [] for cls in classes}

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
            print(f"[DISCOVER] {cls}: {len(files)} files. e.g., {examples}")
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
        print(f"[DISCOVER] {cls}: {len(files)} files. e.g., {examples}")
    return out

def _detect_header_and_cols(first_line: str):
    for d in [",", "\t", ";", " ", "|"]:
        parts = [p.strip().lower() for p in first_line.strip().split(d)]
        if len(parts) >= 3 and {"x", "y", "z"}.issubset(set(parts)):
            return True, (parts.index("x"), parts.index("y"), parts.index("z"))
    has_alpha = any(ch.isalpha() for ch in first_line if ch not in "eE+-.,;|\t ")
    return (True if has_alpha else False), (3, 4, 5)

def _load_csv_xyz(path: str) -> np.ndarray:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            first = f.readline()
    except Exception:
        first = ""
    has_header_hint, target_cols = _detect_header_and_cols(first)

    cand_delims = [",", "\t", ";", " ", "|"]
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

    xs, ys, zs = [], [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            nums = _FLOAT_RE.findall(line)
            if len(nums) >= 6:
                try:
                    xs.append(float(nums[3])); ys.append(float(nums[4])); zs.append(float(nums[5]))
                except Exception:
                    continue
            elif len(nums) >= 3 and not has_header_hint:
                try:
                    xs.append(float(nums[0])); ys.append(float(nums[1])); zs.append(float(nums[2]))
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
        if arr.ndim == 2 and arr.shape[1] >= 3:
            return arr[:, :3].astype(np.float32)
        raise ValueError(f".npy not (N,>=3): {arr.shape}")
    if ext == ".npz":
        with np.load(path) as z:
            for key in ("points", "xyz", "XYZ", "data"):
                if key in z:
                    arr = np.asarray(z[key])
                    break
            else:
                ks = [k for k in z.files if hasattr(z[k], "shape")]
                if not ks:
                    raise ValueError(".npz has no arrays")
                arr = np.asarray(z[ks[0]])
        if arr.ndim == 2 and arr.shape[1] >= 3:
            return arr[:, :3].astype(np.float32)
        raise ValueError(f".npz not (N,>=3): {arr.shape}")
    if ext in (".ply", ".pcd"):
        try:
            import open3d as o3d
        except Exception as e:
            raise ValueError("open3d not installed for ply/pcd") from e
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError("pcd/ply has <3 dims")
        return pts[:, :3].astype(np.float32)
    if ext == ".bin":
        if read_bin_fn is not None:
            pts = np.asarray(read_bin_fn(path))
            if pts.ndim != 2 or pts.shape[1] < 3:
                raise ValueError(f"read_bin bad shape {pts.shape}")
            return pts[:, :3].astype(np.float32)
        b32 = np.fromfile(path, dtype=np.float32)
        for d in (3,4,5,6,7,8,9,10,12,16):
            if b32.size % d == 0 and b32.size >= d:
                arr = b32.reshape(-1, d)[:, :3]
                if arr.shape[0] > 0:
                    return arr.astype(np.float32)
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
        b64 = np.fromfile(path, dtype=np.float64)
        for d in (3,4,5,6,8,10,12,16):
            if b64.size % d == 0 and b64.size >= d:
                arr = b64.reshape(-1, d)[:, :3]
                if arr.shape[0] > 0:
                    return arr.astype(np.float32)
        raise ValueError("Unrecognized .bin layout (prefer CSV or provide read-bin.py)")
    raise ValueError(f"Unsupported extension: {ext}")

def _stem_from_path(path: str) -> str:
    base = os.path.basename(path)
    sk = _stem_for_pick(base)
    if sk is None:
        raise ValueError("bad stem")
    return sk

def _candidate_paths_for_same_stem(path: str) -> List[str]:
    d = os.path.dirname(path)
    stem = _stem_from_path(path)
    cands: List[str] = []
    for group in _PRIORITIES:
        for ext in group:
            p = os.path.join(d, stem + ext)
            if os.path.isfile(p):
                cands.append(p)
    cands = [p for p in cands if not p.lower().endswith(".bin.meta")]
    seen=set(); uniq=[]
    for p in cands:
        if p not in seen:
            uniq.append(p); seen.add(p)
    return uniq

def normalize_unit_sphere(points: np.ndarray):
    c = points.mean(axis=0, keepdims=True)
    centered = points - c
    scale = float(np.linalg.norm(centered, axis=1).max() + 1e-12)
    return (centered / scale).astype(np.float32), c.squeeze(0).astype(np.float32), scale

def farthest_point_sampling(points: np.ndarray, k: int, candidate_idx=None, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    if candidate_idx is None:
        candidate_idx = np.arange(points.shape[0], dtype=np.int32)
    if candidate_idx.size < k:
        raise ValueError("insufficient_candidates")
    cand = points[candidate_idx]
    C = cand.shape[0]
    sel_local = np.empty(k, dtype=np.int32)
    cur = int(rng.integers(0, C))
    sel_local[0] = cur
    d2 = np.full(C, np.inf, dtype=np.float64)
    for i in range(1, k):
        diff = cand - cand[cur]
        d2 = np.minimum(d2, np.einsum('ij,ij->i', diff, diff))
        cur = int(np.argmax(d2))
        sel_local[i] = cur
    return candidate_idx[sel_local]

def always_permute_points(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.empty_like(X)
    for i in range(X.shape[0]):
        out[i] = X[i, rng.permutation(X.shape[1])]
    return out

def split_objects_disjoint(objs: List[str], rng, ratio=(7,1,2)) -> Tuple[List[str], List[str], List[str]]:
    n = len(objs)
    if n == 0:
        return [], [], []
    a,b,c = ratio
    s = a+b+c
    frac = np.array([a,b,c], dtype=np.float64) / float(s)
    counts = np.floor(frac * n).astype(int)
    need = n - counts.sum()
    order = np.argsort(-frac)
    for i in range(need):
        counts[order[i % 3]] += 1
    rng.shuffle(objs)
    x, y = counts[0], counts[1]
    return objs[:x], objs[x:x+y], objs[x+y:]

def build_pools_for_split(file_list: List[str], oversample_P: int, loader_read_bin, rng, verbose=False):
    cache, used_map, errors = {}, {}, {}
    ok = fail = 0
    for chosen_path in file_list:
        tried = []
        success = False
        last_err = "unknown"
        candidates = _candidate_paths_for_same_stem(chosen_path)
        if chosen_path in candidates:
            candidates.remove(chosen_path)
        candidates = [chosen_path] + candidates
        for path in candidates:
            tried.append(os.path.basename(path))
            try:
                pts_all = _load_points_generic(path, read_bin_fn=loader_read_bin)
                pts_norm, _, _ = normalize_unit_sphere(pts_all)
                if pts_norm.shape[0] <= oversample_P:
                    pool = pts_norm
                else:
                    pool = pts_norm[rng.choice(pts_norm.shape[0], size=oversample_P, replace=False)]
                cache[path] = pool
                used_map[path] = np.zeros(pool.shape[0], dtype=bool)
                ok += 1
                success = True
                break
            except Exception as e:
                last_err = str(e)
                continue
        if not success:
            fail += 1
            errors[chosen_path] = f"tried={tried} | last_err={last_err}"
    print(f"[POOL] built: ok={ok}, fail={fail}")
    for p, msg in list(errors.items())[:10]:
        print(f"   - {os.path.basename(p)} → {msg}")
    return cache, used_map

def capacity(cache: Dict[str, np.ndarray], num_points: int) -> int:
    return sum(int(cache[p].shape[0] // num_points) for p in cache)

def fill_quota_with_fps_then_random(num_points: int, target: int,
                                    cache: Dict[str, np.ndarray], used_map: Dict[str, np.ndarray],
                                    label_value: int, rng,
                                    bucket_X: List[np.ndarray], bucket_Y: List[int],
                                    verbose=False, tag="") -> Tuple[int, int]:
    if target <= 0 or len(cache) == 0:
        return 0, 0

    keys = list(cache.keys())
    ptr = 0
    cnt = 0

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
    fallback = 0
    while cnt < target and len(keys) > 0:
        path = keys[int(rng.integers(0, len(keys)))]
        pool = cache[path]
        C = pool.shape[0]
        if C == 0:
            continue
        if C >= num_points:
            sel = rng.choice(C, size=num_points, replace=False)
        else:
            order = rng.permutation(C)
            sel = np.resize(order, num_points)
        bucket_X.append(pool[sel][None, ...])
        bucket_Y.append(label_value)
        cnt += 1
        fallback += 1

    if verbose:
        print(f"[INFO] {tag}: FPS(no-reuse) {first_pass}/{target}; fallback(random) +{fallback}")
    return cnt, fallback

def build_suo(root_dir: str, classes: List[str], num_points: int,
              train_per_class: int, val_per_class: int, test_per_class: int,
              oversample_P: int, rotation: int, seed: int, verbose: bool):
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    class_to_files = discover_objects(root_dir, classes)

    splits = {}
    for cls in classes:
        files = sorted(list(class_to_files.get(cls, [])))
        tr, va, te = split_objects_disjoint(files, rng, ratio=(7,1,2))
        splits[cls] = (tr, va, te)
        if verbose:
            print(f"[SPLIT] {cls}: total={len(files)} → train={len(tr)}, val={len(va)}, test={len(te)}")

    read_bin_fn = _maybe_import_read_bin(root_dir)

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

    def cap_subset(split_files, cache_dict):
        stems = {os.path.splitext(os.path.basename(p))[0] for p in split_files}
        sub = {k:v for k,v in cache_dict.items() if os.path.splitext(os.path.basename(k))[0] in stems}
        return capacity(sub, num_points)

    for cls in classes:
        tr, va, te = splits[cls]
        cap_tr = cap_subset(tr, tr_cache)
        cap_va = cap_subset(va, va_cache)
        cap_te = cap_subset(te, te_cache)
        print(f"[CAP ] {cls}: train/val/test = {cap_tr}/{cap_va}/{cap_te}  (quota {train_per_class}/{val_per_class}/{test_per_class})")
        if cap_tr < train_per_class or cap_va < val_per_class or cap_te < test_per_class:
            print(f"[WARN] {cls}: capacity shortage; will fill remaining with random.")

    label_map = {cls: i for i, cls in enumerate(classes)}
    trX, trY, vaX, vaY, teX, teY = [], [], [], [], [], []
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

        tr_used_sub = {k: tr_used[k] for k in tr_sub}
        va_used_sub = {k: va_used[k] for k in va_sub}
        te_used_sub = {k: te_used[k] for k in te_sub}

        _, fb = fill_quota_with_fps_then_random(num_points, train_per_class, tr_sub, tr_used_sub, lt, rng, trX, trY, verbose, f"{cls} train")
        fb_stats["train"] += fb
        _, fb = fill_quota_with_fps_then_random(num_points, val_per_class, va_sub, va_used_sub, lt, rng, vaX, vaY, verbose, f"{cls} val")
        fb_stats["val"] += fb
        _, fb = fill_quota_with_fps_then_random(num_points, test_per_class, te_sub, te_used_sub, lt, rng, teX, teY, verbose, f"{cls} test")
        fb_stats["test"] += fb

    trX = np.concatenate(trX, axis=0).astype(np.float32)
    vaX = np.concatenate(vaX, axis=0).astype(np.float32)
    teX = np.concatenate(teX, axis=0).astype(np.float32)
    trY = np.asarray(trY, dtype=np.int64)
    vaY = np.asarray(vaY, dtype=np.int64)
    teY = np.asarray(teY, dtype=np.int64)

    trX = always_permute_points(trX, rng)
    vaX = always_permute_points(vaX, rng)
    teX = always_permute_points(teX, rng)

    idx = rng.permutation(len(trX)); trX, trY = trX[idx], trY[idx]
    idx = rng.permutation(len(vaX)); vaX, vaY = vaX[idx], vaY[idx]
    idx = rng.permutation(len(teX)); teX, teY = teX[idx], teY[idx]

    if rotation == 1:
        R = special_ortho_group.rvs(3, random_state=int(rng.integers(0, 1<<31)))
        trX = trX @ R.T
        vaX = vaX @ R.T
        teX = teX @ R.T
        print("[AUG ] Applied one common SO(3) rotation to all splits.")

    print(f"[DONE] totals → train={len(trX)} val={len(vaX)} test={len(teX)}")
    print(f"[FALLBACK] random-with-reuse → train:{fb_stats['train']} val:{fb_stats['val']} test:{fb_stats['test']}")
    return trX, trY, vaX, vaY, teX, teY, fb_stats

def main():
    SCRIPT_DIR = Path(__file__).resolve().parent
    ROOT_DIR = SCRIPT_DIR / "sydney-urban-objects-dataset"

    CLASSES = ["car", "traffic_sign", "pedestrian"]
    TRAIN_PER_CLASS = 700
    VAL_PER_CLASS = 100
    TEST_PER_CLASS = 200
    OVERSAMPLE_P = 2048
    ROTATION = 1
    SEED = 1557

    ap = argparse.ArgumentParser()
    ap.add_argument("--num_points", type=int, required=True)
    args = ap.parse_args()

    if not ROOT_DIR.is_dir():
        raise FileNotFoundError(f"Missing dataset folder: {ROOT_DIR} (place it next to this script)")

    trX, trY, vaX, vaY, teX, teY, fb_stats = build_suo(
        root_dir=str(ROOT_DIR),
        classes=CLASSES,
        num_points=int(args.num_points),
        train_per_class=TRAIN_PER_CLASS,
        val_per_class=VAL_PER_CLASS,
        test_per_class=TEST_PER_CLASS,
        oversample_P=OVERSAMPLE_P,
        rotation=ROTATION,
        seed=SEED,
        verbose=True,
    )

    K = len(CLASSES)
    fname = f"sydney_urban_objects_{K}classes_{int(args.num_points)}_1_fps_train{TRAIN_PER_CLASS}_val{VAL_PER_CLASS}_test{TEST_PER_CLASS}_new.npz"
    out_npz = SCRIPT_DIR / fname

    meta = {
        "version": "suo_quota_fps_then_random_v1",
        "classes": CLASSES,
        "num_classes": int(K),
        "num_points": int(args.num_points),
        "train_per_class": int(TRAIN_PER_CLASS),
        "val_per_class": int(VAL_PER_CLASS),
        "test_per_class": int(TEST_PER_CLASS),
        "oversample_P": int(OVERSAMPLE_P),
        "rotation": int(ROTATION),
        "seed": int(SEED),
        "shuffled": True,
        "disjoint_objects": True,
        "split_ratio": [7,1,2],
        "normalize_policy": "per-object full points → unit sphere",
        "sampling_policy": "Phase1 FPS no-reuse; Phase2 random reuse to meet quota",
        "fallback_counts": fb_stats,
    }

    np.savez_compressed(
        out_npz,
        train_dataset_x=trX, train_dataset_y=trY,
        val_dataset_x=vaX,   val_dataset_y=vaY,
        test_dataset_x=teX,  test_dataset_y=teY,
        classes=np.array(CLASSES),
        num_classes=K,
        meta=json.dumps(meta, ensure_ascii=False),
    )

    print(f"\n✅ Saved: {out_npz}")
    print("Train:", trX.shape, "| Val:", vaX.shape, "| Test:", teX.shape)

if __name__ == "__main__":
    main()

