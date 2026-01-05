#!/usr/bin/env python
# -*- coding: utf-8 -*-
# shapenet_npz_quota_fps_manual_verbose.py
# pip install trimesh numpy tqdm scipy scikit-learn

import os, sys, json, argparse, glob, re
import numpy as np
import trimesh
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from scipy.stats import special_ortho_group

# ---------- ShapeNetCore v2 55 클래스 (고정 인덱스 0~54) ----------
SHAPENET55 = [
    ("02691156","airplane"), ("02747177","trash bin"), ("02773838","bag"), ("02801938","basket"),
    ("02808440","bathtub"), ("02818832","bed"), ("02828884","bench"), ("02843684","birdhouse"),
    ("02871439","bookshelf"), ("02876657","bottle"), ("02880940","bowl"), ("02924116","bus"),
    ("02933112","cabinet"), ("02942699","camera"), ("02946921","can"), ("02954340","cap"),
    ("02958343","car"), ("02992529","cellphone"), ("03001627","chair"), ("03046257","clock"),
    ("03085013","keyboard"), ("03207941","dishwasher"), ("03211117","display"), ("03261776","earphone"),
    ("03325088","faucet"), ("03337140","file cabinet"), ("03467517","guitar"), ("03513137","helmet"),
    ("03593526","jar"), ("03624134","knife"), ("03636649","lamp"), ("03642806","laptop"),
    ("03691459","loudspeaker"), ("03710193","mailbox"), ("03759954","microphone"), ("03761084","microwaves"),
    ("03790512","motorbike"), ("03797390","mug"), ("03928116","piano"), ("03938244","pillow"),
    ("03948459","pistol"), ("03991062","flowerpot"), ("04004475","printer"), ("04044716","remote"),
    ("04090263","rifle"), ("04099429","rocket"), ("04225987","skateboard"), ("04256520","sofa"),
    ("04330267","stove"), ("04379243","table"), ("04401088","telephone"), ("04460130","tower"),
    ("04468005","train"), ("04530566","watercraft"), ("04554684","washer"),
]
IDX2SYN = {i:s for i,(s,_) in enumerate(SHAPENET55)}
SYN2IDX = {s:i for i,(s,_) in enumerate(SHAPENET55)}
SYN2NAME = dict(SHAPENET55)

# ---------- 유틸 ----------
def load_mesh_any(obj_path: str) -> trimesh.Trimesh:
    m = trimesh.load(obj_path, force='mesh', process=True)
    if isinstance(m, trimesh.Trimesh):
        return m
    if hasattr(m, "geometry") and len(m.geometry) > 0:
        parts = [g for g in m.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if len(parts) == 1: return parts[0]
        if len(parts)  > 1: return trimesh.util.concatenate(parts)
    raise ValueError(f"Cannot convert to Trimesh: {obj_path}")

def surface_sample(mesh: trimesh.Trimesh, n: int):
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    return pts

def normalize_unit_sphere(points: np.ndarray):
    """센터링 후 최대반경으로 나눠 단위구 스케일."""
    c = points.mean(axis=0, keepdims=True)
    centered = points - c
    scale = np.linalg.norm(centered, axis=1).max() + 1e-12
    return (centered/scale).astype(np.float32), c.squeeze(0).astype(np.float32), float(scale)

def farthest_point_sampling(points, k, candidate_idx=None, rng=None):
    if rng is None: rng = np.random.default_rng()
    P = points.shape[0]
    if candidate_idx is None:
        candidate_idx = np.arange(P, dtype=np.int32)
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
        d2 = np.minimum(d2, np.einsum("ij,ij->i", diff, diff))
        cur = int(np.argmax(d2))
        sel_local[i] = cur
    return candidate_idx[sel_local]

def always_permute_points(X, rng):
    N, k, _ = X.shape
    out = np.empty_like(X)
    for i in range(N):
        out[i] = X[i, rng.permutation(k)]
    return out

# ---------- 진행 표시가 포함된 헬퍼들 ----------
def _split_objects_disjoint(objs, quotas, rng, verbose=False, synset=None):
    """
    objs: 리스트(OBJ 경로들)
    quotas: (train_per_class, val_per_class, test_per_class)  -> 샘플 개수 '비율'로 해석
    반환: (train_objs, val_objs, test_objs)  # 세 버킷은 disjoint
    """
    n = len(objs)
    if n == 0:
        return [], [], []
    q = np.array(quotas, dtype=np.float64)
    if q.sum() == 0:
        return [], [], []
    frac = q / q.sum()
    counts = np.floor(frac * n).astype(int)
    need = int(n - counts.sum())
    order = np.argsort(-frac)
    for i in range(need):
        counts[order[i % 3]] += 1

    rng.shuffle(objs)
    a, b = counts[0], counts[1]
    train_objs = objs[:a]
    val_objs   = objs[a:a+b]
    test_objs  = objs[a+b:]
    if verbose:
        print(f"[SPLIT] {synset or ''}: total={n} → train={len(train_objs)}, val={len(val_objs)}, test={len(test_objs)}")
    return train_objs, val_objs, test_objs

def _build_pool_for_objs(objs_subset, oversample_P, rng, verbose=False, synset=None, split_name=None):
    """주어진 객체 집합으로만 cache/used_map 생성 (split 고립) + 진행 출력"""
    cache, used_map = {}, {}
    ok, fail = 0, 0
    iterator = objs_subset
    title = f"pre-sample {synset or ''} [{split_name or ''}]"
    if verbose:
        iterator = tqdm(objs_subset, desc=title, leave=False)
    for path in iterator:
        try:
            mesh = load_mesh_any(path)
            pts = surface_sample(mesh, oversample_P)
            # 표면 샘플은 전역 RNG 사용 → 빌드 시작 전에 np.random.seed(seed)로 고정
            pool, _, _ = normalize_unit_sphere(pts)  # OBJ별 1회 정규화
        except Exception:
            fail += 1
            continue
        cache[path] = pool
        used_map[path] = np.zeros(pool.shape[0], dtype=bool)
        ok += 1
    if verbose:
        print(f"[INFO] built pool for {split_name}: {ok}/{len(objs_subset)} objects loaded (fail {fail})")
    return cache, used_map

def _fill_quota_from_pool(num_points, target, cache, used_map, label_value, rng, bucket_X, bucket_Y, verbose=False, tag=""):
    """단일 pool에서만 quota를 채움(다른 split과 객체 공유 X). 반환: 실제 채운 개수"""
    if target <= 0 or len(cache) == 0:
        if verbose:
            print(f"[INFO] {tag}: nothing to fill (target={target}, cache={len(cache)})")
        return 0
    keys = list(cache.keys())
    ptr, cnt, stagnation = 0, 0, 0
    while cnt < target:
        prev = cnt
        path = keys[ptr]
        ptr = (ptr + 1) % len(keys)
        pool = cache[path]             # (P,3)
        used = used_map[path]          # (P,)
        cand = np.where(~used)[0]
        if cand.size >= num_points:
            sel = farthest_point_sampling(pool, num_points, candidate_idx=cand, rng=rng)
            used[sel] = True
            ptsk = pool[sel]           # pool은 이미 정규화됨
            bucket_X.append(ptsk[None, ...])
            bucket_Y.append(label_value)
            cnt += 1
        stagnation = stagnation + 1 if cnt == prev else 0
        if stagnation > len(keys) * 3:
            if verbose:
                print(f"[WARN] {tag}: ran out of disjoint candidates ({cnt}/{target})")
            break
    if verbose:
        print(f"[INFO] {tag}: filled {cnt}/{target}")
    return cnt

# ---------- 핵심: 3-way split (per-class quotas), 객체 단위 완전 분리 ----------
def build_dataset(
    root_dir,
    class_synsets,
    num_points,
    train_per_class,
    val_per_class,
    test_per_class,
    oversample_P=2048,
    rotation=0,
    seed=1557,
    keep_global_index=False,
    shuffle_labels=True,
    verbose=True
):
    """
    반환: (train_X, train_Y, val_X, val_Y, test_X, test_Y, valid, keep_global, label_map_out)
    - 클래스별로 OBJ 리스트를 먼저 train/val/test로 disjoint 분할
    - 각 split 풀에서만 pre-sample → 정규화 → FPS로 샘플 생성
    """
    # 전역 RNG 고정: trimesh.sample.sample_surface가 전역 RNG 사용
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # 유효 synset만 필터 & 글로벌 인덱스 순으로 정렬
    valid = [s for s in class_synsets if s in SYN2IDX]
    if not valid:
        raise ValueError("No valid synsets in class_synsets.")
    valid = sorted(valid, key=lambda s: SYN2IDX[s])

    # 라벨 매핑 (압축 or 글로벌 유지)
    label_map = {s: (SYN2IDX[s] if keep_global_index else i) for i, s in enumerate(valid)}

    # 설정 로그
    if verbose:
        print(f"[CFG] num_points={num_points}, oversample_P={oversample_P}, seed={seed}, rotation={rotation}")
        print(f"[CFG] per-class quotas: train={train_per_class}, val={val_per_class}, test={test_per_class}")
        print(f"[CFG] classes(selected)={len(valid)} → {[SYN2NAME[s] for s in valid]}")

    # 클래스별 OBJ 수집 (순서 안정화를 위해 정렬)
    per_syn_objs = {}
    for s in valid:
        syn_dir = os.path.join(root_dir, s)
        if not os.path.isdir(syn_dir):
            if verbose: print(f"[WARN] missing synset folder: {syn_dir} (0 samples)")
            per_syn_objs[s] = []
            continue
        obj_paths = glob.glob(os.path.join(syn_dir, "*", "models", "model_normalized.obj"))
        if not obj_paths:
            obj_paths = glob.glob(os.path.join(syn_dir, "**", "*.obj"), recursive=True)
        per_syn_objs[s] = sorted(obj_paths)  # <- 정렬 중요

    trX, trY = [], []
    vaX, vaY = [], []
    teX, teY = [], []

    total_tr=total_va=total_te=0

    # 클래스 루프(진행바)
    class_iter = valid if not verbose else tqdm(valid, desc="classes", leave=False)
    for s in class_iter:
        name = SYN2NAME[s]
        objs = per_syn_objs.get(s, [])
        if len(objs) == 0:
            if verbose: print(f"[INFO] synset {s} ({name}) has 0 objects; skip.")
            continue

        # 객체 리스트를 비율로 분할
        train_objs, val_objs, test_objs = _split_objects_disjoint(
            list(objs), (train_per_class, val_per_class, test_per_class), rng, verbose=verbose, synset=f"{s}({name})"
        )

        # 각 split pool 구성
        tr_cache, tr_used = _build_pool_for_objs(sorted(train_objs), oversample_P, rng, verbose, s, "train")
        va_cache, va_used = _build_pool_for_objs(sorted(val_objs),   oversample_P, rng, verbose, s, "val")
        te_cache, te_used = _build_pool_for_objs(sorted(test_objs),  oversample_P, rng, verbose, s, "test")

        # 용량(capacity) 체크: 각 OBJ에서 만들 수 있는 샘플 수 ≈ floor(P/num_points)
        def capacity(cache): return sum(int(cache[p].shape[0] // num_points) for p in cache)
        tr_cap, va_cap, te_cap = capacity(tr_cache), capacity(va_cache), capacity(te_cache)
        if verbose:
            print(f"[CAP ] {s} ({name}): cap train/val/test = {tr_cap}/{va_cap}/{te_cap} "
                  f"(quota {train_per_class}/{val_per_class}/{test_per_class})")
            if tr_cap < train_per_class: print(f"[WARN][{s}] train capacity<{train_per_class}")
            if va_cap < val_per_class:   print(f"[WARN][{s}]   val capacity<{val_per_class}")
            if te_cap < test_per_class:  print(f"[WARN][{s}]  test capacity<{test_per_class}")

        # 각 split에서만 quota 채우기
        filled_tr = _fill_quota_from_pool(num_points, train_per_class, tr_cache, tr_used, label_map[s], rng, trX, trY, verbose, f"{s} train")
        filled_va = _fill_quota_from_pool(num_points, val_per_class,   va_cache, va_used, label_map[s], rng, vaX, vaY, verbose, f"{s} val")
        filled_te = _fill_quota_from_pool(num_points, test_per_class,  te_cache, te_used, label_map[s], rng, teX, teY, verbose, f"{s} test")
        total_tr += filled_tr; total_va += filled_va; total_te += filled_te

    # 배열화 및 타입
    if len(trX) == 0 or len(vaX) == 0 or len(teX) == 0:
        print("[ERROR] No samples produced. Check quotas/oversample_P/OBJs.")
        sys.exit(1)

    trX = np.concatenate(trX, axis=0).astype(np.float32)
    vaX = np.concatenate(vaX, axis=0).astype(np.float32)
    teX = np.concatenate(teX, axis=0).astype(np.float32)

    trY = np.asarray(trY, dtype=np.int64)
    vaY = np.asarray(vaY, dtype=np.int64)
    teY = np.asarray(teY, dtype=np.int64)

    # 포인트 permutation (split별 독립적) — 같은 rng 사용해 재현성 보장
    trX = always_permute_points(trX, rng)
    vaX = always_permute_points(vaX, rng)
    teX = always_permute_points(teX, rng)

    # 라벨 셔플(선택)
    if shuffle_labels:
        idx = rng.permutation(len(trX)); trX, trY = trX[idx], trY[idx]
        idx = rng.permutation(len(vaX)); vaX, vaY = vaX[idx], vaY[idx]
        idx = rng.permutation(len(teX)); teX, teY = teX[idx], teY[idx]
        if verbose: print("[INFO] Labels shuffled for train/val/test")

    # 공통 회전(선택)
    if rotation == 1:
        R = special_ortho_group.rvs(3, random_state=int(rng.integers(0, 1<<31)))
        trX = trX @ R.T
        vaX = vaX @ R.T
        teX = teX @ R.T
        if verbose: print("[AUG ] Applied one common SO(3) rotation to all splits.")

    keep_global = [SYN2IDX[s] for s in valid]
    label_map_out = {s:int(label_map[s]) for s in valid}

    if verbose:
        print(f"[DONE] filled totals → train={total_tr}, val={total_va}, test={total_te}")

    return trX, trY, vaX, vaY, teX, teY, valid, keep_global, label_map_out

# ---------- CLI & 수동 실행 ----------
def run_cli():
    ap = argparse.ArgumentParser(description="Subset ShapeNetCore(OBJ) → NPZ (disjoint per-object splits; normalized pool + FPS)")
    ap.add_argument("--root_dir", required=True)
    ap.add_argument("--class_synsets", nargs="+", required=False)
    ap.add_argument("--num_points", type=int, default=1024)

    # per-class quotas ONLY
    ap.add_argument("--train_per_class", type=int, required=True)
    ap.add_argument("--val_per_class",   type=int, required=True)
    ap.add_argument("--test_per_class",  type=int, required=True)

    ap.add_argument("--oversample_P", type=int, default=2048)
    ap.add_argument("--rotation", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1557)
    ap.add_argument("--keep_global_index", action="store_true", help="Keep global 0..54 labels")
    ap.add_argument("--no_shuffle", action="store_true", help="Do NOT shuffle labels")
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--no_verbose", action="store_true", help="suppress progress logs")
    args = ap.parse_args()

    syn_pat = re.compile(r'^\d{8}$')
    if args.class_synsets:
        sel = [s for s in args.class_synsets if syn_pat.match(s)]
    else:
        sel = sorted([d for d in os.listdir(args.root_dir) if syn_pat.match(d)])

    trX, trY, vaX, vaY, teX, teY, syns, keep_global, label_map_out = build_dataset(
        root_dir=args.root_dir,
        class_synsets=sel,
        num_points=args.num_points,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        test_per_class=args.test_per_class,
        oversample_P=args.oversample_P,
        rotation=args.rotation,
        seed=args.seed,
        keep_global_index=args.keep_global_index,
        shuffle_labels=not args.no_shuffle,
        verbose=not args.no_verbose
    )

    num_classes = len(syns)
    fname = f"shapenet_{num_classes}classes_{args.num_points}_1_fps_train{args.train_per_class}_val{args.val_per_class}_test{args.test_per_class}_new.npz"
    out_npz = args.out_npz if args.out_npz.endswith(".npz") else os.path.join(args.out_npz, fname)
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)

    meta = {
        "version": "shapenet_quota_fps_v4_3way_disjoint",
        "class_synsets": syns,
        "class_names": [SYN2NAME[s] for s in syns],
        "keep_global_index": bool(args.keep_global_index),
        "global_indices": keep_global,
        "label_map": label_map_out,
        "num_classes": int(num_classes),
        "num_points": int(args.num_points),
        "train_per_class": int(args.train_per_class),
        "val_per_class":   int(args.val_per_class),
        "test_per_class":  int(args.test_per_class),
        "oversample_P": int(args.oversample_P),
        "rotation": int(args.rotation),
        "seed": int(args.seed),
        "shuffled": not args.no_shuffle,
        "disjoint_objects": True
    }

    np.savez_compressed(
        out_npz,
        train_dataset_x=trX, train_dataset_y=trY,
        val_dataset_x=vaX,   val_dataset_y=vaY,
        test_dataset_x=teX,  test_dataset_y=teY,
        labels_used=np.array(keep_global if args.keep_global_index else list(range(num_classes)), dtype=np.int64),
        class_synsets=np.array(syns),
        num_classes=num_classes,
        meta=json.dumps(meta, ensure_ascii=False)
    )
    print(f"✅ Saved: {out_npz}")
    print("Train:", trX.shape, "| Val:", vaX.shape, "| Test:", teX.shape)

if __name__ == "__main__":
    USE_MANUAL = True

    if USE_MANUAL:
        ROOT_DIR = "/Users/semin_love_u/Desktop/1st_paper"
        CLASS_SYNSETS = ["02843684", "02876657", "02880940", "02924116", "02954340"]
        NUM_POINTS       = 3
        TRAIN_PER_CLASS  = 700
        VAL_PER_CLASS    = 100
        TEST_PER_CLASS   = 200
        OVERSAMPLE_P     = 2048
        ROTATION         = 1
        SEED             = 1557
        KEEP_GLOBAL_INDEX = False
        SHUFFLE_LABELS    = True
        VERBOSE           = True

        syn_pat = re.compile(r'^\d{8}$')
        if CLASS_SYNSETS:
            sel = [s for s in CLASS_SYNSETS if syn_pat.match(s)]
        else:
            sel = sorted([d for d in os.listdir(ROOT_DIR) if syn_pat.match(d)])

        trX, trY, vaX, vaY, teX, teY, syns, keep_global, label_map_out = build_dataset(
            root_dir=ROOT_DIR,
            class_synsets=sel,
            num_points=NUM_POINTS,
            train_per_class=TRAIN_PER_CLASS,
            val_per_class=VAL_PER_CLASS,
            test_per_class=TEST_PER_CLASS,
            oversample_P=OVERSAMPLE_P,
            rotation=ROTATION,
            seed=SEED,
            keep_global_index=KEEP_GLOBAL_INDEX,
            shuffle_labels=SHUFFLE_LABELS,
            verbose=VERBOSE
        )

        num_classes = len(syns)
        fname = f"shapenet_{num_classes}classes_{NUM_POINTS}_1_fps_train{TRAIN_PER_CLASS}_val{VAL_PER_CLASS}_test{TEST_PER_CLASS}_new.npz"
        OUT_NPZ = f"/Users/semin_love_u/Desktop/1st_paper/{fname}"
        os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)

        meta = {
            "version":"shapenet_quota_fps_v4_3way_disjoint",
            "class_synsets": syns,
            "class_names": [SYN2NAME[s] for s in syns],
            "keep_global_index": bool(KEEP_GLOBAL_INDEX),
            "global_indices": keep_global,
            "label_map": label_map_out,
            "num_classes": int(num_classes),
            "num_points": int(NUM_POINTS),
            "train_per_class": int(TRAIN_PER_CLASS),
            "val_per_class":   int(VAL_PER_CLASS),
            "test_per_class":  int(TEST_PER_CLASS),
            "oversample_P": int(OVERSAMPLE_P),
            "rotation": int(ROTATION),
            "seed": int(SEED),
            "shuffled": bool(SHUFFLE_LABELS),
            "disjoint_objects": True
        }

        np.savez_compressed(
            OUT_NPZ,
            train_dataset_x=trX, train_dataset_y=trY,
            val_dataset_x=vaX,   val_dataset_y=vaY,
            test_dataset_x=teX,  test_dataset_y=teY,
            labels_used=np.array(keep_global if KEEP_GLOBAL_INDEX else list(range(num_classes)), dtype=np.int64),
            class_synsets=np.array(syns),
            num_classes=num_classes,
            meta=json.dumps(meta, ensure_ascii=False)
        )

        print(f"✅ Saved: {OUT_NPZ}")
        print("Train:", trX.shape, "| Val:", vaX.shape, "| Test:", teX.shape)
    else:
        run_cli()
