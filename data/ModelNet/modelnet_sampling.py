#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ModelNet40 HDF5 -> NPZ (subset K classes, object-disjoint train/val; official test kept)
- Loads official train/test H5 pools separately
- Filters to chosen class labels and relabels to 0..K-1 (given order)
- Splits TRAIN objects into train/val disjointly (by row index = object)
- TEST samples come ONLY from official test pool (disjoint by construction)
- Per-object point sampling: FPS (unified implementation)
- Optional shared random rotation
- Auto output filename: modelnet40_{K}classes_{points_per_sample}_{num_reupload}_{sampling}_train{...}_val{...}_test{...}_new.npz
"""

import os, glob, json, argparse
import numpy as np
import h5py
from scipy.stats import special_ortho_group

# -------------------- H5 discovery --------------------
def discover_modelnet40_files(data_dir):
    train_patterns = ["ply_data_train*.h5", "*train*.h5", "modelnet40_train*.h5"]
    test_patterns  = ["ply_data_test*.h5",  "*test*.h5",  "modelnet40_test*.h5"]
    train_files, test_files = [], []
    for pat in train_patterns:
        train_files += glob.glob(os.path.join(data_dir, pat))
    for pat in test_patterns:
        test_files  += glob.glob(os.path.join(data_dir, pat))
    train_files = sorted(set(train_files))
    test_files  = sorted(set(test_files))
    if not train_files or not test_files:
        raise FileNotFoundError(f"Could not find full ModelNet40 HDF5 sets under {data_dir}")
    print("[discover] train files:", [os.path.basename(x) for x in train_files])
    print("[discover] test  files:", [os.path.basename(x) for x in test_files])
    return train_files, test_files

# -------------------- Load H5 --------------------
def _read_one_h5(path):
    with h5py.File(path, 'r') as f:
        if 'data' not in f or 'label' not in f:
            raise KeyError(f"{path} missing 'data' or 'label'")
        data  = f['data'][:]               # (N, P, 3)
        label = f['label'][:].squeeze()    # (N,) or (N,1)->(N,)
        normal= f['normal'][:] if 'normal' in f else None
        return data, label.astype(np.int64), normal

def load_pool(paths):
    all_X, all_y, all_N = [], [], []
    for p in paths:
        X, y, N = _read_one_h5(p)
        print(f"[load] {os.path.basename(p)} -> X:{X.shape} y:{y.shape}" + (f" normal:{N.shape}" if N is not None else ""))
        all_X.append(X); all_y.append(y)
        if N is not None: all_N.append(N)
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    N = np.concatenate(all_N, axis=0) if len(all_N)==len(paths) else None
    return X, y, N

# -------------------- Filter & relabel --------------------
def filter_and_relabel(X, y, N, class_labels):
    class_labels = list(class_labels)
    keep = np.isin(y, class_labels)
    Xf = X[keep]; yf = y[keep]
    Nf = N[keep] if N is not None else None
    label_map = {orig:i for i,orig in enumerate(class_labels)}
    ynew = np.array([label_map[int(t)] for t in yf], dtype=np.int64)
    print(f"[filter] kept {len(Xf)} objects across {len(class_labels)} classes; map={label_map}")
    return Xf, ynew, Nf, label_map

# -------------------- Unified FPS --------------------
def farthest_point_sampling(points, k, candidate_idx=None, rng=None):
    """Unified FPS implementation using einsum (consistent with other scripts)"""
    if rng is None:
        rng = np.random.default_rng()
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

def sample_points(object_points, m, method="fps", rng=None):
    N = object_points.shape[0]
    if rng is None: rng = np.random.default_rng()
    if N < m:
        rep = (m + N - 1) // N
        idx = np.tile(np.arange(N), rep)[:m]
        return object_points[idx]
    if method == "random":
        idx = rng.choice(N, m, replace=False)
    elif method == "fps":
        idx = farthest_point_sampling(object_points, m, rng=rng)
    else:
        raise ValueError("sampling must be 'fps' or 'random'")
    return object_points[idx]

# -------------------- Split helpers --------------------
def split_train_val_by_objects(num_obj, train_needed, val_needed, rng):
    """Return (train_indices, val_indices) of object rows (disjoint) from [0..num_obj)."""
    order = rng.permutation(num_obj)
    a = min(num_obj, train_needed)
    b = min(max(num_obj - a, 0), val_needed)
    train_idx = order[:a]
    val_idx   = order[a:a+b]
    return train_idx, val_idx

# -------------------- Main pipeline --------------------
def run(
    h5_dir,
    out_dir,
    class_labels,
    points_per_sample=4,
    num_reupload=1,
    train_per_class=700,
    val_per_class=100,
    test_per_class=200,
    sampling="fps",
    rotation=0,
    seed=42,
):
    # Seed fixing
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # 1) discover & load official pools
    tr_files, te_files = discover_modelnet40_files(h5_dir)
    Xtr_all, ytr_all, _ = load_pool(tr_files)
    Xte_all, yte_all, _ = load_pool(te_files)

    # 2) filter & relabel separately (keeps official split boundary)
    Xtr, ytr, _, label_map = filter_and_relabel(Xtr_all, ytr_all, None, class_labels)
    Xte, yte, _, _         = filter_and_relabel(Xte_all, yte_all, None, class_labels)
    K = len(class_labels)
    P_tr = Xtr.shape[1]; P_te = Xte.shape[1]
    assert P_tr == P_te, "Train/Test points-per-object mismatch"
    print(f"[info] points-per-object in H5: {P_tr}")

    # 3) per-class object lists
    cls_indices_tr = {c: np.where(ytr == c)[0] for c in np.unique(ytr)}
    cls_indices_te = {c: np.where(yte == c)[0] for c in np.unique(yte)}

    # 4) build splits (object-disjoint)
    # ModelNet40 provides only official train/test splits.
    # We further split the TRAIN pool into train/val (object-disjoint)
    # to prevent any object from appearing in multiple splits.
    trX, trY, vaX, vaY, teX, teY = [], [], [], [], [], []
    for c in range(K):
        obj_idx_tr = cls_indices_tr.get(c, np.array([], dtype=np.int64))
        obj_idx_te = cls_indices_te.get(c, np.array([], dtype=np.int64))

        # train/val from official TRAIN pool only
        tr_objs_rel, va_objs_rel = split_train_val_by_objects(len(obj_idx_tr), train_per_class, val_per_class, rng)
        tr_objs = obj_idx_tr[tr_objs_rel]
        va_objs = obj_idx_tr[va_objs_rel]

        # test from official TEST pool only
        rng.shuffle(obj_idx_te)
        te_take = min(len(obj_idx_te), test_per_class)
        te_objs = obj_idx_te[:te_take]

        if te_take < test_per_class:
            print(f"[warn][cls {c}] test pool has only {len(obj_idx_te)} objects < {test_per_class}; will cycle within TEST split.")

        # sampling per split
        def sample_from_objs(Xpool, obj_rows, target, bucket_X, bucket_Y, label_id):
            if len(obj_rows) == 0:
                print(f"[warn][cls {label_id}] no objects in this split; duplicating first available TRAIN obj into this split (still disjoint across splits).")
                if len(cls_indices_tr.get(label_id, [])) == 0:
                    return
                obj_rows = cls_indices_tr[label_id][:1]
            for i in range(target):
                obj = obj_rows[i % len(obj_rows)]
                pts = Xpool[obj]
                sel = sample_points(pts, points_per_sample, method=sampling, rng=rng)
                bucket_X.append(sel[None, ...])
                bucket_Y.append(label_id)

        sample_from_objs(Xtr, tr_objs, train_per_class, trX, trY, c)
        sample_from_objs(Xtr, va_objs, val_per_class,   vaX, vaY, c)
        sample_from_objs(Xte, te_objs, test_per_class,  teX, teY, c)

    # 5) stack
    trX = np.concatenate(trX, axis=0).astype(np.float32)
    vaX = np.concatenate(vaX, axis=0).astype(np.float32)
    teX = np.concatenate(teX, axis=0).astype(np.float32)
    trY = np.asarray(trY, dtype=np.int64)
    vaY = np.asarray(vaY, dtype=np.int64)
    teY = np.asarray(teY, dtype=np.int64)

    # 6) optional shared rotation
    if rotation == 1:
        R = special_ortho_group.rvs(3, random_state=int(rng.integers(0, 1<<31)))
        trX = trX @ R.T
        vaX = vaX @ R.T
        teX = teX @ R.T

    # 7) shuffle inside each split
    def shuffle_xy(X, Y, rng):
        perm = rng.permutation(len(Y))
        return X[perm], Y[perm]
    trX, trY = shuffle_xy(trX, trY, rng)
    vaX, vaY = shuffle_xy(vaX, vaY, rng)
    teX, teY = shuffle_xy(teX, teY, rng)

    # 8) save
    os.makedirs(out_dir, exist_ok=True)
    fname = f"modelnet40_{K}classes_{points_per_sample}_{num_reupload}_{sampling}_train{train_per_class}_val{val_per_class}_test{test_per_class}_new.npz"
    out_npz = os.path.join(out_dir, fname)

    meta = {
        "source": "ModelNet40-HDF5",
        "class_labels_original": list(class_labels),
        "label_map_to_compact": {int(k): int(v) for k, v in {k: i for i, k in enumerate(class_labels)}.items()},
        "num_classes": int(K),
        "points_per_sample": int(points_per_sample),
        "num_reupload": int(num_reupload),
        "train_per_class": int(train_per_class),
        "val_per_class": int(val_per_class),
        "test_per_class": int(test_per_class),
        "sampling": sampling,
        "rotation": int(rotation),
        "seed": int(seed),
        "official_test_kept": True,
        "object_disjoint_train_val": True
    }

    np.savez_compressed(
        out_npz,
        train_dataset_x=trX, train_dataset_y=trY,
        val_dataset_x=vaX,   val_dataset_y=vaY,
        test_dataset_x=teX,  test_dataset_y=teY,
        num_classes=K,
        meta=json.dumps(meta, ensure_ascii=False)
    )
    print(f"\n✅ Saved: {out_npz}")
    print("Train:", trX.shape, "| Val:", vaX.shape, "| Test:", teX.shape)

# -------------------- CLI / No-arg default --------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        run(
            h5_dir="/Users/semin_love_u/Desktop/1st_paper/modelnet40_ply_hdf5_2048",
            out_dir="/Users/semin_love_u/Desktop/1st_paper",
            class_labels=[5, 6, 10, 19, 32],
            points_per_sample=3 ,
            num_reupload=1,
            train_per_class=700,
            val_per_class=100,
            test_per_class=200,
            sampling="fps",
            rotation=0,
            seed=42,
        )
    else:
        ap = argparse.ArgumentParser(description="ModelNet40 -> NPZ (subset & object-disjoint train/val; official test kept)")
        ap.add_argument("--h5_dir", required=True)
        ap.add_argument("--out_dir", required=True)
        ap.add_argument("--class_labels", nargs="+", type=int, required=False)
        ap.add_argument("--points_per_sample", type=int, default=4)
        ap.add_argument("--num_reupload", type=int, default=1)
        ap.add_argument("--train_per_class", type=int, default=700)
        ap.add_argument("--val_per_class",   type=int, default=100)
        ap.add_argument("--test_per_class",  type=int, default=200)
        ap.add_argument("--sampling", choices=["fps","random"], default="fps")
        ap.add_argument("--rotation", type=int, default=0)
        ap.add_argument("--seed", type=int, default=42)
        args = ap.parse_args()

        labels = args.class_labels if args.class_labels else [5, 6, 10, 19, 32]
        run(
            h5_dir=args.h5_dir,
            out_dir=args.out_dir,
            class_labels=labels,
            points_per_sample=args.points_per_sample,
            num_reupload=args.num_reupload,
            train_per_class=args.train_per_class,
            val_per_class=args.val_per_class,
            test_per_class=args.test_per_class,
            sampling=args.sampling,
            rotation=args.rotation,
            seed=args.seed,
        )