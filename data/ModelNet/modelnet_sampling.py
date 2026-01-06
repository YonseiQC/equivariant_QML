import os, glob, json, argparse
from pathlib import Path
import numpy as np
import h5py
from scipy.stats import special_ortho_group

CLASS_LABELS = [5, 6, 10, 19, 32]
TRAIN_PER_CLASS = 700
VAL_PER_CLASS = 100
TEST_PER_CLASS = 200
SAMPLING = "fps"
ROTATION = 1
SEED = 1557
NUM_REUPLOAD = 1
H5_FOLDER_NAME = "modelnet40_ply_hdf5_2048"

def discover_modelnet40_files(data_dir):
    train_patterns = ["ply_data_train*.h5", "*train*.h5", "modelnet40_train*.h5", "train*.h5"]
    test_patterns  = ["ply_data_test*.h5",  "*test*.h5",  "modelnet40_test*.h5",  "test*.h5"]
    train_files, test_files = [], []
    for pat in train_patterns:
        train_files += glob.glob(os.path.join(str(data_dir), pat))
    for pat in test_patterns:
        test_files  += glob.glob(os.path.join(str(data_dir), pat))
    train_files = sorted(set(train_files))
    test_files  = sorted(set(test_files))
    if not train_files or not test_files:
        raise FileNotFoundError(f"Could not find full ModelNet40 HDF5 sets under {data_dir}")
    print("[discover] train files:", [os.path.basename(x) for x in train_files])
    print("[discover] test  files:", [os.path.basename(x) for x in test_files])
    return train_files, test_files

def _read_one_h5(path):
    with h5py.File(path, "r") as f:
        if "data" not in f or "label" not in f:
            raise KeyError(f"{path} missing 'data' or 'label'")
        data = f["data"][:]
        label = f["label"][:].squeeze()
        normal = f["normal"][:] if "normal" in f else None
        return data, label.astype(np.int64), normal

def load_pool(paths):
    all_X, all_y, all_N = [], [], []
    for p in paths:
        X, y, N = _read_one_h5(p)
        print(f"[load] {os.path.basename(p)} -> X:{X.shape} y:{y.shape}" + (f" normal:{N.shape}" if N is not None else ""))
        all_X.append(X)
        all_y.append(y)
        if N is not None:
            all_N.append(N)
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    N = np.concatenate(all_N, axis=0) if len(all_N) == len(paths) else None
    return X, y, N

def filter_and_relabel(X, y, N, class_labels):
    class_labels = list(class_labels)
    keep = np.isin(y, class_labels)
    Xf = X[keep]
    yf = y[keep]
    Nf = N[keep] if N is not None else None
    label_map = {orig: i for i, orig in enumerate(class_labels)}
    ynew = np.array([label_map[int(t)] for t in yf], dtype=np.int64)
    print(f"[filter] kept {len(Xf)} objects across {len(class_labels)} classes; map={label_map}")
    return Xf, ynew, Nf, label_map

def farthest_point_sampling(points, k, candidate_idx=None, rng=None):
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
    if rng is None:
        rng = np.random.default_rng()
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

def split_train_val_by_objects(num_obj, train_needed, val_needed, rng):
    order = rng.permutation(num_obj)
    a = min(num_obj, train_needed)
    b = min(max(num_obj - a, 0), val_needed)
    train_idx = order[:a]
    val_idx = order[a:a + b]
    return train_idx, val_idx

def resolve_h5_dir(script_dir: Path):
    cand = script_dir / H5_FOLDER_NAME
    if cand.is_dir():
        return cand
    return script_dir

def run(points_per_sample: int):
    script_dir = Path(__file__).resolve().parent
    h5_dir = resolve_h5_dir(script_dir)

    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    tr_files, te_files = discover_modelnet40_files(h5_dir)
    Xtr_all, ytr_all, _ = load_pool(tr_files)
    Xte_all, yte_all, _ = load_pool(te_files)

    Xtr, ytr, _, label_map = filter_and_relabel(Xtr_all, ytr_all, None, CLASS_LABELS)
    Xte, yte, _, _ = filter_and_relabel(Xte_all, yte_all, None, CLASS_LABELS)
    K = len(CLASS_LABELS)

    P_tr = Xtr.shape[1]
    P_te = Xte.shape[1]
    if P_tr != P_te:
        raise ValueError("Train/Test points-per-object mismatch")
    if points_per_sample > P_tr:
        raise ValueError(f"num_points must be <= {P_tr}, got {points_per_sample}")

    cls_indices_tr = {c: np.where(ytr == c)[0] for c in np.unique(ytr)}
    cls_indices_te = {c: np.where(yte == c)[0] for c in np.unique(yte)}

    trX, trY, vaX, vaY, teX, teY = [], [], [], [], [], []
    for c in range(K):
        obj_idx_tr = cls_indices_tr.get(c, np.array([], dtype=np.int64))
        obj_idx_te = cls_indices_te.get(c, np.array([], dtype=np.int64))

        tr_objs_rel, va_objs_rel = split_train_val_by_objects(len(obj_idx_tr), TRAIN_PER_CLASS, VAL_PER_CLASS, rng)
        tr_objs = obj_idx_tr[tr_objs_rel]
        va_objs = obj_idx_tr[va_objs_rel]

        obj_idx_te = obj_idx_te.copy()
        rng.shuffle(obj_idx_te)
        te_take = min(len(obj_idx_te), TEST_PER_CLASS)
        te_objs = obj_idx_te[:te_take]

        def sample_from_objs(Xpool, obj_rows, target, bucket_X, bucket_Y, label_id):
            if len(obj_rows) == 0:
                if len(cls_indices_tr.get(label_id, [])) == 0:
                    return
                obj_rows = cls_indices_tr[label_id][:1]
            for i in range(target):
                obj = obj_rows[i % len(obj_rows)]
                pts = Xpool[obj]
                sel = sample_points(pts, points_per_sample, method=SAMPLING, rng=rng)
                bucket_X.append(sel[None, ...])
                bucket_Y.append(label_id)

        sample_from_objs(Xtr, tr_objs, TRAIN_PER_CLASS, trX, trY, c)
        sample_from_objs(Xtr, va_objs, VAL_PER_CLASS, vaX, vaY, c)
        sample_from_objs(Xte, te_objs, TEST_PER_CLASS, teX, teY, c)

    trX = np.concatenate(trX, axis=0).astype(np.float32)
    vaX = np.concatenate(vaX, axis=0).astype(np.float32)
    teX = np.concatenate(teX, axis=0).astype(np.float32)
    trY = np.asarray(trY, dtype=np.int64)
    vaY = np.asarray(vaY, dtype=np.int64)
    teY = np.asarray(teY, dtype=np.int64)

    if ROTATION == 1:
        R = special_ortho_group.rvs(3, random_state=int(rng.integers(0, 1 << 31)))
        trX = trX @ R.T
        vaX = vaX @ R.T
        teX = teX @ R.T

    def shuffle_xy(X, Y, rng):
        perm = rng.permutation(len(Y))
        return X[perm], Y[perm]

    trX, trY = shuffle_xy(trX, trY, rng)
    vaX, vaY = shuffle_xy(vaX, vaY, rng)
    teX, teY = shuffle_xy(teX, teY, rng)

    fname = f"modelnet40_{K}classes_{points_per_sample}_{NUM_REUPLOAD}_{SAMPLING}_train{TRAIN_PER_CLASS}_val{VAL_PER_CLASS}_test{TEST_PER_CLASS}_new.npz"
    out_npz = script_dir / fname

    meta = {
        "source": "ModelNet40-HDF5",
        "class_labels_original": list(CLASS_LABELS),
        "label_map_to_compact": {int(k): int(v) for k, v in label_map.items()},
        "num_classes": int(K),
        "points_per_sample": int(points_per_sample),
        "num_reupload": int(NUM_REUPLOAD),
        "train_per_class": int(TRAIN_PER_CLASS),
        "val_per_class": int(VAL_PER_CLASS),
        "test_per_class": int(TEST_PER_CLASS),
        "sampling": SAMPLING,
        "rotation": int(ROTATION),
        "seed": int(SEED),
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

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("num_points", type=int)
    args = ap.parse_args()
    run(args.num_points)
