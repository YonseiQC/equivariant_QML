import os, sys, json, argparse, glob, re
from pathlib import Path
import numpy as np
import trimesh
from tqdm import tqdm
from scipy.stats import special_ortho_group

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
SYN2IDX = {s:i for i,(s,_) in enumerate(SHAPENET55)}
SYN2NAME = dict(SHAPENET55)

CLASS_SYNSETS = ["02843684", "02876657", "02880940", "02924116", "02954340"]
TRAIN_PER_CLASS = 700
VAL_PER_CLASS = 100
TEST_PER_CLASS = 200
OVERSAMPLE_P = 2048
ROTATION = 1
SEED = 1557
KEEP_GLOBAL_INDEX = False
SHUFFLE_LABELS = True

def load_mesh_any(obj_path: str) -> trimesh.Trimesh:
    m = trimesh.load(obj_path, force="mesh", process=True)
    if isinstance(m, trimesh.Trimesh):
        return m
    if hasattr(m, "geometry") and len(m.geometry) > 0:
        parts = [g for g in m.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if len(parts) == 1:
            return parts[0]
        if len(parts) > 1:
            return trimesh.util.concatenate(parts)
    raise ValueError(f"Cannot convert to Trimesh: {obj_path}")

def surface_sample(mesh: trimesh.Trimesh, n: int):
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    return pts

def normalize_unit_sphere(points: np.ndarray):
    c = points.mean(axis=0, keepdims=True)
    centered = points - c
    scale = np.linalg.norm(centered, axis=1).max() + 1e-12
    return (centered / scale).astype(np.float32)

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

def always_permute_points(X, rng):
    N, k, _ = X.shape
    out = np.empty_like(X)
    for i in range(N):
        out[i] = X[i, rng.permutation(k)]
    return out

def _split_objects_disjoint(objs, quotas, rng):
    n = len(objs)
    if n == 0:
        return [], [], []
    q = np.array(quotas, dtype=np.float64)
    frac = q / q.sum()
    counts = np.floor(frac * n).astype(int)
    need = int(n - counts.sum())
    order = np.argsort(-frac)
    for i in range(need):
        counts[order[i % 3]] += 1
    rng.shuffle(objs)
    a, b = counts[0], counts[1]
    return objs[:a], objs[a:a+b], objs[a+b:]

def _build_pool_for_objs(objs_subset, oversample_P, rng):
    cache, used_map = {}, {}
    for path in objs_subset:
        mesh = load_mesh_any(path)
        pts = surface_sample(mesh, oversample_P)
        pool = normalize_unit_sphere(pts)
        cache[path] = pool
        used_map[path] = np.zeros(pool.shape[0], dtype=bool)
    return cache, used_map

def _fill_quota_from_pool(num_points, target, cache, used_map, label_value, rng, bucket_X, bucket_Y):
    if target <= 0 or len(cache) == 0:
        return 0
    keys = list(cache.keys())
    ptr, cnt, stagnation = 0, 0, 0
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
    return cnt

def build_dataset(root_dir, class_synsets, num_points):
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    valid = [s for s in class_synsets if s in SYN2IDX]
    valid = sorted(valid, key=lambda s: SYN2IDX[s])
    if not valid:
        raise ValueError("No valid synsets in CLASS_SYNSETS.")

    label_map = {s: (SYN2IDX[s] if KEEP_GLOBAL_INDEX else i) for i, s in enumerate(valid)}

    per_syn_objs = {}
    for s in valid:
        syn_dir = os.path.join(str(root_dir), s)
        if not os.path.isdir(syn_dir):
            per_syn_objs[s] = []
            continue
        obj_paths = glob.glob(os.path.join(syn_dir, "*", "models", "model_normalized.obj"))
        if not obj_paths:
            obj_paths = glob.glob(os.path.join(syn_dir, "**", "*.obj"), recursive=True)
        per_syn_objs[s] = sorted(obj_paths)

    trX, trY, vaX, vaY, teX, teY = [], [], [], [], [], []

    for s in tqdm(valid, desc="classes", leave=False):
        objs = list(per_syn_objs.get(s, []))
        if len(objs) == 0:
            continue

        train_objs, val_objs, test_objs = _split_objects_disjoint(
            objs, (TRAIN_PER_CLASS, VAL_PER_CLASS, TEST_PER_CLASS), rng
        )

        tr_cache, tr_used = _build_pool_for_objs(sorted(train_objs), OVERSAMPLE_P, rng)
        va_cache, va_used = _build_pool_for_objs(sorted(val_objs),   OVERSAMPLE_P, rng)
        te_cache, te_used = _build_pool_for_objs(sorted(test_objs),  OVERSAMPLE_P, rng)

        _fill_quota_from_pool(num_points, TRAIN_PER_CLASS, tr_cache, tr_used, label_map[s], rng, trX, trY)
        _fill_quota_from_pool(num_points, VAL_PER_CLASS,   va_cache, va_used, label_map[s], rng, vaX, vaY)
        _fill_quota_from_pool(num_points, TEST_PER_CLASS,  te_cache, te_used, label_map[s], rng, teX, teY)

    if len(trX) == 0 or len(vaX) == 0 or len(teX) == 0:
        raise RuntimeError("No samples produced. Check OBJ paths under script folder.")

    trX = np.concatenate(trX, axis=0).astype(np.float32)
    vaX = np.concatenate(vaX, axis=0).astype(np.float32)
    teX = np.concatenate(teX, axis=0).astype(np.float32)
    trY = np.asarray(trY, dtype=np.int64)
    vaY = np.asarray(vaY, dtype=np.int64)
    teY = np.asarray(teY, dtype=np.int64)

    trX = always_permute_points(trX, rng)
    vaX = always_permute_points(vaX, rng)
    teX = always_permute_points(teX, rng)

    if SHUFFLE_LABELS:
        idx = rng.permutation(len(trX)); trX, trY = trX[idx], trY[idx]
        idx = rng.permutation(len(vaX)); vaX, vaY = vaX[idx], vaY[idx]
        idx = rng.permutation(len(teX)); teX, teY = teX[idx], teY[idx]

    if ROTATION == 1:
        R = special_ortho_group.rvs(3, random_state=int(rng.integers(0, 1 << 31)))
        trX = trX @ R.T
        vaX = vaX @ R.T
        teX = teX @ R.T

    keep_global = [SYN2IDX[s] for s in valid]
    label_map_out = {s: int(label_map[s]) for s in valid}
    return trX, trY, vaX, vaY, teX, teY, valid, keep_global, label_map_out

def run(num_points: int):
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir

    syn_pat = re.compile(r"^\d{8}$")
    sel = [s for s in CLASS_SYNSETS if syn_pat.match(s)]

    trX, trY, vaX, vaY, teX, teY, syns, keep_global, label_map_out = build_dataset(
        root_dir=root_dir,
        class_synsets=sel,
        num_points=num_points
    )

    num_classes = len(syns)
    fname = f"shapenet_{num_classes}classes_{num_points}_1_fps_train{TRAIN_PER_CLASS}_val{VAL_PER_CLASS}_test{TEST_PER_CLASS}_new.npz"
    out_npz = script_dir / fname

    meta = {
        "version": "shapenet_quota_fps_fixed",
        "class_synsets": syns,
        "class_names": [SYN2NAME[s] for s in syns],
        "keep_global_index": bool(KEEP_GLOBAL_INDEX),
        "global_indices": keep_global,
        "label_map": label_map_out,
        "num_classes": int(num_classes),
        "num_points": int(num_points),
        "train_per_class": int(TRAIN_PER_CLASS),
        "val_per_class": int(VAL_PER_CLASS),
        "test_per_class": int(TEST_PER_CLASS),
        "oversample_P": int(OVERSAMPLE_P),
        "rotation": int(ROTATION),
        "seed": int(SEED),
        "shuffled": bool(SHUFFLE_LABELS),
        "disjoint_objects": True
    }

    np.savez_compressed(
        out_npz,
        train_dataset_x=trX, train_dataset_y=trY,
        val_dataset_x=vaX,   val_dataset_y=vaY,
        test_dataset_x=teX,  test_dataset_y=teY,
        labels_used=np.array(keep_global if KEEP_GLOBAL_INDEX else list(range(num_classes)), dtype=np.int64),
        class_synsets=np.array(syns),
        num_classes=num_classes,
        meta=json.dumps(meta, ensure_ascii=False)
    )

    print(f"✅ Saved: {out_npz}")
    print("Train:", trX.shape, "| Val:", vaX.shape, "| Test:", teX.shape)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("num_points", type=int)
    args = ap.parse_args()
    run(args.num_points)
