import os

# must be done BEFORE any numpy/torch/julia import to avoid crashes
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["JULIA_NUM_THREADS"] = "1"

import re
import json
import pickle
import hashlib
import datetime
import platform
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import yaml
from sklearn.model_selection import KFold

from juliacall import Main as jl


# ══════════════════════════════════════════════════════════════════════
# CONFIG 
# ══════════════════════════════════════════════════════════════════════

MI_DIR = r"D:\\Data\\FII_BCI_Corpus\\MI"                  # folder containing the .npz files per database
OUTPUT_ROOT = r"D:\\Data\\FII_BCI_Corpus\\MI_Preprocessed"   # output root
PRECOND_JL_PATH = r"utils\\precond.jl"

CLASSES = ["left_hand", "right_hand", "feet", "tongue"] # 
MIN_TRIALS_PER_CLASS = 20     # inclusion: minimum(tpc values) > this

BAND_PASS = (8, 32)
UPPER_LIMIT = 1.2

PRECOND = True
PRECOND_TIKHONOV = 1e-8
PRECOND_EXPL_VAR = 0.9999     

TRAIN_PCT = 0.7
TEST_PCT = 0.15               # val_pct = 1 - train_pct - test_pct
K_FOLDS = 5
SEED = 0

# ══════════════════════════════════════════════════════════════════════
# Script
# ══════════════════════════════════════════════════════════════════════

def make_data_signature(**kwargs):
    payload = json.dumps(kwargs, sort_keys=True, default=str)
    return hashlib.md5(payload.encode()).hexdigest()


FNAME_PATTERN = re.compile(r"subject_(\d+)_session_(\d+)\.npz$")


def setup_julia():
    print("Installing/loading Julia packages (NPZ, Eegle)...")
    jl.seval('import Pkg; Pkg.add("NPZ"); Pkg.add("Eegle")')
    jl.seval('using NPZ')
    jl.seval('using Eegle')
    jl.include(PRECOND_JL_PATH)  
    print("Julia setup done.")


def select_databases():

    jl.MIDir = MI_DIR
    jl.classes_py = CLASSES
    jl.min_tpc = MIN_TRIALS_PER_CLASS

    DBs = jl.seval(
        """
        classes = Vector{String}(classes_py)
        inclusion = (("tpc", x -> minimum(values(x)) > min_tpc),)
        selectDB(MIDir, :MI; classes, inclusion)
        """
    )
    return DBs


def extract_subject_session_covs(DB):

    db_name = str(DB.dbName)
    print(f"\n--- Extracting covariances for database {db_name} ---")

    per_subject = defaultdict(list)

    n_files = len(DB.files)
    for d, file in enumerate(DB.files, start=1):
        file_str = str(file)
        print(f"  [{d}/{n_files}] {file_str}")

        m = FNAME_PATTERN.search(file_str)
        if m is None:
            print("    !! filename doesn't match subject_XX_session_XX.npz pattern, skipping")
            continue
        subject_id = int(m.group(1))
        session_id = int(m.group(2))

        jl.file = file
        jl.classes_py = CLASSES
        jl.bp_low, jl.bp_high = BAND_PASS
        jl.upper_limit = UPPER_LIMIT

        covs_tensor = jl.seval(
            """
            classes = Vector{String}(classes_py)
            o = readNY(file; bandPass=(bp_low, bp_high), upperLimit=upper_limit, classes)
            C = encode(o; covtype=SCM)
            global _labels = o.y
            cat(C...; dims=3)
            """
        )

        covs_tensor = np.array(covs_tensor)
        covs = np.transpose(covs_tensor, (2, 0, 1))  # (n_trials, n_chans, n_chans)

        labels = np.array(jl.seval("_labels"))
        labels = np.array([int(el) - 1 for el in labels]) 

        per_subject[subject_id].append((session_id, covs, labels))

    return db_name, per_subject


def run_precond(C_train, C_test, C_val, tikhonov, expl_var):
    n_chans = C_train.shape[-1]
    pipeline = jl.Precond.make_pipeline(tikhonov, expl_var, n_chans)
    C_train_p, C_test_p, C_val_p = jl.Precond.pre_cond(C_train, C_test, C_val, pipeline)
    return np.array(C_train_p), np.array(C_test_p), np.array(C_val_p)


def process_database(db_name, per_subject, out_root):

    signature = make_data_signature(
        db_name=db_name,
        classes=CLASSES,
        min_trials_per_class=MIN_TRIALS_PER_CLASS,
        band_pass=BAND_PASS,
        upper_limit=UPPER_LIMIT,
        precond=PRECOND,
        precond_tikhonov=PRECOND_TIKHONOV,
        precond_explVar=PRECOND_EXPL_VAR,
        train_pct=TRAIN_PCT,
        test_pct=TEST_PCT,
        k_folds=K_FOLDS,
        seed=SEED,
    )

    out_dir = Path(out_root) / db_name / signature
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    fold_stats = defaultdict(lambda: {"train": [], "val": [], "test": []})

    for subject_id, sessions in sorted(per_subject.items()):
        # concatenate sessions
        covs_list, labels_list, dom_list = [], [], []
        for session_id, covs, labels in sorted(sessions):
            covs_list.append(covs)
            labels_list.append(labels)
            dom_list.append(np.full(len(labels), session_id))

        covs_subj = np.concatenate(covs_list, axis=0)
        labels_subj = np.concatenate(labels_list, axis=0)
        dom_subj = np.concatenate(dom_list, axis=0)

        n = len(labels_subj)
        kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

        for fold_id, (trainval_idx, test_idx) in enumerate(kf.split(np.arange(n))):
            n_train = int(TRAIN_PCT * len(trainval_idx))
            perm = rng.permutation(len(trainval_idx))
            train_idx = trainval_idx[perm[:n_train]]
            val_idx = trainval_idx[perm[n_train:]]

            fold_stats[fold_id]["train"].append(train_idx)
            fold_stats[fold_id]["val"].append(val_idx)
            fold_stats[fold_id]["test"].append(test_idx)

            C_train = covs_subj[train_idx]
            C_val = covs_subj[val_idx]
            C_test = covs_subj[test_idx]

            if PRECOND:
                print(f"  [subj {subject_id} | fold {fold_id}] preconditioning "
                      f"(train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})")
                C_train, C_test, C_val = run_precond(
                    C_train, C_test, C_val, PRECOND_TIKHONOV, PRECOND_EXPL_VAR
                )

            fold_data = {
                "train": (C_train, labels_subj[train_idx], dom_subj[train_idx]),
                "val":   (C_val,   labels_subj[val_idx],   dom_subj[val_idx]),
                "test":  (C_test,  labels_subj[test_idx],  dom_subj[test_idx]),
            }

            pkl_path = out_dir / f"fold_{fold_id}_subj{subject_id}.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(fold_data, f)
            print(f"    Saved {pkl_path}")

    write_metadata(out_dir / "metadata.yaml", db_name, signature, fold_stats)
    return out_dir


def _yaml_safe(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_yaml_safe(v) for v in obj]
    return obj


def _compute_fold_statistics(fold_stats):
    stats = {}
    for fold_id, splits in fold_stats.items():
        stats[f"fold_{fold_id}"] = {}
        for split, indices_list in splits.items():
            all_indices = np.concatenate(indices_list) if indices_list else []
            stats[f"fold_{fold_id}"][split] = {"n_samples": int(len(all_indices))}
    return stats


def write_metadata(path, db_name, signature, fold_stats):
    metadata = {
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "python_version": sys.version,
        "platform": platform.platform(),
        "data_specification": _yaml_safe(dict(
            db_name=db_name,
            mi_dir=MI_DIR,
            classes=CLASSES,
            min_trials_per_class=MIN_TRIALS_PER_CLASS,
            band_pass=BAND_PASS,
            upper_limit=UPPER_LIMIT,
            precond=PRECOND,
            precond_tikhonov=PRECOND_TIKHONOV,
            precond_explVar=PRECOND_EXPL_VAR,
            train_pct=TRAIN_PCT,
            test_pct=TEST_PCT,
            k_folds=K_FOLDS,
            seed=SEED,
        )),
        "experiment_specification": _yaml_safe(dict(
            signature=signature,
            output_root=OUTPUT_ROOT,
        )),
        "fold_statistics": _compute_fold_statistics(fold_stats),
        "extra": {
            "julia_pipeline": (
                f"Tikhonov({PRECOND_TIKHONOV}) → Recenter(eVar={PRECOND_EXPL_VAR}) → Equalize"
                if PRECOND else None
            ),
            "grouping": "per_subject (sessions combined before K-Fold)",
        },
    }
    with open(path, "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)
    print(f"\nSaved metadata: {path}")


def main():
    setup_julia()
    DBs = select_databases()

    for DB in DBs:
        db_name, per_subject = extract_subject_session_covs(DB)
        print(f"\nSubjects found for {db_name}: {sorted(per_subject.keys())}")
        out_dir = process_database(db_name, per_subject, OUTPUT_ROOT)
        print(f"\n✓ Done with {db_name} -> {out_dir}")


if __name__ == "__main__":
    main()