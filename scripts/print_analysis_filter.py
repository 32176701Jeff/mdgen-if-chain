import numpy as np
import tqdm
import sys
import pickle
import warnings
import pandas as pd
import scipy.stats
import argparse


def correlations(a, b, prefix=''):
    return {
        prefix + 'pearson': scipy.stats.pearsonr(a, b)[0],
        prefix + 'spearman': scipy.stats.spearmanr(a, b)[0],
        prefix + 'kendall': scipy.stats.kendalltau(a, b)[0],
    }


def load_selected_names(csv_path):
    df = pd.read_csv(csv_path)

    if "name" not in df.columns:
        raise ValueError(f"{csv_path} 裡面找不到 'name' 欄位")

    return set(df["name"].astype(str).str.strip())


def analyze_data(data, selected_names=None):
    df = []

    for name, out in data.items():
        if selected_names is not None and name not in selected_names:
            continue

        item = {
            'name': name,
            'md_pairwise': out['ref_mean_pairwise_rmsd'],
            'af_pairwise': out['af_mean_pairwise_rmsd'],
            'cosine_sim': abs(out['cosine_sim']),
            'emd_mean': np.square(out['emd_mean']).mean() ** 0.5,
            'emd_var': np.square(out['emd_var']).mean() ** 0.5,
        } | correlations(out['af_rmsf'], out['ref_rmsf'], prefix='rmsf_')

        if 'EMD,ref' not in out:
            out['EMD,ref'] = out['EMD-2,ref']
            out['EMD,af2'] = out['EMD-2,af2']
            out['EMD,joint'] = out['EMD-2,joint']

        for emd_dict, emd_key in [
            (out['EMD,ref'], 'ref'),
            (out['EMD,joint'], 'joint')
        ]:
            item.update({
                emd_key + 'emd': emd_dict['ref|af'],
                emd_key + 'emd_tr': emd_dict['ref mean|af mean'],
                emd_key + 'emd_int': (
                    emd_dict['ref|af']**2 - emd_dict['ref mean|af mean']**2
                )**0.5,
            })

        try:
            crystal_contact_mask = out['crystal_distmat'] < 0.8
            ref_transient_mask = (~crystal_contact_mask) & (out['ref_contact_prob'] > 0.1)
            af_transient_mask = (~crystal_contact_mask) & (out['af_contact_prob'] > 0.1)
            ref_weak_mask = crystal_contact_mask & (out['ref_contact_prob'] < 0.9)
            af_weak_mask = crystal_contact_mask & (out['af_contact_prob'] < 0.9)

            item.update({
                'weak_contacts_iou': (ref_weak_mask & af_weak_mask).sum() / (ref_weak_mask | af_weak_mask).sum(),
                'transient_contacts_iou': (ref_transient_mask & af_transient_mask).sum() / (ref_transient_mask | af_transient_mask).sum()
            })
        except Exception:
            item.update({
                'weak_contacts_iou': np.nan,
                'transient_contacts_iou': np.nan,
            })

        sasa_thresh = 0.02
        buried_mask = out['crystal_sasa'][0] < sasa_thresh
        ref_sa_mask = (out['ref_sa_prob'] > 0.1) & buried_mask
        af_sa_mask = (out['af_sa_prob'] > 0.1) & buried_mask

        item.update({
            'num_sasa': ref_sa_mask.sum(),
            'sasa_iou': (ref_sa_mask & af_sa_mask).sum() / (ref_sa_mask | af_sa_mask).sum(),
        })

        item.update(
            correlations(
                out['ref_mi_mat'].flatten(),
                out['af_mi_mat'].flatten(),
                prefix='exposon_mi_'
            )
        )

        df.append(item)

    if len(df) == 0:
        empty_df = pd.DataFrame(columns=['name']).set_index('name')
        return np.array([]), np.array([]), empty_df, data

    df = pd.DataFrame(df).set_index('name')

    all_ref_rmsf = np.concatenate([data[name]['ref_rmsf'] for name in df.index])
    all_af_rmsf = np.concatenate([data[name]['af_rmsf'] for name in df.index])

    return all_ref_rmsf, all_af_rmsf, df, data


def safe_pearson(a, b):
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return scipy.stats.pearsonr(a, b)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", nargs="+", help="input pickle paths")
    parser.add_argument("--csv_filter", type=str, default=None, help="只分析 csv 裡 name 欄位提到的 protein")
    args = parser.parse_args()

    selected_names = None
    if args.csv_filter is not None:
        selected_names = load_selected_names(args.csv_filter)
        print(f"[INFO] loaded {len(selected_names)} names from {args.csv_filter}")

    datas = {}

    for path in tqdm.tqdm(args.paths):
        with open(path, 'rb') as f:
            data = pickle.load(f)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            datas[path] = analyze_data(data, selected_names=selected_names)

    new_df = []
    for key in datas:
        ref_rmsf, af_rmsf, df, data = datas[key]

        if len(df) == 0:
            new_df.append({
                'path': key,
                'count': 0,
                'MD pairwise RMSD': np.nan,
                'Pairwise RMSD': np.nan,
                'Pairwise RMSD r': np.nan,
                'MD RMSF': np.nan,
                'RMSF': np.nan,
                'Global RMSF r': np.nan,
                'Per target RMSF r': np.nan,
                'RMWD': np.nan,
                'RMWD trans': np.nan,
                'RMWD var': np.nan,
                'MD PCA W2': np.nan,
                'Joint PCA W2': np.nan,
                'PC sim > 0.5 %': np.nan,
                'Weak contacts J': np.nan,
                'Weak contacts nans': np.nan,
                'Transient contacts J': np.nan,
                'Transient contacts nans': np.nan,
                'Exposed residue J': np.nan,
                'Exposed MI matrix rho': np.nan,
            })
            continue

        new_df.append({
            'path': key,
            'count': len(df),
            'MD pairwise RMSD': df.md_pairwise.median(),
            'Pairwise RMSD': df.af_pairwise.median(),
            'Pairwise RMSD r': safe_pearson(df.md_pairwise, df.af_pairwise),
            'MD RMSF': np.median(ref_rmsf),
            'RMSF': np.median(af_rmsf),
            'Global RMSF r': safe_pearson(ref_rmsf, af_rmsf),
            'Per target RMSF r': df.rmsf_pearson.median(),
            'RMWD': np.sqrt(df.emd_mean**2 + df.emd_var**2).median(),
            'RMWD trans': df.emd_mean.median(),
            'RMWD var': df.emd_var.median(),
            'MD PCA W2': df.refemd.median(),
            'Joint PCA W2': df.jointemd.median(),
            'PC sim > 0.5 %': (df.cosine_sim > 0.5).mean() * 100,
            'Weak contacts J': df.weak_contacts_iou.median(),
            'Weak contacts nans': df.weak_contacts_iou.isna().mean(),
            'Transient contacts J': df.transient_contacts_iou.median(),
            'Transient contacts nans': df.transient_contacts_iou.isna().mean(),
            'Exposed residue J': df.sasa_iou.median(),
            'Exposed MI matrix rho': df.exposon_mi_spearman.median(),
        })

    new_df = pd.DataFrame(new_df).set_index('path')
    print(new_df.round(2).T)


if __name__ == "__main__":
    main()