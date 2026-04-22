import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='splits/atlas.csv')
parser.add_argument('--csv_dir', type=str)
parser.add_argument('--sim_dir', type=str, default='/data/cb/scratch/datasets/atlas')
parser.add_argument('--outdir', type=str, default='./data_atlas')
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--atlas', action='store_true')
parser.add_argument('--stride', type=int, default=1)
args = parser.parse_args()

import mdtraj, os, tqdm
import pandas as pd 
from multiprocessing import Pool
import numpy as np
from mdgen import residue_constants as rc
#
os.makedirs(args.outdir, exist_ok=True)
df = pd.read_csv(args.split)
proteins = df.proteins
fragments = df.fragments
#
def main():
    jobs = [[p, f] for p, f in zip(proteins, fragments)]

    if args.num_workers > 1:
        p = Pool(args.num_workers)
        p.__enter__()
        __map__ = p.imap
    else:
        __map__ = map
    for _ in tqdm.tqdm(__map__(do_job, jobs), total=len(jobs)):
        pass
    if args.num_workers > 1:
        p.__exit__(None, None, None)

def traj_to_atom14(traj):
    arr = np.zeros((traj.n_frames, traj.n_residues, 14, 3), dtype=np.float32)
    for i, resi in enumerate(traj.top.residues):
        for at in resi.atoms:
            if at.name not in rc.restype_name_to_atom14_names[resi.name]:
                print(resi.name, at.name, 'not found'); continue
            j = rc.restype_name_to_atom14_names[resi.name].index(at.name)
            arr[:,i,j] = traj.xyz[:,at.index] * 10.0 
    return arr

def export_sequence_to_csv(pdb_path, protein_id, fragment_id, chain, out_csv):
    import os
    import pandas as pd
    import mdtraj

    """
    讀取單一 chain 的 PDB，轉換為 1 字母序列，並輸出 / 追加到 CSV。

    Args:
        pdb_path: 輸入的單一 chain PDB 檔案路徑
        protein_id: 蛋白質名稱
        fragment_id: 片段編號
        chain: 鏈名稱 (如 'A')
        out_csv: 輸出的 CSV 檔名
    """
    top = mdtraj.load_topology(pdb_path)
    sequence = ""
    for res in top.residues:
        one_letter = rc.restype_3to1.get(res.name, "X")
        sequence += one_letter
    row = {
        "protein": protein_id,
        "fragment": fragment_id,
        "chain": chain,
        "seqres": sequence
    }
    df = pd.DataFrame([row])
    if not os.path.exists(out_csv):
        df.to_csv(out_csv, index=False)
        print(f"✅ 建立新 CSV 並寫入第一筆資料: {out_csv}")
    else:
        df.to_csv(out_csv, mode="a", header=False, index=False)
        print(f"✅ 已追加一筆資料到 CSV: {out_csv}")

if args.atlas:
    def do_job(name):
        try:
            protein_id,fragment_id=name[0],name[1]
            for i in ['A','B','C']:
                pdb_path = f'{args.sim_dir}/{protein_id}-{fragment_id}-{i}.pdb'
                traj = mdtraj.load(f'{args.sim_dir}/{protein_id}-{fragment_id}-{i}.xtc', top=pdb_path) 
                traj.atom_slice([a.index for a in traj.top.atoms if a.element.symbol != 'H'], True)
                traj.superpose(traj)
                arr = traj_to_atom14(traj)
                np.save(f'{args.outdir}/{protein_id}-{fragment_id}-{i}.npy', arr[::args.stride])
                #
                if args.csv_dir is not None:
                    reseq_csv = f'{args.csv_dir}/{protein_id}-{fragment_id}.csv'
                    export_sequence_to_csv(pdb_path, protein_id, fragment_id, i, reseq_csv)
        except Exception as e:
            print(f"❌ Error processing {name}: {e}")
            return

if __name__ == "__main__":
    main()