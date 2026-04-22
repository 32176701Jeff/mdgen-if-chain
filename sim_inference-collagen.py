import argparse
import torch
import argparse

torch.serialization.add_safe_globals([argparse.Namespace])
parser = argparse.ArgumentParser()
parser.add_argument('--sim_ckpt', type=str, default=None, required=True)
parser.add_argument('--data_dir', type=str, default=None, required=True)
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--pdb_id', nargs='*', default=[])
parser.add_argument('--num_frames', type=int, default=1000)
parser.add_argument('--num_rollouts', type=int, default=100)
parser.add_argument('--no_frames', action='store_true')
parser.add_argument('--tps', action='store_true')
parser.add_argument('--xtc', action='store_true')
parser.add_argument('--out_dir', type=str, default=".")
parser.add_argument('--split', type=str, default='splits/4AA_test.csv')
parser.add_argument('--csv_dir', type=str, default='splits/4AA_test.csv')
args = parser.parse_args()

import os, torch, mdtraj, tqdm, time
import numpy as np
from mdgen.geometry import atom14_to_frames, atom14_to_atom37, atom37_to_torsions
from mdgen.residue_constants import restype_order, restype_atom37_mask
from mdgen.tensor_utils import tensor_tree_map
from mdgen.wrapper import NewMDGenWrapper
from mdgen.utils import atom14_to_pdb
import pandas as pd




os.makedirs(args.out_dir, exist_ok=True)



def get_batch(protein, fragment, seqres, letter, num_frames):
    # 1. 讀取與 Frame 截取
    arr = np.lib.format.open_memmap(f'{args.data_dir}/{protein}-{fragment}-{letter}.npy', 'r')
    
    if not args.tps:
        arr = np.copy(arr[0:1]).astype(np.float32)
    else:
        arr = np.copy(arr[0:num_frames]).astype(np.float32)
    
    # 2. 處理序列與未知氨基酸
    # residue_constants 定義 unk_restype_index 為 20
    seqres_ids = torch.tensor([restype_order[c] if c in restype_order else 20 for c in seqres], dtype=torch.long)
    
    # 3. 基礎轉換
    frames = atom14_to_frames(torch.from_numpy(arr))
    L = len(seqres_ids)
    mask = torch.ones(L, dtype=torch.float32)
    cid = {'A': 0, 'B': 1, 'C': 2}[letter]
    chain_ids = torch.full((L,), cid, dtype=torch.long)
    
    # 4. 計算 Torsions (必須傳入 batch 維度的 seqres)
    atom37 = torch.from_numpy(atom14_to_atom37(arr, seqres_ids[None])).float()
    torsions, torsion_mask = atom37_to_torsions(atom37, seqres_ids[None])
    torsion_mask = torsion_mask[0]
    
    trans = frames._trans
    rots = frames._rots._rot_mats

    # @@@@@@@@@@@ 仿照 Train 時的 C 鏈 Padding 邏輯 @@@@@@@@@@@
    if letter == 'C':
        pad = 1
        T = trans.shape[0]
        
        # 1. Padding Trans & Rots (維持 256 殘基維度)
        trans = torch.cat([trans, torch.zeros((T, pad, 3), dtype=torch.float32)], dim=1)
        identity_rots = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).expand(T, pad, -1, -1)
        rots = torch.cat([rots, identity_rots], dim=1)
        
        # 2. Padding Mask & Sequence
        mask = torch.cat([mask, torch.zeros(pad, dtype=torch.float32)])
        # 訓練時 seqres Padding 是 0 (ALA)
        seqres_ids = torch.cat([seqres_ids, torch.zeros(pad, dtype=torch.long)])
        
        # 3. Padding Torsions & Torsion Mask
        torsions = torch.cat([torsions, torch.zeros((T, pad, 7, 2), dtype=torch.float32)], dim=1)
        torsion_mask = torch.cat([torsion_mask, torch.zeros((pad, 7), dtype=torch.float32)])
        
        # 4. Padding Chain ID (設為 3)
        chain_ids = torch.cat([chain_ids, torch.full((pad,), 3, dtype=torch.long)])
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    if args.no_frames:
        # 修正：從 residue_constants 載入 RESTYPE_ATOM37_MASK
        # 這裡需要轉成 torch tensor 才能索引
        full_atom37_mask = torch.from_numpy(RESTYPE_ATOM37_MASK)
        return {
            'atom37': atom37,
            'seqres': seqres_ids,
            'mask': full_atom37_mask[seqres_ids],
        }
        
    return {
        'torsions': torsions,
        'torsion_mask': torsion_mask,
        'trans': trans,
        'rots': rots,
        'seqres': seqres_ids,
        'mask': mask,
        'chain_ids': chain_ids,
    }



def rollout(model, batch):

    #print('Start sim', batch['trans'][0,0,0])
    if args.no_frames:
        expanded_batch = {
            'atom37': batch['atom37'].expand(-1, args.num_frames, -1, -1, -1),
            'seqres': batch['seqres'],
            'mask': batch['mask'],
            'chain_ids': batch['chain_ids'], #@@@@@@@@ postprocess
        }
    else:    
        expanded_batch = {
            'torsions': batch['torsions'].expand(-1, args.num_frames, -1, -1, -1),
            'torsion_mask': batch['torsion_mask'],
            'trans': batch['trans'].expand(-1, args.num_frames, -1, -1),
            'rots': batch['rots'].expand(-1, args.num_frames, -1, -1, -1),
            'seqres': batch['seqres'],
            'mask': batch['mask'],
            'chain_ids': batch['chain_ids'], #@@@@@@@@ postprocess
        }
    atom14, _ = model.inference(expanded_batch)
    new_batch = {**batch}
    if args.no_frames:
        new_batch['atom37'] = torch.from_numpy(
            atom14_to_atom37(atom14[:,-1].cpu(), batch['seqres'][0].cpu())
        ).cuda()[:,None].float()  
    else:
        frames = atom14_to_frames(atom14[:,-1])
        new_batch['trans'] = frames._trans[None]
        new_batch['rots'] = frames._rots._rot_mats[None]
        atom37 = atom14_to_atom37(atom14[0,-1].cpu(), batch['seqres'][0].cpu())
        torsions, _ = atom37_to_torsions(atom37, batch['seqres'][0].cpu())
        new_batch['torsions'] = torsions[None, None].cuda()
    return atom14, new_batch
    
# @@@@@@@@@@@@@@ postprocess
def do(model, protein, fragment, csv_dir):

    Letter = ['A', 'B', 'C']
    # ---------- read protein csv ----------
    protein_csv = os.path.join(csv_dir, f"{protein}-{fragment}.csv")
    df = pd.read_csv(protein_csv, index_col='chain')
    
    # ---------- build batch for A/B/C ----------
    batches = []
    seqres_list = []
    lengths = []
    for letter in Letter:
        seqres = df.seqres[letter]
        item = get_batch(protein, fragment, seqres, letter,
                         num_frames=model.args.num_frames)
        # DataLoader 會增加一個 Batch 維度，形狀變為 (B=1, ...)
        batch = next(iter(torch.utils.data.DataLoader([item])))
        batch = tensor_tree_map(lambda x: x.cuda(), batch)
        batches.append(batch)
        seqres_list.append(seqres)
        # 這裡記錄各鏈實際長度，包含 C 鏈的 Padding (85, 85, 86)
        lengths.append(batch['seqres'].shape[1])

    # ---------- concat batches (修正維度錯誤) ----------
    batch_abc = {}
    for key in batches[0].keys():
        if key in ['seqres', 'mask', 'torsion_mask', 'chain_ids']:
            batch_abc[key] = torch.cat([b[key] for b in batches], dim=1)
        elif key in ['torsions', 'rots', 'trans']:
            batch_abc[key] = torch.cat([b[key] for b in batches], dim=2)
        else:
            batch_abc[key] = batches[0][key]

    # ---------- rollout (保留原始邏輯) ----------
    all_atom14 = []
    start = time.time()
    batch = batch_abc
    for _ in tqdm.trange(args.num_rollouts):
        atom14, batch = rollout(model, batch)
        all_atom14.append(atom14)
    
    print(f"Rollout Time: {time.time() - start:.2f}s")
    all_atom14 = torch.cat(all_atom14, 1) # 合併所有 time steps

    # ---------- split atom14 back to A/B/C (根據 lengths 拆分) ----------
    LA, LB, LC = lengths
    idxA = slice(0, LA)
    idxB = slice(LA, LA + LB)
    idxC = slice(LA + LB, LA + LB + LC)
    
    atom14_A = all_atom14[:, :, idxA]
    atom14_B = all_atom14[:, :, idxB]
    atom14_C = all_atom14[:, :, idxC]

    # ---------- postprocess (保留原始儲存與 XTC 邏輯) ----------
    for letter, atom14_chain, idx in zip(
            Letter,
            [atom14_A, atom14_B, atom14_C],
            [idxA, idxB, idxC]):

        out_name = f'{protein}-{fragment}-{letter}'
        path = os.path.join(args.out_dir, f'{out_name}.pdb')

        # 這裡會根據拆分出的 idx 取出對應的 atom14 與 seqres
        save_atom14 = atom14_chain[0].cpu().numpy()
        save_seqres = batch_abc['seqres'][0, idx].cpu().numpy()

        # 如果是 C 鏈且包含 Padding (86個)，切回 85 以符合原始數據
        if letter == 'C' and save_atom14.shape[1] == 86:
            save_atom14 = save_atom14[:, :85]
            save_seqres = save_seqres[:85]

        atom14_to_pdb(save_atom14, save_seqres, path)

        if args.xtc:
            traj = mdtraj.load(path)
            traj.superpose(traj)
            traj.save(os.path.join(args.out_dir, f'{out_name}.xtc'))
            traj[0].save(path)
# @@@@@@@@@@@@@@ postprocess

@torch.no_grad()
def main():

    ckpt = torch.load(args.sim_ckpt, map_location='cpu')
    model_args = ckpt['hyper_parameters']['args']
    if not hasattr(model_args, 'unfreeze_start_layer'):
        setattr(model_args, 'unfreeze_start_layer', 0)
    if not hasattr(model_args, 'finetune'):
        setattr(model_args, 'finetune', False)
    model = NewMDGenWrapper(model_args)
    # ---- 載入 checkpoint 權重 ----
    state_dict = ckpt["state_dict"]
    # Lightning 存的是 wrapper 的 state_dict
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("missing keys =", missing)
    print("unexpected keys =", unexpected)
    model.eval().to('cuda')

    #@@@@@@@@ postprocess
    df = pd.read_csv(args.split)
    csv_dir = args.csv_dir
    for idx, row in df.iterrows():
        protein = row['proteins']
        fragment = row['fragments']
        try:
            print(f"\n[INFO] Processing {protein}-{fragment} ...")
            do(model, protein, fragment, csv_dir)
            print(f"[INFO] Finished {protein}-{fragment}")

        except Exception as e:
            print(f"[ERROR] Failed on {protein}-{fragment}: {e}")
            continue   # skip and go to the next one
main()