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



def get_batch(name, seqres, letter, num_frames):
    #@@@@@@@@@@@@@@@ postprocess
    arr = np.lib.format.open_memmap(f'{args.data_dir}/{name}{args.suffix}_fit.npy', 'r')
    cid = {'A': 0, 'B': 1, 'C': 2}[letter]
    chain_ids = torch.full((len(seqres),), cid, dtype=torch.long)
    #@@@@@@@@@@@@@@@ postprocess

    if not args.tps: # else keep all frames
        arr = np.copy(arr[0:1]).astype(np.float32)
    frames = atom14_to_frames(torch.from_numpy(arr))

    ####如果遇到沒有見過的胺基酸 就設為20
    seqres = torch.tensor([restype_order[c] if c in restype_order else 20 for c in seqres])
    ####如果遇到沒有見過的胺基酸 就設為20

    atom37 = torch.from_numpy(atom14_to_atom37(arr, seqres[None])).float()
    L = len(seqres)
    mask = torch.ones(L)
    if args.no_frames:
        return {
            'atom37': atom37,
            'seqres': seqres,
            'mask': restype_atom37_mask[seqres],
        }
        
    torsions, torsion_mask = atom37_to_torsions(atom37, seqres[None])
    return {
        'torsions': torsions,
        'torsion_mask': torsion_mask[0],
        'trans': frames._trans,
        'rots': frames._rots._rot_mats,
        'seqres': seqres,
        'mask': mask, # (L,)
        'chain_ids': chain_ids, #@@@@@@@@ postprocess
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
    
#@@@@@@@@@@@@@@ postprocess
def do(model, name, seqres):

    Letter = ['A']
    # ---------- build batch for A/B/C ----------
    batches = []
    seqres_list = []
    lengths = []

    for letter in Letter:
        item = get_batch(name, seqres, letter,
                         num_frames=model.args.num_frames)
        batch = next(iter(torch.utils.data.DataLoader([item])))
        batch = tensor_tree_map(lambda x: x.cuda(), batch)

        batches.append(batch)
        seqres_list.append(seqres)
        lengths.append(len(seqres))

    # ---------- concat batches (along residue dim) ----------
    batch_abc = {}

    for key in batches[0].keys():
        if key in ['seqres', 'mask', 'torsion_mask']:
            batch_abc[key] = torch.cat([b[key] for b in batches], dim=1)
        elif key == 'chain_ids':
            batch_abc[key] = torch.cat([b[key] for b in batches], dim=0)
        elif key in ['torsions', 'rots']:
            batch_abc[key] = torch.cat([b[key] for b in batches], dim=2)
        elif key == 'trans':
            batch_abc[key] = torch.cat([b[key] for b in batches], dim=2)
        else:
            batch_abc[key] = batches[0][key]

    # ---------- rollout ----------
    all_atom14 = []
    start = time.time()

    batch = batch_abc
    for _ in tqdm.trange(args.num_rollouts):
        atom14, batch = rollout(model, batch)
        all_atom14.append(atom14)
    
    print(time.time() - start)
    all_atom14 = torch.cat(all_atom14, 1)
    print('1')

# ---------- split atom14 back to A/B/C ----------
    LA = lengths[0]
    idxA = slice(0, LA)
    atom14_A = all_atom14[:, :, idxA]
    # ---------- postprocess ----------
    for letter, atom14_chain, idx in zip(
            Letter,
            [atom14_A],
            [idxA]):

        out_name = f'{name}'
        path = os.path.join(args.out_dir, f'{out_name}.pdb')

        atom14_to_pdb(
            atom14_chain[0].cpu().numpy(),
            batch['seqres'][0, idx].cpu().numpy(),
            path
        )

        if args.xtc:
            traj = mdtraj.load(path)
            traj.superpose(traj)
            traj.save(os.path.join(args.out_dir, f'{out_name}.xtc'))
            traj[0].save(path)
#@@@@@@@@@@@@@@ postprocess

@torch.no_grad()
def main():

    # model = NewMDGenWrapper.load_from_checkpoint(args.sim_ckpt)
    # model.eval().to('cuda')
    #@@@@@@@@ postprocess
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

    df = pd.read_csv(args.split, index_col='name')
    #@@@@@@@@ postprocess
    for name in df.index:
        if args.pdb_id and name not in args.pdb_id:
            continue
        try:
            print(f"\n[INFO] Processing {name} ...")
            seqres = df.seqres[name]
            do(model, name, seqres)
            print(f"[INFO] Finished {name}")

        except Exception as e:
            print(f"[ERROR] Failed on {name}: {e}")
            continue   # skip and go to the next one
main()