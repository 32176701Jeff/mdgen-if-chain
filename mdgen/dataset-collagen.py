import torch
from .rigid_utils import Rigid
from .residue_constants import restype_order
import numpy as np
import pandas as pd
from .geometry import atom37_to_torsions, atom14_to_atom37, atom14_to_frames
import os
       
class MDGenDataset(torch.utils.data.Dataset):
    def __init__(self, args, split, repeat=1):
        super().__init__()
        self.df = pd.read_csv(split)
        self.args = args
        self.repeat = repeat
        self.csv_dir = args.csv_dir
        self.cropping = args.crop
    def __len__(self):
        if self.args.overfit_peptide:
            return 1000
        return self.repeat * len(self.df)

    def __getitem__(self, idx):
        idx = idx % len(self.df)
        Letter = ['A', 'B', 'C']
        torsions_list = []
        torsion_mask_list = []
        trans_list = []
        rots_list = []
        seqres_list = []
        mask_list = []
        chain_id_list = []

        start = 0  # residue start
        frame_start = 0
        end = frame_start + self.args.num_frames

        protein = self.df.proteins[idx]
        fragment = self.df.fragments[idx]
        window_csv = f'{self.csv_dir}/{protein}-{fragment}.csv'
        self.df_window = pd.read_csv(window_csv, index_col='chain')
        full_name = f'{protein}_{fragment}'

        import random

        # --------------------------------------------------
        # sample-level cropping setting
        # r ~ atlas_gap_distribution, s ~ [0, r]
        # slicing follows your requested format: [s : self.cropping - r]
        # --------------------------------------------------
        crop_start = 0
        crop_end = self.cropping

        for letter in Letter:
            seqres = self.df_window.seqres[letter]
            k = random.choice([1, 2, 3])
            arr = np.lib.format.open_memmap(
                f'{self.args.data_dir}/chain-r{k}/{protein}-{fragment}-{letter}.npy', 'r'
            )

            arr = np.copy(arr[frame_start:end]).astype(np.float32)  # / 10.0 # convert to nm
            if self.args.copy_frames:
                arr[1:] = arr[0]

            # arr should be in ANGSTROMS
            frames = atom14_to_frames(torch.from_numpy(arr))
            seqres = np.array([restype_order[c] for c in seqres])
            aatype = torch.from_numpy(seqres)[None].expand(self.args.num_frames, -1)
            atom37 = torch.from_numpy(atom14_to_atom37(arr, aatype)).float()

            L = frames.shape[1]
            mask = np.ones(L, dtype=np.float32)
            torsions, torsion_mask = atom37_to_torsions(atom37, aatype)
            torsion_mask = torsion_mask[0]

            # @@@@@@@@@@@ datapreprocess chainID
            cid = {'A': 0, 'B': 1, 'C': 2}[letter]
            chain_id = np.full(len(seqres), cid, dtype=np.int64)
            ## @@@@@@@@@@@ datapreprocess chainID

            # @@@@@@@@@@@ cropping window @@@@@@@@@@@
            torsions = torsions[:, crop_start:crop_end]
            frames = frames[:, crop_start:crop_end]
            seqres = seqres[crop_start:crop_end]
            mask = mask[crop_start:crop_end]
            torsion_mask = torsion_mask[crop_start:crop_end]
            chain_id = chain_id[crop_start:crop_end]
            ## @@@@@@@@@@@ cropping window @@@@@@@@@@@

            # @@@@@@@@@@@ datapreprocess concate1
            torsions_list.append(torsions)                     # (T, L, 7, 2)
            torsion_mask_list.append(torsion_mask)             # (L, 7)
            trans_list.append(frames._trans)                   # (T, L, 3)
            rots_list.append(frames._rots._rot_mats)           # (T, L, 3, 3)
            seqres_list.append(seqres)                         # (L,)
            mask_list.append(mask)                             # (L,)
            chain_id_list.append(chain_id)                     # (L,)
            ## @@@@@@@@@@@ datapreprocess concate1

        # @@@@@@@@@@@ datapreprocess concate2
        torsions = torch.cat(torsions_list, dim=1)          # (T, L_total, 7, 2)
        torsion_mask = torch.cat(torsion_mask_list, dim=0)  # (L_total, 7)
        trans = torch.cat(trans_list, dim=1)                # (T, L_total, 3)
        rots = torch.cat(rots_list, dim=1)                  # (T, L_total, 3, 3)
        seqres = np.concatenate(seqres_list, axis=0)        # (L_total,)
        mask = np.concatenate(mask_list, axis=0)            # (L_total,)
        chain_ids = np.concatenate(chain_id_list, axis=0)   # (L_total,)
        ## @@@@@@@@@@@ datapreprocess concate2

        # @@@@@@@@@@@ add padding after chain C to make cropping window = 256 @@@@@@@@@@@
        current_len = len(seqres)
        pad = 256 - current_len

        if pad < 0:
            raise ValueError(
                f'Concatenated length {current_len} exceeds cropping window 256 '
            )

        if pad > 0:
            frames = Rigid(
                rots=Rigid.identity((1, 1), requires_grad=False, fmt='rot_mat')._rots.__class__(rots),
                trans=trans
            )
            frames = Rigid.cat([
                frames,
                Rigid.identity((self.args.num_frames, pad), requires_grad=False, fmt='rot_mat')
            ], 1)

            mask = np.concatenate([mask, np.zeros(pad, dtype=np.float32)])
            seqres = np.concatenate([seqres, np.zeros(pad, dtype=int)])
            torsions = torch.cat(
                [torsions, torch.zeros((torsions.shape[0], pad, 7, 2), dtype=torch.float32)],
                1
            )
            torsion_mask = torch.cat(
                [torsion_mask, torch.zeros((pad, 7), dtype=torch.float32)]
            )
            chain_ids = np.concatenate([chain_ids, np.full(pad, 3, dtype=np.int64)])

            trans = frames._trans
            rots = frames._rots._rot_mats
        # @@@@@@@@@@@ add padding after chain C to make cropping window = 256 @@@@@@@@@@@

        return {
            'name': full_name,
            'frame_start': frame_start,
            'torsions': torsions,
            'torsion_mask': torsion_mask,
            'trans': trans,  # frames._trans
            'rots': rots,    # frames._rots._rot_mats
            'seqres': seqres,
            'mask': mask,    # (L,)
            'chain_ids': chain_ids,  # chainID
        }