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
        self.df = pd.read_csv(split, index_col='name')
        self.args = args
        self.repeat = repeat
    def __len__(self):
        if self.args.overfit_peptide:
            return 1000
        return self.repeat * len(self.df)

    def __getitem__(self, idx):
        idx = idx % len(self.df)

        name = self.df.index[idx]
        seqres = self.df.loc[name, 'seqres']
        full_name = name

        kkk = np.random.randint(1, 4)
        arr = np.lib.format.open_memmap(
            f'{self.args.data_dir}/{name}_R{kkk}_fit.npy', 'r'
        )

        frame_start = self.args.start
        frame_stride = self.args.stride
        frame_end = frame_start + self.args.num_frames * frame_stride

        if frame_stride <= 0:
            raise ValueError(f"stride must be > 0, got {frame_stride}")

        if frame_start < 0:
            raise ValueError(f"start must be >= 0, got {frame_start}")

        if frame_start >= arr.shape[0]:
            raise ValueError(
                f"start ({frame_start}) is out of range for trajectory with "
                f"{arr.shape[0]} frames"
            )

        arr = np.copy(arr[frame_start:frame_end:frame_stride]).astype(np.float32)  # / 10.0  # convert to nm

        if arr.shape[0] < self.args.num_frames:
            raise ValueError(
                f"Not enough frames after slicing: got {arr.shape[0]}, "
                f"need {self.args.num_frames}. "
                f"start={frame_start}, stride={frame_stride}, total_available_frames={arr.shape[0]}"
            )

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

        # only chain A
        chain_id = np.full(len(seqres), 0, dtype=np.int64)

        if self.args.atlas:
            if L > self.args.crop:
                start = np.random.randint(0, L - self.args.crop + 1)

                torsions = torsions[:, start:start + self.args.crop]
                frames = frames[:, start:start + self.args.crop]
                seqres = seqres[start:start + self.args.crop]
                mask = mask[start:start + self.args.crop]
                torsion_mask = torsion_mask[start:start + self.args.crop]
                chain_id = chain_id[start:start + self.args.crop]

            elif L < self.args.crop:
                pad = self.args.crop - L

                frames = Rigid.cat([
                    frames,
                    Rigid.identity(
                        (self.args.num_frames, pad),
                        requires_grad=False,
                        fmt='rot_mat'
                    )
                ], 1)

                mask = np.concatenate([mask, np.zeros(pad, dtype=np.float32)])
                seqres = np.concatenate([seqres, np.zeros(pad, dtype=int)])
                torsions = torch.cat([
                    torsions,
                    torch.zeros((torsions.shape[0], pad, 7, 2), dtype=torch.float32)
                ], 1)
                torsion_mask = torch.cat([
                    torsion_mask,
                    torch.zeros((pad, 7), dtype=torch.float32)
                ])
                chain_id = np.concatenate([
                    chain_id,
                    np.full(pad, 3, dtype=np.int64)
                ])

        return {
            'name': full_name,
            'frame_start': frame_start,
            'torsions': torsions,
            'torsion_mask': torsion_mask,
            'trans': frames._trans,
            'rots': frames._rots._rot_mats,
            'seqres': seqres,
            'mask': mask,
            'chain_ids': chain_id,
        }

