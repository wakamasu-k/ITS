"""Run the inherited Stage-1 trainer with the installed PyTorch AMP API."""
from pathlib import Path
import runpy
import sys

import torch


def main():
    if not hasattr(torch.amp, 'GradScaler'):
        def cuda_grad_scaler(device, *, enabled=True):
            if device != 'cuda':
                raise ValueError(f'Legacy GradScaler only supports CUDA: {device}')
            return torch.cuda.amp.GradScaler(enabled=enabled)

        torch.amp.GradScaler = cuda_grad_scaler
        print(f'[compat] PyTorch {torch.__version__}: using torch.cuda.amp.GradScaler', flush=True)

    trainer = (Path(__file__).resolve().parents[2] / 'handover_shimizu/src/LoFTR/tools'
               / 'train_stage1_step2a_backbone0_coarse_fast_bf16v2.py')
    sys.argv[0] = str(trainer)
    runpy.run_path(str(trainer), run_name='__main__')


if __name__ == '__main__':
    main()
