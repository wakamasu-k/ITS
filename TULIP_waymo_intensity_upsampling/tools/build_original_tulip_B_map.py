"""Use the existing B map builder, keeping TensorFlow preprocessing on CPU.

PyTorch TULIP inference still uses the requested CUDA device. This prevents
TensorFlow and PyTorch from reserving the same GPU memory during map building.
"""
from pathlib import Path
import faulthandler
import runpy
import sys


def main():
    source=Path(__file__).with_name('build_tulip_16_to_32_static_map.py')
    diagnostic='--help' not in sys.argv and '-h' not in sys.argv
    if diagnostic:
        faulthandler.enable()
        # Prints the waiting Python stack, but does not terminate the process.
        faulthandler.dump_traceback_later(120,repeat=True)
    try:
        if diagnostic:
            # Load PyTorch's native libraries before TensorFlow's runtime.
            # Standalone CUDA checks already use this order; report each import.
            print('[B startup] Importing PyTorch before TensorFlow',flush=True)
            import torch
            print('[B startup] PyTorch imported: '+str(torch.__version__),flush=True)
            print('[B startup] Importing TensorFlow (geometry will use CPU)',flush=True)
            import tensorflow as tf
            tf.config.set_visible_devices([], 'GPU')
            print('[B startup] TensorFlow geometry: CPU; importing PyTorch/map builder next',flush=True)
        sys.argv[0]=str(source)
        runpy.run_path(str(source),run_name='__main__')
    finally:
        if diagnostic: faulthandler.cancel_dump_traceback_later()


if __name__=='__main__': main()
