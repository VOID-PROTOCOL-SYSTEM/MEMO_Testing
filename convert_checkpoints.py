"""One-time, torch-free converter: policy_wings{2,3,4}.pt  ->  .npz

Run this once (locally, wherever the .pt files currently live) to produce
the .npz weight files the WASM notebook actually loads at runtime. It does
NOT need torch installed - it reads the raw pickle/zip structure that
torch.save() produces directly.

Usage:
    python convert_checkpoints.py policy_wings2.pt policy_wings3.pt policy_wings4.pt
    # writes policy_wings2.npz, policy_wings3.npz, policy_wings4.npz next to each input
"""
import io
import os
import pickle
import sys
import zipfile

import numpy as np


class _FakeStorage:
    def __init__(self, flat_array):
        self.flat = flat_array


def _rebuild_tensor_v2(storage, storage_offset, size, stride, *_):
    flat = storage.flat
    n = 1
    for s in size:
        n *= s
    arr = flat[storage_offset:storage_offset + n]
    return np.asarray(arr, dtype=flat.dtype).reshape(size)


class _TorchFreeUnpickler(pickle.Unpickler):
    """Unpickles a torch.save() state-dict archive without importing torch.
    Only supports what a plain nn.Module state_dict needs: nested
    dict/OrderedDict of float32 CPU tensors - exactly what Actor/Critic
    checkpoints are."""

    def __init__(self, file, zf, prefix):
        super().__init__(file)
        self._zf = zf
        self._prefix = prefix

    def find_class(self, module, name):
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _rebuild_tensor_v2
        if module == "torch" and name.endswith("Storage"):
            return lambda *a, **kw: None
        if module == "collections" and name == "OrderedDict":
            import collections
            return collections.OrderedDict
        return super().find_class(module, name)

    def persistent_load(self, pid):
        assert pid[0] == "storage"
        key = pid[2]
        numel = pid[4]
        raw = self._zf.read(f"{self._prefix}/data/{key}")
        flat = np.frombuffer(raw, dtype=np.float32, count=numel).copy()
        return _FakeStorage(flat)


def torch_free_load(path):
    zf = zipfile.ZipFile(path)
    prefix = zf.namelist()[0].split("/")[0]
    data = zf.read(f"{prefix}/data.pkl")
    return _TorchFreeUnpickler(io.BytesIO(data), zf, prefix).load()


def convert(pt_path):
    ckpt = torch_free_load(pt_path)
    flat = {}
    for top in ("actor", "critic"):
        for k, v in ckpt[top].items():
            flat[f"{top}.{k}"] = v
    out_path = os.path.splitext(pt_path)[0] + ".npz"
    np.savez(out_path, n_wings=np.array(ckpt.get("n_wings", -1)), **flat)
    print(f"{pt_path}  ->  {out_path}  ({len(flat)} arrays)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for p in sys.argv[1:]:
        convert(p)
