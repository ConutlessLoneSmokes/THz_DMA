"""Unitary representations; complex physical channels retain their original units."""
import numpy as np
import torch
import json
from pathlib import Path, PureWindowsPath

from thz_dma.estimators.diffusion import pack_channel, unpack_channel
from thz_dma.paths import relative_path


def cache_directory(directory, metadata):
    name = metadata.get("source_run_id") or PureWindowsPath(metadata["source_directory"]).name
    path = relative_path(name)
    if len(path.parts) != 1:
        raise ValueError("cached source must name one sibling run directory")
    return Path(directory).parent / path


def angular(h, *, inverse=False):
    left = torch.fft.fft if inverse else torch.fft.ifft
    right = torch.fft.ifft if inverse else torch.fft.fft
    return right(left(h, dim=-2, norm="ortho"), dim=-1, norm="ortho")


def dma_spatial(image, *, inverse=False):
    transform = torch.fft.ifft2 if inverse else torch.fft.fft2
    return pack_channel(transform(unpack_channel(image), dim=(-2, -1), norm="ortho"))


def encode(channel, scenario):
    if scenario == "mimo":
        transformed = angular(channel)
        return torch.stack((transformed.real, transformed.imag), dim=1)
    if scenario == "dma":
        return dma_spatial(pack_channel(channel))
    raise ValueError("unknown channel scenario")


def decode(image, scenario):
    if scenario == "mimo":
        return angular(torch.complex(image[:, 0], image[:, 1]), inverse=True)
    if scenario == "dma":
        return unpack_channel(dma_spatial(image, inverse=True))
    raise ValueError("unknown channel scenario")


def signal_scale(noisy_energy, count, noise_real_variance):
    """Training-only complex RMS after subtracting the known complex noise power."""
    if count < 1 or not np.isfinite(noise_real_variance) or noise_real_variance < 0:
        raise ValueError("positive sample count and nonnegative known noise required")
    power = noisy_energy / count - 2 * noise_real_variance
    if not np.isfinite(power) or power <= 0:
        raise ValueError("training noisy moments do not resolve positive signal power")
    return float(np.sqrt(power))


class ChannelData:
    """Read only a batch of physical CSI; cache datasets keep existing NPZ files."""
    def __init__(self, directory, metadata):
        self.directory, self.metadata = Path(directory), metadata
        self.scenario = metadata["scenario"]
        self.scale = float(metadata["complex_scale"])
        if self.scenario not in ("mimo", "dma") or not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("valid scenario and positive data scale required")
        self.counts = metadata["counts"]
        if any(not isinstance(n, int) or n < 1 for n in self.counts.values()):
            raise ValueError("positive split counts required")
        self.shape = tuple(metadata["channel_shape"])
        self.arrays = {}
        self.parameters = {}
        if metadata["storage"] == "npy":
            for split, name in metadata["files"].items():
                array = np.load(self.directory / relative_path(name), mmap_mode="r", allow_pickle=False)
                if array.shape != (self.counts[split], *self.shape) or array.dtype != np.complex64:
                    raise ValueError("channel array shape/dtype differs from its manifest")
                self.arrays[split] = array
        elif metadata["storage"] == "existing_npz":
            self.source_directory = cache_directory(self.directory, metadata)
        else:
            raise ValueError("unknown channel storage")

    def physical(self, split, indices):
        indices = list(indices)
        if not indices or min(indices) < 0 or max(indices) >= self.counts[split]:
            raise ValueError("sample indices exceed the declared split")
        if self.metadata["storage"] == "npy":
            if not self.arrays:
                raise ValueError("channel dataset is closed")
            channels = np.array(self.arrays[split][indices], copy=True)
        else:
            values = []
            for index in indices:
                path = self.source_directory / "data" / relative_path(self.metadata["index"][split][index])
                with np.load(path, allow_pickle=False) as file:
                    values.append(file["channel"])
            channels = np.stack(values)
        if channels.shape != (len(indices), *self.shape) or not np.isfinite(channels).all():
            raise ValueError("nonfinite channel or inconsistent physical shape")
        return channels

    def images(self, split, indices, device="cpu", *, dtype=torch.float32):
        physical = torch.as_tensor(self.physical(split, indices), device=device)
        return encode(physical / self.scale, self.scenario).to(dtype=dtype)

    def record(self, split, index):
        if split not in self.counts or not 0 <= index < self.counts[split]:
            raise ValueError("parameter index exceeds the declared split")
        if self.metadata["storage"] != "existing_npz":
            scene_id = f"{self.metadata['split_seeds'][split]}:{index}"
            names = [name for name in self.metadata.get("file_sha256", {})
                     if name.replace("\\", "/").startswith(f"parameters/{split}_") and name.endswith(".json")]
            if not names:
                return {"scene_id": scene_id}
            starts = {int(relative_path(name).stem.rsplit("_", 1)[1]): name for name in names}
            start = max((value for value in starts if value <= index), default=-1)
            if start < 0:
                raise ValueError("missing channel parameter batch")
            name = starts[start]
            if name not in self.parameters:
                self.parameters[name] = json.loads((self.directory / relative_path(name)).read_text(encoding="utf-8"))
            records = self.parameters[name]
            offset = index - start
            if offset >= len(records) or records[offset].get("scene_id") != scene_id:
                raise ValueError("channel parameters do not match the declared scene")
            return records[offset]
        path = self.source_directory / "data" / relative_path(self.metadata["index"][split][index])
        with np.load(path, allow_pickle=False) as file:
            return json.loads(str(file["record"]))

    def close(self):
        self.arrays.clear()
        self.parameters.clear()
