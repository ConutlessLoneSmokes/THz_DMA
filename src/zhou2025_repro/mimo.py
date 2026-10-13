"""Minimal two-ended M0 channel adapter; physical building blocks are shared."""
import numpy as np
import torch

from thz_dma.channels.atmosphere import C0_M_PER_S
from thz_dma.channels.geometry import far_field_array_response, ula_positions
from thz_dma.channels.planar import (planar_dma_positions, sample_stage2_paths,
                                    separable_compatibility_path_channel)


def parametric_mimo(config, rng, absorption_table):
    settings = config["mimo"]
    carrier = float(config["system"]["carrier_frequency_hz"])
    spacing = (float(settings["spacing_m"]) if "spacing_m" in settings else
               float(settings["spacing_wavelengths"]) * C0_M_PER_S / carrier)
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("positive finite physical antenna spacing required")
    nr, nt = int(settings["receive_antennas"]), int(settings["transmit_antennas"])
    receive = planar_dma_positions(1, nr, spacing, spacing)
    transmit = ula_positions(nt, spacing)
    paths = sample_stage2_paths(config, rng)
    angles = rng.uniform(settings["departure_min_deg"], settings["departure_max_deg"], len(paths))
    channel = np.zeros((nr, nt), dtype=np.complex128)
    record = []
    for path, departure in zip(paths, angles):
        rx = separable_compatibility_path_channel([carrier], receive, path,
                                                  carrier_frequency_hz=carrier).reshape(nr)
        if config["channel"]["include_absorption"]:
            rx *= absorption_table.amplitude_transmission(
                carrier, path.range_m, method=config["atmosphere"]["interpolation"])
        tx = far_field_array_response(carrier, transmit, np.deg2rad(departure), normalize=False)
        channel += rx[:, None] * tx.conj()[None]
        record.append({"range_m": path.range_m, "arrival_azimuth_rad": path.azimuth_rad,
                       "arrival_elevation_rad": path.elevation_rad,
                       "departure_azimuth_rad": float(np.deg2rad(departure)),
                       "gain_real": path.complex_gain.real, "gain_imag": path.complex_gain.imag})
    return channel.astype(np.complex64), record


def qpsk_pilots(nt, count, rng, *, device="cpu"):
    if nt < 1 or count < 1:
        raise ValueError("positive pilot and antenna counts required")
    symbols = (2 * rng.integers(0, 2, (nt, count)) - 1
               + 1j * (2 * rng.integers(0, 2, (nt, count)) - 1)) / np.sqrt(2)
    return torch.as_tensor(symbols, device=device, dtype=torch.complex64)


class MimoOperator:
    """Receive-DFT observations: Y_ad = H_ad (F_T^H P)."""
    def __init__(self, pilots, receive_antennas):
        if (pilots.ndim != 2 or min(pilots.shape) < 1 or receive_antennas < 1
                or not torch.isfinite(pilots).all()):
            raise ValueError("finite pilot matrix and positive receive count required")
        self.b = torch.fft.ifft(pilots, dim=0, norm="ortho")
        self.image_shape = (2, int(receive_antennas), len(pilots))
        gram = self.b.T @ self.b.conj()
        values, self.vectors = torch.linalg.eigh((gram + gram.H) / 2)
        self.values = values.clamp_min(0)
        self.pilots = pilots

    def forward(self, image):
        return torch.complex(image[:, 0], image[:, 1]) @ self.b

    def adjoint(self, observation):
        image = observation @ self.b.H
        return torch.stack((image.real, image.imag), dim=1)

    def likelihood_score(self, image, observation, variance, alpha_bar):
        if not (np.isfinite(variance) and variance > 0 and 0 < alpha_bar < 1):
            raise ValueError("positive variance and alpha_bar in (0,1) required")
        root = np.sqrt(alpha_bar)
        residual = observation - self.forward(image) / root
        denominator = variance + 2 * (1 - alpha_bar) / alpha_bar * self.values
        dual = ((residual @ self.vectors.conj()) / denominator) @ self.vectors.T
        return 2 * self.adjoint(dual) / root
