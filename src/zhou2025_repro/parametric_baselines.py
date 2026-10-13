"""DFT-bin OMP and an explicitly genie-aided MIMO geometry reference."""
import numpy as np
import torch

from thz_dma.channels.atmosphere import C0_M_PER_S
from thz_dma.channels.geometry import far_field_array_response, ula_positions
from thz_dma.channels.planar import (PlanarPath, planar_dma_positions,
                                    separable_compatibility_path_channel)
from thz_dma.estimators.planar_sparse import (PlanarTemplateDictionary,
                                            _fit_dictionary_support,
                                            omp_three_path_estimate)
from .data import angular


def _validate_observation(y, b):
    if (y.ndim != 3 or b.ndim != 2 or min(*y.shape, *b.shape) < 1
            or y.shape[-1] != b.shape[-1] or not y.is_complex() or not b.is_complex()
            or not torch.isfinite(y).all() or not torch.isfinite(b).all()):
        raise ValueError("finite complex (B,Nr,Np) observations and (Nt,Np) operator required")


def angular_omp(y, b, max_atoms=3):
    """Independently recover each receive-DFT row with a fixed DFT-bin budget.

    The budget is declared before evaluation; it is never read from scene truth.
    All selected gains are jointly refitted by the shared complex OMP kernel.
    """
    _validate_observation(y, b)
    if isinstance(max_atoms, bool) or not isinstance(max_atoms, int) or not 1 <= max_atoms <= b.shape[0]:
        raise ValueError("OMP max_atoms must be an integer in [1,Nt]")
    matrix = b.detach().cpu().numpy()
    values = y.detach().cpu().numpy()
    # Unique bin labels let the shared OMP kernel exclude selected candidates.
    labels = np.arange(b.shape[0], dtype=float)
    dictionary = PlanarTemplateDictionary(labels, np.zeros_like(labels),
                                          np.zeros_like(labels), matrix)
    estimates = np.zeros((*y.shape[:-1], b.shape[0]), dtype=complex)
    supports = np.empty((*y.shape[:-1], max_atoms), dtype=int)
    residuals = np.empty(y.shape[:-1])
    conditions = np.empty(y.shape[:-1])
    for batch in range(len(values)):
        for row in range(values.shape[1]):
            fit = omp_three_path_estimate(dictionary, values[batch, row], num_paths=max_atoms)
            supports[batch, row] = fit["candidate_index"]
            estimates[batch, row, fit["candidate_index"]] = fit["complex_gain"]
            residuals[batch, row] = fit["residual_energy"]
            conditions[batch, row] = fit["support_condition_number"]
    return torch.as_tensor(estimates, device=y.device, dtype=y.dtype), {
        "candidate_index": supports, "residual_energy": residuals,
        "support_condition_number": conditions, "max_atoms": max_atoms}


def mimo_path_templates(config, path_records, absorption_table):
    """Physical unit-gain templates using geometry only; saved true gains are ignored."""
    settings = config["mimo"]
    carrier = float(config["system"]["carrier_frequency_hz"])
    if not np.isfinite(carrier) or carrier <= 0:
        raise ValueError("positive finite carrier required")
    spacing = (float(settings["spacing_m"]) if "spacing_m" in settings else
               float(settings["spacing_wavelengths"]) * C0_M_PER_S / carrier)
    if (not np.isfinite([carrier, spacing]).all() or carrier <= 0 or spacing <= 0
            or not path_records):
        raise ValueError("positive finite carrier/spacing and saved path geometry required")
    nr, nt = int(settings["receive_antennas"]), int(settings["transmit_antennas"])
    receive = planar_dma_positions(1, nr, spacing, spacing)
    transmit = ula_positions(nt, spacing)
    templates = []
    for record in path_records:
        path = PlanarPath(float(record["range_m"]), float(record["arrival_azimuth_rad"]),
                          float(record["arrival_elevation_rad"]), 1. + 0.j)
        departure = float(record["departure_azimuth_rad"])
        if not np.isfinite(departure):
            raise ValueError("saved departure angle must be finite")
        rx = separable_compatibility_path_channel([carrier], receive, path,
                                                  carrier_frequency_hz=carrier).reshape(nr)
        if config["channel"]["include_absorption"]:
            if absorption_table is None:
                raise ValueError("absorption table required for geometry templates")
            rx *= absorption_table.amplitude_transmission(
                carrier, path.range_m, method=config["atmosphere"]["interpolation"])
        tx = far_field_array_response(carrier, transmit, departure, normalize=False)
        templates.append(rx[:, None] * tx.conj()[None])
    values = np.stack(templates)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite path geometry templates")
    return values


def oracle_true_geometry(y, b, templates):
    """Known true geometry with unknown gains fitted to the supplied observation.

    Templates have shape (L,Nr,Nt) and must use the same channel units as y.
    This reference requires scene truth and is not a deployable estimator.
    """
    _validate_observation(y, b)
    physical = torch.as_tensor(templates, dtype=torch.complex128, device="cpu").detach()
    if (physical.ndim != 3 or len(physical) < 1
            or tuple(physical.shape[1:]) != (y.shape[1], b.shape[0])
            or not torch.isfinite(physical).all()):
        raise ValueError("finite nonempty (L,Nr,Nt) unit-gain path templates required")
    basis = angular(physical).numpy()
    measured = (basis @ b.detach().cpu().numpy()).reshape(len(basis), -1)
    if not np.isfinite(measured).all():
        raise ValueError("geometry observations must be finite")
    if np.any(np.sum(abs(measured)**2, axis=1) <= 0):
        raise ValueError("geometry contains an unobserved zero-energy path")
    estimates = np.empty((len(y), *basis.shape[1:]), dtype=complex)
    gains = np.empty((len(y), len(basis)), dtype=complex)
    residuals, conditions = np.empty(len(y)), np.empty(len(y))
    for batch, observation in enumerate(y.detach().cpu().numpy()):
        fit, _, residuals[batch], conditions[batch] = _fit_dictionary_support(
            measured, observation.reshape(-1), list(range(len(basis))))
        gains[batch] = fit
        estimates[batch] = np.einsum("l,lrt->rt", fit, basis)
    return torch.as_tensor(estimates, device=y.device, dtype=y.dtype), {
        "complex_gain": gains, "residual_energy": residuals,
        "support_condition_number": conditions, "oracle_geometry": True}
