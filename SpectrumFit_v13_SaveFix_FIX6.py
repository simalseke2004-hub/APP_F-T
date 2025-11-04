#!/usr/bin/env python3
"""
All-in-one Spectrum Fit GUI (Tkinter) — v5
- Fixes indentation error in v4.
- Includes toggle for red dot placement (nearest data vs fitted y at μ).
- Alt-fit (download only) k capped by available peaks.
- Timestamped saving for all plot types and CSVs.

Run:
    python spectrum_fit_allinone_gui_v5.py
"""
from __future__ import annotations


# --- Injected: Tk / DnD safety imports and Matplotlib embedding ---
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
# Optional drag-and-drop support
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    DND_AVAILABLE = True
except Exception:
    TkinterDnD = None
    DND_FILES = None
    DND_AVAILABLE = False

def _have_dnd() -> bool:
    return bool(DND_AVAILABLE and (TkinterDnD is not None))

import matplotlib as _mpl
try:
    # Force TkAgg to avoid blank canvas on macOS when default backend is non-interactive
    _mpl.use('TkAgg')
except Exception:
    pass
import sys, os, platform
NON_INTERACTIVE = False
BOOT_ERROR = None
try:
    import tkinter as _tk  # ensure Tk is present
    import matplotlib as _mpl
    try:
        _mpl.use('TkAgg', force=True)
    except Exception as _be:
        BOOT_ERROR = _be
        NON_INTERACTIVE = True
except Exception as _e:
    BOOT_ERROR = _e
    NON_INTERACTIVE = True
from matplotlib.figure import Figure
try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
except Exception as _e:
    BOOT_ERROR = BOOT_ERROR or _e
    NON_INTERACTIVE = True

import re
# --- End injected imports ---

# ==== Peak shape models & dispatcher (added) ====
def gaussian_profile(x, A, mu, sigma):
    x = np.asarray(x, dtype=float)
    sigma = float(abs(sigma)) if np.isfinite(sigma) else 1.0
    return float(A) * np.exp(-0.5 * ((x - float(mu)) / max(sigma, 1e-12)) ** 2)

def gaussian_with_baseline(x, y0, A, mu, sigma):
    return float(y0) + gaussian_profile(x, A, mu, sigma)

def multi_gaussian_with_baseline(x, *params):
    x = np.asarray(x, dtype=float)
    if len(params) == 0:
        return np.zeros_like(x, dtype=float)
    y0 = float(params[0])
    y = np.full_like(x, y0, dtype=float)
    for i in range(1, len(params), 3):
        if i + 2 >= len(params):
            break
        A, mu, sg = params[i], params[i+1], params[i+2]
        y += gaussian_profile(x, A, mu, sg)
    return y

def pseudo_voigt_profile(x, A, mu, sigma):
    x = np.asarray(x, dtype=float)
    sigma = float(abs(sigma)) if np.isfinite(sigma) else 1.0
    fwhm = 2.3548200450309493 * max(sigma, 1e-12)
    gamma = 0.5 * fwhm
    G = np.exp(-0.5 * ((x - float(mu)) / max(sigma, 1e-12)) ** 2)
    L = 1.0 / (1.0 + ((x - float(mu)) / max(gamma, 1e-12)) ** 2)
    return float(A) * (0.5 * G + 0.5 * L)

def emg_profile(x, A, mu, sigma):
    import math
    x = np.asarray(x, dtype=float)
    sigma = float(abs(sigma)) if np.isfinite(sigma) else 1.0
    tau = max(sigma, 1e-12)
    k = 1.0 / tau
    # Stable exponent with clipping to avoid overflow
    expo = k * (float(mu) - x) + 0.5 * (k * sigma) ** 2
    expo = np.clip(expo, -700.0, 700.0)
    # erfc is bounded in [0,2], vectorized via math.erfc
    arg = (float(mu) + (k * sigma * sigma) - x) / (math.sqrt(2.0) * max(sigma, 1e-12))
    erfc_vec = np.vectorize(math.erfc)
    return float(A) * 0.5 * np.exp(expo) * erfc_vec(arg)


def make_multi_model(profile_fn):
    def multi_fn(x, *params):
        x = np.asarray(x, dtype=float)
        if len(params) == 0:
            return np.zeros_like(x, dtype=float)
        y0 = float(params[0])
        y = np.full_like(x, y0, dtype=float)
        for i in range(1, len(params), 3):
            if i + 2 >= len(params):
                break
            A, mu, sg = params[i], params[i+1], params[i+2]
            y += profile_fn(x, A, mu, sg)
        return y
    return multi_fn

def get_multipeak_model_by_name(name: str):
    n = str(name).strip().lower() if isinstance(name, str) else "gaussian"
    if "voigt" in n:
        return make_multi_model(pseudo_voigt_profile), "pseudo-voigt"
    if "emg" in n or "exp" in n:
        return make_multi_model(emg_profile), "emg"
    return make_multi_model(gaussian_profile), "gaussian"
# ==== end models ====
import os
import sys
import math
from concurrent.futures import ThreadPoolExecutor
import traceback
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

# --- Missing helper utilities added ---
def _median_dx(x):
    """Robust median spacing of x (handles NaNs and unsorted input).
    Returns a positive float; falls back to 1.0 if insufficient data.
    """
    import numpy as _np
    try:
        xa = _np.asarray(x, dtype=float)
        if xa.size < 2:
            return 1.0
        # drop NaNs
        xa = xa[_np.isfinite(xa)]
        if xa.size < 2:
            return 1.0
        xa = _np.sort(xa, axis=None)
        dx = _np.diff(xa)
        dx = dx[_np.isfinite(dx)]
        if dx.size == 0:
            return 1.0
        med = float(_np.median(_np.abs(dx)))
        if not _np.isfinite(med) or med <= 0:
            # use mean positive spacing as last resort
            pos = dx[dx > 0]
            return float(pos.mean()) if pos.size else 1.0
        return med
    except Exception:
        return 1.0

def _decimate_for_plot(x, y, max_points=4000):
    """Return (x_dec, y_dec, idx) with at most max_points samples for plotting.
    Keeps endpoints and samples approximately uniformly across the domain.
    idx are integer indices into the original arrays.
    """
    import numpy as _np
    xa = _np.asarray(x, dtype=float)
    ya = _np.asarray(y, dtype=float)
    n = xa.size
    if n == 0:
        return xa, ya, _np.arange(0, dtype=int)
    if n <= int(max_points):
        return xa, ya, _np.arange(n, dtype=int)
    # stride selection
    stride = max(int(_np.floor(n / int(max_points))), 1)
    idx = _np.arange(0, n, stride, dtype=int)
    # ensure last index present
    if idx[-1] != n-1:
        idx = _np.append(idx, n-1)
    return xa[idx], ya[idx], idx
# --- End helpers ---


import numpy as np
import pandas as pd

# Fitting and peak utilities
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit

def _resolve_model_display_name(name: str) -> str:
    try:
        n = str(name).strip().lower()
    except Exception:
        return "Gaussian"
    if "voigt" in n:
        return "Pseudo-Voigt"
    if "emg" in n or "exp" in n:
        return "EMG"
    # default
    return "Gaussian"

def _fit_candidates_and_choose(x_use: np.ndarray, y_use: np.ndarray, peak_indices: List[int], maxfev: int = 50000):
    """Try Gaussian, Pseudo-Voigt, and EMG; select best by AIC (fallback to SSE)."""
    cands = []
    for name in ("gaussian", "pseudo-voigt", "emg"):
        try:
            gf = fit_global_multi_model(x_use, y_use, list(map(int, peak_indices)), model_name=name, robust=True, maxfev=maxfev)
            if gf is None or not hasattr(gf, "y_fit"):
                continue
            resid = y_use - gf.y_fit
            sse = float(np.sum(np.square(resid)))
            k = 1 + 3*len(peak_indices)
            n = max(len(y_use), 1)
            aic = n * np.log(max(sse/n, 1e-30)) + 2*k
            bic = n * np.log(max(sse/n, 1e-30)) + k * np.log(max(n, 2))
            cands.append({"name": name, "gf": gf, "SSE": sse, "AIC": aic, "BIC": bic})
        except Exception:
            continue
    if not cands:
        return None, []
    cands.sort(key=lambda d: (d.get("AIC", float("inf")), d.get("SSE", float("inf"))))
    best = cands[0]
    # tag model name for plotting/labels
    try:
        best["gf"].model_name = _resolve_model_display_name(best["name"])
    except Exception:
        pass
    ranking = [{"model": c["name"], "AIC": c["AIC"], "BIC": c["BIC"], "SSE": c["SSE"]} for c in cands]
    return best["gf"], ranking

def analyze_spectrum_auto(x: np.ndarray, y: np.ndarray, min_height_frac: float = 0.5, window_factor: float = 1.5,
                          distance: Optional[int] = None, prominence: Optional[float] = None,
                          plot: bool = False, save_plots: bool = False, outdir: Optional[str] = None,
                          file_prefix: str = "spectrum") -> Dict[str, Any]:
    """Wrapper that detects peaks, then auto-selects the best global model."""
    # First run the standard analyzer without global fit to get peaks
    base = analyze_spectrum(x=x, y=y, line_shape="Gaussian", min_height_frac=min_height_frac,
                            distance=distance, prominence=prominence, window_factor=window_factor,
                            do_global_fit=False, plot=False, save_plots=False)
    peaks_idx = list(map(int, base.get("peaks_idx", [])))
    peaks_pos = np.asarray(base.get("peaks_pos", []), dtype=float)
    # Build a union ROI around peaks for fair model comparison
    if len(peaks_idx) > 0:
        try:
            sigmas = [max(_initial_sigma_from_widths(np.asarray(x, dtype=float), np.asarray(y, dtype=float), int(i)), 1e-6) for i in peaks_idx]
            mus = [float(np.asarray(x, dtype=float)[int(i)]) for i in peaks_idx]
            lo = min(mu - window_factor * s for mu, s in zip(mus, sigmas))
            hi = max(mu + window_factor * s for mu, s in zip(mus, sigmas))
            lo = max(float(np.nanmin(x)), lo); hi = min(float(np.nanmax(x)), hi)
            mask = (np.asarray(x) >= lo) & (np.asarray(x) <= hi)
            x_glob = np.asarray(x, dtype=float)[mask]; y_glob = np.asarray(y, dtype=float)[mask]
        except Exception:
            x_glob = np.asarray(x, dtype=float); y_glob = np.asarray(y, dtype=float)
    else:
        x_glob = np.asarray(x, dtype=float); y_glob = np.asarray(y, dtype=float)
    # Try models and choose
    gf, ranking = _fit_candidates_and_choose(x_glob, y_glob, peaks_idx)
    # Merge back into the base result
    base["global_fit"] = gf
    base["line_shape_used"] = getattr(gf, "model_name", "Gaussian") if gf is not None else "Gaussian"
    base["model_ranking"] = ranking
    # Optionally save plots using existing helpers
    if save_plots and gf is not None:
        outdir_use = outdir if outdir is not None else "."
        try:
            os.makedirs(outdir_use, exist_ok=True)
        except Exception:
            outdir_use = "."
        tag = str(getattr(gf, 'model_name', 'Gaussian')).lower()
        tag = ('voigt' if 'voigt' in tag else ('emg' if 'emg' in tag or 'exp' in tag else 'gaussian'))
        main_path = f"{outdir_use}/{file_prefix}_{tag}_main_fit_annotated.png"
        res_path  = f"{outdir_use}/{file_prefix}_{tag}_residuals.png"
        zoom_path = f"{outdir_use}/{file_prefix}_zoom_main_peak.png"
        base_path = f"{outdir_use}/{file_prefix}_baseline_corrected.png"
        try:
            _save_plot_main_annotated(np.asarray(x, dtype=float), np.asarray(y, dtype=float), base.get("peaks_pos"), base.get("peaks_idx"), gf, main_path, marker_mode="data")
        except Exception:
            pass
        try:
            _save_plot_residuals(np.asarray(x, dtype=float), np.asarray(y, dtype=float), gf, res_path)
        except Exception:
            pass
        try:
            _save_plot_zoom(np.asarray(x, dtype=float), np.asarray(y, dtype=float), gf, zoom_path)
        except Exception:
            pass
        try:
            _save_plot_baseline_corrected(np.asarray(x, dtype=float), np.asarray(y, dtype=float), gf, base_path)
        except Exception:
            pass
    return base

# Tkinter GUI + Matplotlib
import tkinter as tk
from tkinter import ttk

# --- Windows DPI awareness (prevents tiny/blurred UI on HiDPI screens) ---
import sys
if sys.platform.startswith('win'):
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # System DPI aware
    except Exception:
        pass
# --------------------------------------------------------------------------



class HScrollFrame(ttk.Frame):
    """Horizontally scrollable single-row container for toolbars (cross-platform).
       This version auto-sizes the canvas height to the inner content so widgets are visible.
    """
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.canvas = tk.Canvas(self, highlightthickness=0)  # no fixed height
        self.inner = ttk.Frame(self.canvas)
        self.hbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=self.hbar.set)
        self._win = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")

        self.canvas.pack(side="top", fill="x", expand=True)
        self.hbar.pack(side="bottom", fill="x")

        def _resize(_=None):
            # Set scrollregion to inner bbox and match canvas height to inner's requested height
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            try:
                req_h = max(32, self.inner.winfo_reqheight())
                self.canvas.configure(height=req_h)
            except Exception:
                pass
        self.inner.bind("<Configure>", _resize)
        self.canvas.bind("<Configure>", _resize)

        # Cross-platform wheel/trackpad horizontal scrolling
        def _bind_wheel(widget):
            widget.bind("<Shift-MouseWheel>", lambda e: self.canvas.xview_scroll(int(-1*(e.delta/120)), "units"))
            widget.bind("<MouseWheel>",      lambda e: self.canvas.xview_scroll(int(-1*(e.delta/120)), "units"))
            widget.bind("<Button-4>",        lambda e: self.canvas.xview_scroll(-1, "units"))   # X11
            widget.bind("<Button-5>",        lambda e: self.canvas.xview_scroll( 1, "units"))   # X11
        _bind_wheel(self.canvas)
        _bind_wheel(self.inner)
@dataclass
class PeakFit:
    A: float
    mu: float
    sigma: float
    y0: float
    FWHM: float
    area: float
    A_se: Optional[float] = None
    mu_se: Optional[float] = None
    sigma_se: Optional[float] = None
    y0_se: Optional[float] = None
    R2: Optional[float] = None
    cov: Optional[np.ndarray] = None

@dataclass
class GlobalFit:
    y0: float
    params: List[PeakFit]
    R2: float
    y_fit: np.ndarray
    cov: Optional[np.ndarray] = None

def _initial_sigma_from_widths(x: np.ndarray, y: np.ndarray, peak_idx: int) -> float:
    """Estimate sigma using scipy.signal.peak_widths at ~half height."""
    try:
        widths, height, left_ips, right_ips = peak_widths(y, [peak_idx], rel_height=0.5)
        w = float(widths[0]) if len(widths) else 5.0  # index units
        dx = float(np.mean(np.diff(x))) if len(x) > 1 else 1.0
        fwhm_guess = max(w * dx, 1e-6)
        sigma = max(fwhm_guess / 2.3548200450309493, 1e-6)
        return sigma
    except Exception:
        return max((np.nanmax(x) - np.nanmin(x)) / 50.0, 1e-6)

def fit_peak_gaussian(
    x: np.ndarray,
    y: np.ndarray,
    peak_index: int,
    window_factor: float = 1.5,
    min_points: int = 7,
) -> PeakFit:
    """Fit a single Gaussian (with constant baseline) around a detected peak."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    peak_index = int(peak_index)
    if n < min_points or not (0 <= peak_index < n):
        raise ValueError("Insufficient data or invalid peak index.")

    sigma0 = _initial_sigma_from_widths(x, y, peak_index)
    fwhm0 = 2.3548200450309493 * sigma0
    half_width = window_factor * fwhm0 / 2.0
    x0 = float(x[peak_index])

    lo = x0 - half_width
    hi = x0 + half_width
    mask = (x >= lo) & (x <= hi)
    if mask.sum() < min_points:
        order = np.argsort(np.abs(x - x0))
        mask = np.zeros_like(x, dtype=bool)
        mask[order[:max(min_points, 2)]] = True

    xw = x[mask]
    yw = y[mask]

    y0 = float(np.median(yw))
    A0 = float(np.max(yw) - y0)
    mu0 = x0
    sigma0 = max(np.std(xw) / 2.0, sigma0)

    def model(xx, y0_, A_, mu_, s_):
        return gaussian_with_baseline(xx, y0_, A_, mu_, s_)

    p0 = [y0, A0, mu0, sigma0]
    sigma_floor = max(_median_dx(xw) * 0.35, 1e-6)
    lower = [-np.inf, 0.0, float(np.min(xw)), sigma_floor]
    upper = [ np.inf,  np.inf, float(np.max(xw)), float(max(np.ptp(xw), sigma_floor*50))]
    popt, pcov = curve_fit(model, xw, yw, p0=p0, bounds=(lower, upper), maxfev=20000)
    yfit = model(xw, *popt)
    ss_res = float(np.sum((yw - yfit) ** 2))
    ss_tot = float(np.sum((yw - np.mean(yw)) ** 2)) or 1e-15
    R2 = 1.0 - ss_res / ss_tot

    y0_, A_, mu_, s_ = popt
    FWHM = 2.3548200450309493 * abs(float(s_))
    area = float(A_) * abs(float(s_)) * math.sqrt(2.0 * math.pi)

    se = None
    if pcov is not None and np.all(np.isfinite(pcov)):
        try:
            se = np.sqrt(np.diag(pcov))
        except Exception:
            se = None
    A_se = float(se[1]) if isinstance(se, np.ndarray) and len(se) > 1 else None
    mu_se = float(se[2]) if isinstance(se, np.ndarray) and len(se) > 2 else None
    sigma_se = float(se[3]) if isinstance(se, np.ndarray) and len(se) > 3 else None
    y0_se = float(se[0]) if isinstance(se, np.ndarray) and len(se) > 0 else None

    return PeakFit(A=A_, mu=mu_, sigma=s_, y0=y0_, FWHM=FWHM, area=area,
                   A_se=A_se, mu_se=mu_se, sigma_se=sigma_se, y0_se=y0_se, R2=R2, cov=pcov)

from scipy.optimize import least_squares

def fit_global_multi_model(
    x: np.ndarray,
    y: np.ndarray,
    peak_indices: List[int],
    model_name: str = "gaussian",
    robust: bool = False,
    maxfev: int = 50000,
) -> Optional[GlobalFit]:
    """
    Generic global fit: baseline + sum of identical peak shapes (3 params per peak: A, mu, sigma).
    model_name in {"gaussian", "pseudo-voigt", "emg"}.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(peak_indices) == 0:
        return None

    peak_indices = list(map(int, peak_indices))
    peak_indices = sorted(peak_indices, key=lambda idx: x[idx])

    multi_fn, resolved_name = get_multipeak_model_by_name(model_name)

    # Initial guess
    y0 = float(np.median(y))
    p0 = [y0]
    for idx in peak_indices:
        idx = int(idx)
        A0 = max(float(y[idx] - y0), 1e-9)
        mu0 = float(x[idx])
        sigma0 = _initial_sigma_from_widths(x, y, idx)
        p0.extend([A0, mu0, sigma0])

    # Bounds: y0 free; A >= 0; mu within data; sigma > 0
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    n_peaks = len(peak_indices)
    lb = [-np.inf] + [0.0, xmin, 1e-9] * n_peaks
    ub = [ np.inf] + [np.inf, xmax, (xmax - xmin)] * n_peaks

    # Sanitize initial params for numerical stability
    def _sanitize_params(p):
        pp = list(p)
        if len(pp) > 0 and not np.isfinite(pp[0]):
            pp[0] = 0.0
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
        span = hi - lo if np.isfinite(hi - lo) else 1.0
        for i in range(1, len(pp), 3):
            if i < len(pp) and not np.isfinite(pp[i]):
                pp[i] = 1.0
            if i+1 < len(pp):
                if not np.isfinite(pp[i+1]):
                    pp[i+1] = lo + 0.5*span
                pp[i+1] = min(max(pp[i+1], lo - 0.1*span), hi + 0.1*span)
            if i+2 < len(pp):
                if not np.isfinite(pp[i+2]) or abs(pp[i+2]) < 1e-9:
                    pp[i+2] = max(1e-3, 0.02*span)
                else:
                    pp[i+2] = max(1e-6, abs(pp[i+2]))
        return pp
    p0 = _sanitize_params(p0)
    def residuals(p):
        yh = multi_fn(x, *p)
        if not np.all(np.isfinite(yh)):
            yh = np.nan_to_num(yh, nan=0.0, posinf=1e12, neginf=-1e12)
        r = yh - y
        if not np.all(np.isfinite(r)):
            r = np.nan_to_num(r, nan=0.0, posinf=1e12, neginf=-1e12)
        return r

    loss = 'linear'
    f_scale = 1.0
    if robust:
        loss = 'soft_l1'
        f_scale = np.std(y - np.median(y)) if len(y) > 0 else 1.0
        if not np.isfinite(f_scale) or f_scale <= 0:
            f_scale = 1.0

    res = least_squares(residuals, x0=np.asarray(p0, dtype=float),
                        bounds=(np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)),
                        max_nfev=maxfev, loss=loss, f_scale=f_scale)

    if not res.success:
        # Graceful fallback
        return None

    p_opt = res.x
    y_fit = multi_fn(x, *p_opt)
    resid = y - y_fit
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y))**2)) if len(y) > 1 else 0.0
    R2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    # Parse params into PeakFit list
    y0_fit = float(p_opt[0])
    params = []
    for i in range(1, len(p_opt), 3):
        if i + 2 >= len(p_opt):
            break
        A, mu, sg = map(float, p_opt[i:i+3])
        # Shape-dependent area + FWHM estimates (with our 3-param tie-ins)
        if resolved_name == "gaussian":
            FWHM = 2.3548200450309493 * sg
            area = A * sg * math.sqrt(2.0 * math.pi)
        elif resolved_name == "pseudo-voigt":
            FWHM = 2.3548200450309493 * sg
            gamma = FWHM / 2.0
            eta = 0.5
            area = A * (eta * math.pi * gamma + (1.0 - eta) * sg * math.sqrt(2.0 * math.pi))
        else:  # EMG
            # Use Gaussian-equivalent area approximation for simplicity
            FWHM = 2.3548200450309493 * sg
            area = A * sg * math.sqrt(2.0 * math.pi)
        params.append(PeakFit(A=A, mu=mu, sigma=sg, y0=y0_fit, FWHM=FWHM, area=area, R2=R2))

    return GlobalFit(y0=y0_fit, params=params, R2=R2, y_fit=np.asarray(y_fit, dtype=float), cov=None)

def fit_global_multi_model_robust(
    x: np.ndarray,
    y: np.ndarray,
    peak_indices: List[int],
    model_name: str = "gaussian",
    maxfev: int = 50000,
) -> Optional[GlobalFit]:
    return fit_global_multi_model(x, y, peak_indices, model_name=model_name, robust=True, maxfev=maxfev)

def fit_global_multi_gaussian_robust(
    x: np.ndarray,
    y: np.ndarray,
    peak_indices: List[int],
    maxfev: int = 50000,
) -> Optional[GlobalFit]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(peak_indices) == 0:
        return None
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    y0 = float(np.median(y))
    p0 = [y0]
    sigmas0 = []
    try:
        from scipy.signal import peak_widths
    except Exception:
        peak_widths = None
    for idx in peak_indices:
        idx = int(idx)
        A0 = max(float(y[idx] - y0), 1e-9)
        mu0 = float(x[idx])
        if peak_widths is not None:
            try:
                w_res = peak_widths(y, [idx], rel_height=0.5)
                width_pts = float(w_res[0][0])
                dx_med = _median_dx(x)
                FWHM0 = max(width_pts * dx_med, dx_med)
                s0 = max(FWHM0 / 2.3548200450309493, dx_med * 0.35)
            except Exception:
                s0 = max(np.std(y) * 0.02, _median_dx(x) * 0.35)
        else:
            s0 = max(np.std(y) * 0.02, _median_dx(x) * 0.35)
        sigmas0.append(s0)
        p0 += [A0, mu0, s0]

    xmin, xmax = float(np.min(x)), float(np.max(x))
    dxm = _median_dx(x)
    sigma_floor = max(dxm * 0.35, 1e-6)

    lower = [-np.inf]
    upper = [ np.inf]
    for s0 in sigmas0:
        lower += [0.0, xmin, sigma_floor]
        upper += [np.inf, xmax, (xmax - xmin)]

    def resid(p):
        yhat = multi_gaussian_with_baseline(x, *p)
        s = np.std(y) or 1.0
        return (yhat - y) / s

    try:
        ls = least_squares(resid, p0, bounds=(lower, upper), loss="soft_l1", f_scale=1.0, max_nfev=maxfev)
    except Exception:
        return None
    pp = ls.x
    y_fit = multi_gaussian_with_baseline(x, *pp)
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) or 1e-15
    R2 = 1.0 - ss_res / ss_tot

    y0_fit = float(pp[0])
    peaks_params: List[PeakFit] = []
    for i in range(1, len(pp), 3):
        A = float(pp[i + 0])
        mu = float(pp[i + 1])
        sigma = float(pp[i + 2])
        FWHM = 2.3548200450309493 * abs(sigma)
        area = A * abs(sigma) * math.sqrt(2.0 * math.pi)
        peaks_params.append(PeakFit(
            A=A, mu=mu, sigma=sigma, y0=y0_fit, FWHM=FWHM, area=area,
            A_se=None, mu_se=None, sigma_se=None, y0_se=None, R2=None, cov=None
        ))
    return GlobalFit(y0=y0_fit, params=peaks_params, R2=R2, y_fit=y_fit, cov=None)
def fit_global_multi_gaussian(
    x: np.ndarray,
    y: np.ndarray,
    peak_indices: List[int],
    window_expand_factor: float = 1.2,
    maxfev: int = 20000,
) -> Optional[GlobalFit]:
    """Fit the entire spectrum with a sum of Gaussians + constant baseline."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(peak_indices) == 0:
        return None

    peak_indices = list(map(int, peak_indices))
    peak_indices = sorted(peak_indices, key=lambda idx: x[idx])

    y0 = float(np.median(y))
    p0 = [y0]
    for pi in peak_indices:
        mu0 = float(x[pi])
        A0 = float(y[pi] - y0)
        s0 = _initial_sigma_from_widths(x, y, pi) * float(window_expand_factor)
        p0 += [A0, mu0, max(s0, 1e-6)]

    try:
        sigma_floor = max(_median_dx(x) * 0.35, 1e-6)
        lower = [ -np.inf ]
        upper = [  np.inf ]
        # For each peak: (A >= 0, mu within a local window, sigma >= sigma_floor)
        for idx in peak_indices:
            lo = float(max(np.min(x), x[idx] - 5 * sigma_floor))
            hi = float(min(np.max(x), x[idx] + 5 * sigma_floor))
            lower += [ 0.0, lo, sigma_floor ]
            upper += [ np.inf, hi, max(float(np.ptp(x)), sigma_floor*100) ]
        popt, pcov = curve_fit(lambda xx, *pp: multi_gaussian_with_baseline(xx, *pp),
                               x, y, p0=p0, bounds=(lower, upper), maxfev=maxfev)
        y_fit = multi_gaussian_with_baseline(x, *popt)
        ss_res = float(np.sum((y - y_fit) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2)) or 1e-15
        R2 = 1.0 - ss_res / ss_tot

        y0_fit = float(popt[0])
        peaks_params: List[PeakFit] = []
        se = None
        if pcov is not None and np.all(np.isfinite(pcov)):
            try:
                se = np.sqrt(np.diag(pcov))
            except Exception:
                se = None

        for k in range(0, (len(popt) - 1) // 3):
            i = 1 + 3*k
            A = float(popt[i + 0])
            mu = float(popt[i + 1])
            sigma = float(popt[i + 2])
            FWHM = 2.3548200450309493 * abs(sigma)
            area = A * abs(sigma) * math.sqrt(2.0 * math.pi)

            A_se = float(se[i + 0]) if isinstance(se, np.ndarray) and len(se) > (i + 0) else None
            mu_se = float(se[i + 1]) if isinstance(se, np.ndarray) and len(se) > (i + 1) else None
            sigma_se = float(se[i + 2]) if isinstance(se, np.ndarray) and len(se) > (i + 2) else None

            peaks_params.append(PeakFit(
                A=A, mu=mu, sigma=sigma, y0=y0_fit, FWHM=FWHM, area=area,
                A_se=A_se, mu_se=mu_se, sigma_se=sigma_se, y0_se=None, R2=None, cov=None
            ))

        return GlobalFit(y0=y0_fit, params=peaks_params, R2=float(R2), y_fit=y_fit, cov=pcov)
    except Exception:
        return None

def _info_criteria(y_true: np.ndarray, y_pred: np.ndarray, n_params: int) -> Tuple[float, float, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    resid = (y_true - y_pred)
    sse = float(np.sum(resid ** 2))
    if not np.isfinite(sse) or sse <= 0:
        sse = np.finfo(float).tiny
    aic = n * np.log(sse / n) + 2 * n_params
    bic = n * np.log(sse / n) + n_params * np.log(n)
    return sse, aic, bic

def _select_main_peak_from_global(g: Optional[GlobalFit], x: Optional[np.ndarray]=None, y: Optional[np.ndarray]=None) -> Optional[PeakFit]:
    if g is None or not hasattr(g, "params") or len(g.params) == 0:
        return None
    # 1) If raw data available: choose the fitted peak whose μ is closest to the highest data point
    if x is not None and y is not None and len(x) == len(y) and len(x) > 0:
        try:
            x_argmax = float(x[int(np.nanargmax(y))])
            return min(g.params, key=lambda pf: abs(pf.mu - x_argmax))
        except Exception:
            pass
    # 2) Fallback: choose the peak closest to the maximum of the model curve y_fit
    if hasattr(g, "y_fit") and g.y_fit is not None and isinstance(g.y_fit, np.ndarray) and x is not None and len(x) == len(g.y_fit):
        try:
            xfit_argmax = float(x[int(np.nanargmax(g.y_fit))])
            return min(g.params, key=lambda pf: abs(pf.mu - xfit_argmax))
        except Exception:
            pass
    # 3) Last resort: largest fitted amplitude above baseline
    return max(g.params, key=lambda pf: float(getattr(pf, "A", 0.0)))

def _save_plot_main_annotated(x: np.ndarray, y: np.ndarray, peaks_pos: np.ndarray,
                              peaks_idx: np.ndarray, global_fit: Optional[GlobalFit],
                              outpath: str, marker_mode: str = "data") -> None:
    """
    marker_mode: "data" (red dot at nearest data point to μ) or "fit" (dot at fitted y at μ).
    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(x, y, label="data")
    red_x = None
    red_y = None
    if global_fit is not None and hasattr(global_fit, "y_fit"):
        plt.plot(x, global_fit.y_fit, label=f"{_resolve_model_display_name(getattr(global_fit, 'model_name', 'Gaussian'))} fit (R2={getattr(global_fit, 'R2', float('nan')):.3f})")
        main = _select_main_peak_from_global(global_fit, x, y)
        if main is not None:
            mu = float(main.mu)
            sigma = abs(float(main.sigma))
            fwhm = 2.3548200450309493 * sigma
            plt.axvline(mu, linestyle="--", label=f"μ={mu:.6f} nm")
            x1, x2 = mu - fwhm/2.0, mu + fwhm/2.0
            plt.fill_between([x1, x2], [np.min(y)], [np.max(y)], alpha=0.1, label=f"FWHM={fwhm:.6f} nm")
            txt = f"μ={mu:.6f} nm\nFWHM={fwhm:.6f} nm\nA={main.A:.3g}"
            try:
                plt.gca().text(mu, float(np.max(y))*0.85, txt, ha="center", va="top")
            except Exception:
                pass
            if marker_mode == "fit":
                red_x = mu
                red_y = float(global_fit.y0 + sum(gaussian(mu, pf.A, pf.mu, pf.sigma) for pf in getattr(global_fit, "params", [])))
            else:
                idx = int(np.nanargmin(np.abs(x - mu)))
                red_x = float(x[idx])
                red_y = float(y[idx])
    try:
        if peaks_pos is not None and peaks_idx is not None and len(peaks_pos):
            plt.plot(np.asarray(peaks_pos), y[np.asarray(peaks_idx)], marker="o", linestyle="None", label="detected peaks")
    except Exception:
        pass
    if red_x is not None and red_y is not None:
        plt.plot([red_x], [red_y], marker="o", color="red")
    plt.xlabel("wavelength_nm")
    plt.ylabel("voltage_V")
    plt.title(f"Spectrum with {_resolve_model_display_name(getattr(global_fit, 'model_name', 'Gaussian'))} fit and annotations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')

    plt.close()

    plt.close()

    plt.close()

def _save_plot_main(x: np.ndarray, y: np.ndarray, global_fit: Optional[GlobalFit], outpath: str,
                    peaks_idx: Optional[Sequence[int]] = None, peaks_pos: Optional[Sequence[float]] = None,
                    marker_nearest: bool = True) -> None:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(x, y, label="data")
    if global_fit is not None and hasattr(global_fit, "y_fit"):
        yfit = np.asarray(global_fit.y_fit, dtype=float)
        if len(yfit) == len(y):
            plt.plot(x, yfit, label=f"{_resolve_model_display_name(getattr(global_fit, 'model_name','Gaussian'))} fit", linewidth=2.0)
        else:
            xf = np.asarray(getattr(global_fit, "x_fit", x), dtype=float)
            plt.plot(xf[:len(yfit)], yfit, label=f"{_resolve_model_display_name(getattr(global_fit, 'model_name','Gaussian'))} fit", linewidth=2.0)
    # Mark detected peaks if provided
    try:
        if peaks_pos is not None and peaks_idx is not None and len(peaks_pos):
            plt.plot(np.asarray(peaks_pos), y[np.asarray(peaks_idx)], marker="o", linestyle="None", label="detected peaks")
    except Exception:
        pass
    # Red dot at main peak
    red_x = red_y = None
    try:
        if global_fit is not None and hasattr(global_fit, "params") and len(getattr(global_fit, "params", [])):
            main = max(global_fit.params, key=lambda p: getattr(p, "A", 0.0))
            mu = float(getattr(main, "mu", float("nan")))
            if np.isfinite(mu):
                j = int(np.nanargmin(np.abs(x - mu)))
                red_x = float(x[j])
                red_y = float(y[j]) if marker_nearest or not hasattr(global_fit, "y_fit") or len(global_fit.y_fit)!=len(x) else float(global_fit.y_fit[j])
        else:
            j = int(np.nanargmax(y)) if len(y) else None
            if j is not None:
                red_x = float(x[j]); red_y = float(y[j])
    except Exception:
        pass
    if red_x is not None and red_y is not None:
        plt.plot([red_x], [red_y], marker="o", color="red")
    plt.xlabel("wavelength_nm"); plt.ylabel("voltage_V")
    try:
        plt.minorticks_on()
    except Exception:
        pass
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(); plt.tight_layout(); plt.savefig(outpath, dpi=300, bbox_inches='tight'); plt.close()
def _save_plot_residuals(x: np.ndarray, y: np.ndarray, global_fit: Optional[GlobalFit], outpath: str) -> None:
    if global_fit is None or not hasattr(global_fit, "y_fit"):
        return
    import matplotlib.pyplot as plt
    resid = y - global_fit.y_fit
    plt.figure()
    plt.plot(x, resid, marker=".", linestyle="-")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("wavelength_nm")
    plt.ylabel("residual (data - fit)")
    plt.title("Residuals vs wavelength")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')

    plt.close()

    plt.close()

    plt.close()

def _save_plot_zoom(x: np.ndarray, y: np.ndarray, global_fit: Optional[GlobalFit], outpath: str) -> None:
    if global_fit is None or not hasattr(global_fit, "params") or len(global_fit.params) == 0:
        return
    import matplotlib.pyplot as plt
    main = _select_main_peak_from_global(global_fit, x, y)
    if main is None:
        return
    mu = float(main.mu)
    sigma = abs(float(main.sigma))
    lo, hi = mu - 3*sigma, mu + 3*sigma
    mask = (x >= lo) & (x <= hi)
    plt.figure()
    plt.plot(x[mask], y[mask], label="data (zoom)")
    if hasattr(global_fit, "y_fit"):
        plt.plot(x[mask], global_fit.y_fit[mask], label=f"{_resolve_model_display_name(getattr(global_fit, 'model_name', 'Gaussian'))} fit (zoom)")
    plt.axvline(mu, linestyle="--", label="μ")
    plt.xlabel("wavelength_nm")
    plt.ylabel("voltage_V")
    plt.title("Zoomed view around main peak (±3σ)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')

    plt.close()

    plt.close()

    plt.close()

def _save_plot_baseline_corrected(x: np.ndarray, y: np.ndarray, global_fit: Optional[GlobalFit], outpath: str) -> None:
    if global_fit is None or not hasattr(global_fit, "params") or len(global_fit.params) == 0:
        return
    import matplotlib.pyplot as plt
    main = _select_main_peak_from_global(global_fit, x, y)
    if main is None:
        return
    mu = float(main.mu)
    sigma = abs(float(main.sigma))
    y0 = float(getattr(global_fit, "y0", 0.0))
    y_corr = y - y0
    yfit_corr = (global_fit.y_fit - y0) if hasattr(global_fit, "y_fit") else None
    lo, hi = mu - 3*sigma, mu + 3*sigma
    mask = (x >= lo) & (x <= hi)
    plt.figure()
    plt.plot(x, y_corr, label="data (baseline-corrected)")
    if yfit_corr is not None:
        plt.plot(x, yfit_corr, label=f"{_resolve_model_display_name(getattr(global_fit, 'model_name', 'Gaussian'))} fit (baseline-corrected)")
        try:
            plt.fill_between(x[mask], 0, yfit_corr[mask], alpha=0.2, label="peak area (±3σ)")
        except Exception:
            pass
    plt.xlabel("wavelength_nm")
    plt.ylabel("voltage_V (baseline-corrected)")
    plt.title("Baseline-corrected spectrum & peak area")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')

    plt.close()

    plt.close()

    plt.close()

def analyze_spectrum(
    x: np.ndarray,
    y: np.ndarray,
    pre_baseline: bool = True,
    line_shape: str = "Gaussian",
    min_height_frac: float = 0.5,
    distance: Optional[int] = None,
    prominence: Optional[float] = None,
    max_peaks: Optional[int] = None,
    window_factor: float = 1.5,
    do_global_fit: bool = True,
    plot: bool = False,
    save_plots: bool = False,
    outdir: Optional[str] = None,
    file_prefix: str = "spectrum",
) -> Dict[str, Any]:
    # ##__SORT_CHECK__ ensure x is ascending
    try:
        order = np.argsort(x)
        if not np.all(order == np.arange(len(x))):
            x = x[order]
            y = y[order]
    except Exception:
        pass
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    # Drop NaNs/Infs and keep ascending x
    msk = np.isfinite(x) & np.isfinite(y)
    if not np.all(msk):
        x = x[msk]; y = y[msk]
    y_for_detect = y
    if pre_baseline:
        try:
            y_for_detect = y - _asls_baseline(y)
        except Exception:
            y_for_detect = y
    y_max = np.nanmax(y_for_detect) if np.size(y_for_detect) else 0.0
    height = float(min_height_frac) * (y_max if np.isfinite(y_max) else 1.0)
    kwargs = dict(height=height)
    if distance is not None:
        kwargs["distance"] = int(distance)
    if prominence is not None:
        kwargs["prominence"] = float(prominence)

        # ##__PROM__ adaptive peak params if not supplied
    if 'prominence' not in kwargs or kwargs.get('prominence') in (None, 0):
        med = float(np.median(y)) if len(y) else 0.0
        mad = float(np.median(np.abs(y - med))) if len(y) else 0.0
        kwargs['prominence'] = max(3.0 * mad, 0.0)
    if 'distance' not in kwargs or kwargs.get('distance') in (None, 0):
        try:
            dx = float(np.median(np.diff(x)))
            span = float(np.ptp(x))
            n = max(int(round(span / max(dx, 1e-12))), 1)
            kwargs['distance'] = max(n // 200, 3)
        except Exception:
            kwargs['distance'] = 3
    raw_peaks, props = find_peaks(y, **kwargs)
    heights = props.get("peak_heights", np.array([]))

    order = np.argsort(heights)[::-1] if len(heights) else np.arange(len(raw_peaks))
    if max_peaks is not None and len(order) > max_peaks:
        order = order[:max_peaks]

    peaks_idx = raw_peaks[order]
    peaks_pos = x[peaks_idx]
    peaks_height = y[peaks_idx]

    fitted: List[PeakFit] = []
    for idx in peaks_idx:
        try:
            pf = fit_peak_gaussian(x, y, int(idx), window_factor=window_factor)
            fitted.append(pf)
        except Exception:
            continue

    global_fit = None
    if do_global_fit and len(peaks_idx) > 0:
        order_x = np.argsort(peaks_pos)
        ordered_indices = peaks_idx[order_x]
        global_fit = (
        fit_global_multi_gaussian_robust(x, y, list(map(int, ordered_indices))) if (str(line_shape).strip().lower() in ("gaussian","gauss"))
        else fit_global_multi_model_robust(x, y, list(map(int, ordered_indices)), model_name=str(line_shape))
    ) or (
        fit_global_multi_gaussian(x, y, list(map(int, ordered_indices))) if (str(line_shape).strip().lower() in ("gaussian","gauss"))
        else fit_global_multi_model(x, y, list(map(int, ordered_indices)), model_name=str(line_shape))
    )

    
    # tag the model name for plotting/labels
    try:
        global_fit.model_name = _resolve_model_display_name(line_shape)
    except Exception:
        pass

    model_selection: List[Dict[str, float]] = []
    if len(peaks_idx) > 0:
        order_h = np.argsort(peaks_height)[::-1]
        top_idx = peaks_idx[order_h]
        kmax = min(3, len(top_idx))
        for k in range(1, kmax+1):
            try:
                sel = list(map(int, top_idx[:k]))
                gf_k = (fit_global_multi_gaussian(x, y, sel) if str(line_shape).strip().lower() in ("gaussian","gauss") else fit_global_multi_model(x, y, sel, model_name=str(line_shape)))
                if gf_k is not None and hasattr(gf_k, "y_fit"):
                    p = 1 + 3*k
                    sse, aic, bic = _info_criteria(y, gf_k.y_fit, p)
                    model_selection.append({"k": k, "params": p, "SSE": sse, "AIC": aic, "BIC": bic, "R2": getattr(gf_k, "R2", np.nan)})
            except Exception:
                continue

    # Tag model name for labels/titles
    try:
        if global_fit is not None and not hasattr(global_fit, "model_name"):
            global_fit.model_name = _resolve_model_display_name(line_shape)
    except Exception:
        pass

    files = {}
    if save_plots:
        outdir_use = outdir if outdir is not None else "."
        try:
            os.makedirs(outdir_use, exist_ok=True)
        except Exception:
            outdir_use = "."
        main_path = f"{outdir_use}/{file_prefix}_main_fit_annotated.png"
        res_path = f"{outdir_use}/{file_prefix}_residuals.png"
        zoom_path = f"{outdir_use}/{file_prefix}_zoom_main_peak.png"
        base_path = f"{outdir_use}/{file_prefix}_baseline_corrected.png"
        try:
            _save_plot_main_annotated(x, y, peaks_pos, peaks_idx, global_fit, main_path, marker_mode="data")
            files["plot_main"] = main_path
        except Exception:
            pass
        try:
            _save_plot_residuals(x, y, global_fit, res_path)
            files["plot_residuals"] = res_path
        except Exception:
            pass
        try:
            _save_plot_zoom(x, y, global_fit, zoom_path)
            files["plot_zoom"] = zoom_path
        except Exception:
            pass
        try:
            _save_plot_baseline_corrected(x, y, global_fit, base_path)
            files["plot_baseline_corrected"] = base_path
        except Exception:
            pass

    return {
        "line_shape_used": str(line_shape),
        "peaks_idx": peaks_idx,
        "peaks_pos": peaks_pos,
        "peaks_height": peaks_height,
        "peaks_fitted": fitted,
        "global_fit": global_fit,
        "model_selection": model_selection,
        "files": files,
    }

# ---------------------------
# Tkinter GUI
# ---------------------------

APP_TITLE = "Spectrum Fit – v13 (SaveFix)"
DEFAULT_MIN_HEIGHT_FRAC = 0.15
DEFAULT_WINDOW_FACTOR = 1.5

def parse_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    if df.columns[0].lower().startswith("index"):
        df = df.iloc[:, 1:]
    return df

def pick_xy_columns(df: pd.DataFrame):
    cols = list(df.columns)
    x_candidates = [c for c in cols if "wave" in c.lower() or "nm" in c.lower() or c.lower() in ("x","wavelength","lambda")]
    y_candidates = [c for c in cols if "volt" in c.lower() or "int" in c.lower() or "pmt" in c.lower() or c.lower() in ("y","intensity")]
    x_col = x_candidates[0] if x_candidates else None
    y_col = y_candidates[0] if y_candidates else None
    if x_col is None or y_col is None:
        numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric) >= 2:
            x_col, y_col = numeric[:2]
    if x_col is None or y_col is None:
        raise ValueError("Could not determine x/y columns. Please rename columns to include e.g. 'wavelength_nm' and 'voltage_V'.")
    return x_col, y_col
        # Menubar with cross-platform shortcuts

# --- Injected helper: simple ZoomManager for drag-zoom/back/reset ---
class ZoomManager:
    def __init__(self, ax, canvas, on_draw=None):
        self.ax = ax
        self.canvas = canvas
        self._press = None
        self.history = []
        self.active = True
        self.cid_press = canvas.mpl_connect('button_press_event', self._on_press)
        self.cid_release = canvas.mpl_connect('button_release_event', self._on_release)
        self.cid_scroll = canvas.mpl_connect('scroll_event', self._on_scroll)

    def _on_press(self, event):
        if not self.active or event.inaxes != self.ax:
            return
        # Save previous limits for back()
        self.history.append((self.ax.get_xlim(), self.ax.get_ylim()))
        self._press = (event.xdata, event.ydata)

    def _on_release(self, event):
        if not self.active or self._press is None or event.inaxes != self.ax:
            self._press = None
            return
        x0, y0 = self._press
        x1, y1 = event.xdata, event.ydata
        self._press = None
        if x0 is None or x1 is None or y0 is None or y1 is None:
            return
        if abs(x1 - x0) < 1e-12 or abs(y1 - y0) < 1e-12:
            return
        xmin, xmax = sorted([x0, x1])
        ymin, ymax = sorted([y0, y1])
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.on_draw(idle=True)

    def back(self):
        if self.history:
            xlim, ylim = self.history.pop()
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
            self.on_draw(idle=True)

    def reset_to_full(self):
        # Autoscale to show all data
        try:
            self.ax.relim()
            self.ax.autoscale_view()
        except Exception:
            pass
        self.on_draw(idle=True)

    def toggle(self):
        self.active = not self.active
# --- End ZoomManager ---
class App:
    def on_save_alt_all(self):
        """Save alternate-fit plots and fitted CSVs for k=1..K in one click."""
        try:
            if getattr(self, "df", None) is None or getattr(self, "result", None) is None:
                try:
                    from tkinter import messagebox
                    messagebox.showwarning("Nothing to save", "Load data and run a fit first.")
                except Exception:
                    pass
                return
            # determine K from combobox
            try:
                K = int(str(self.k_var.get()).strip())
            except Exception:
                K = 2
            from datetime import datetime
            import os
            x_col, y_col = pick_xy_columns(self.df)
            x = self.df[x_col].to_numpy(dtype=float)
            y = self.df[y_col].to_numpy(dtype=float)
            outdir = self._ensure_output_dir()
            saved = []
            for ki in range(1, max(1, K)+1):
                prefix = f"altk{ki}_" + datetime.now().strftime('%Y%m%d_%H%M%S')
                # Save the 4 plots
                for kind in ("main", "residuals", "zoom", "baseline_corrected"):
                    try:
                        p = None
                        try:
                            # Preferred path: dedicated saver
                            p = self._save_kind(kind, prefix, x, y, self.result)
                        except Exception:
                            # Back-compat: use generic kind saver
                            self.save_plot_kind(kind, alt_k=ki)
                        if p:
                            saved.append(p)
                    except Exception:
                        pass
                # Try to write fitted curve CSVs (y_fit), alt-aware if possible
                try:
                    import numpy as np, pandas as pd
                    y_fit = None
                    # Typical sources: result['alt_fits'][ki-1]['y_fit'] OR a predictor
                    alt = None
                    try:
                        alt_fits = self.result.get('alt_fits') if isinstance(self.result, dict) else None
                        if isinstance(alt_fits, (list, tuple)) and 0 <= ki-1 < len(alt_fits):
                            alt = alt_fits[ki-1]
                    except Exception:
                        alt = None
                    if alt is not None:
                        if isinstance(alt, dict) and 'y_fit' in alt:
                            y_fit = np.asarray(alt['y_fit'], dtype=float)
                        elif hasattr(alt, 'predict'):
                            try:
                                y_fit = alt.predict(x)
                            except Exception:
                                pass
                    # Fallback to global fit if no alt-specific available
                    if y_fit is None:
                        g = self.result.get('global_fit') if isinstance(self.result, dict) else None
                        if g is not None and hasattr(g, 'predict'):
                            try:
                                y_fit = g.predict(x)
                            except Exception:
                                pass
                    if y_fit is not None and y_fit.shape == x.shape:
                        df_fit = pd.DataFrame({"x": x, "y_fit": y_fit})
                        p_fit = os.path.join(outdir, f"{prefix}_fitted_curve.csv")
                        df_fit.to_csv(p_fit, index=False); saved.append(p_fit)
                        df_comb = pd.DataFrame({"x": x, "y": y, "y_fit": y_fit, "residual": y - y_fit})
                        p_comb = os.path.join(outdir, f"{prefix}_data_and_fit.csv")
                        df_comb.to_csv(p_comb, index=False); saved.append(p_comb)
                except Exception:
                    pass
            if saved:
                try:
                    from tkinter import messagebox
                    messagebox.showinfo("Saved", "Saved files:\n" + "\n".join(saved))
                except Exception:
                    pass
                self._safe_status(f"Saved {len(saved)} files to {outdir}")
        except Exception:
            pass
    def on_save_alt_fit(self):
        """Export alternate-fit-only plots based on K (self.k_var)."""
        try:
            if getattr(self, "df", None) is None or getattr(self, "result", None) is None:
                try:
                    from tkinter import messagebox
                    messagebox.showwarning("Nothing to save", "Load data and run a fit first.")
                except Exception:
                    pass
                return
            try:
                k = int(str(self.k_var.get()).strip())
            except Exception:
                k = 2
            kinds = ("main", "residuals", "zoom", "baseline_corrected")
            for ki in range(1, max(1, k)+1):
                for kind in kinds:
                    try:
                        try:
                            self.save_plot_kind(kind, alt_k=ki)
                        except TypeError:
                            self.save_plot_kind(kind)
                    except Exception:
                        pass
            try:
                from tkinter import messagebox
                messagebox.showinfo("Saved", "Alternate-fit plots exported.")
            except Exception:
                pass
        except Exception:
            pass
    def _update_save_buttons(self, have_data: bool = False, have_fit: bool = False):
        """Enable/disable Save buttons based on availability of data/fit."""
        try:
            b = getattr(self, 'btn_save_all', None)
            if b is not None:
                b.configure(state=('normal' if have_fit else 'disabled'))
        except Exception:
            pass

    def _ensure_output_dir(self) -> str:
        dvar = getattr(self, 'output_dir_var', None)
        outdir = dvar.get() if dvar is not None else getattr(self, 'output_dir', os.getcwd())
        outdir = outdir or os.getcwd()
        try:
            os.makedirs(outdir, exist_ok=True)
        except Exception:
            outdir = os.getcwd()
        return outdir



    
    def _force_canvas_fullsize(self, event=None):
        """Debounced resize: schedule a figure resize/draw once after window settles."""
        try:
            if getattr(self, 'plot_frame', None) is None or getattr(self, 'canvas', None) is None:
                return
            try:
                self._pending_size = (int(self.plot_frame.winfo_width()), int(self.plot_frame.winfo_height()))
            except Exception:
                self._pending_size = None
            try:
                if getattr(self, '_resize_job', None):
                    self.root.after_cancel(self._resize_job)
            except Exception:
                pass
            try:
                self._resize_job = self.root.after(120, self._resize_figure_to_frame)
            except Exception:
                self._resize_figure_to_frame()
        except Exception:
            pass


    def _resize_figure_to_frame(self, event=None):
        """Resize matplotlib Figure to match plot_frame pixel size."""
        try:
            if not hasattr(self, 'plot_frame') or self.plot_frame is None:
                return
            w = int(self.plot_frame.winfo_width())
            h = int(self.plot_frame.winfo_height())
            if w <= 10 or h <= 10 or self.fig is None:
                return
            dpi = float(self.fig.get_dpi() or 100.0)
            self.fig.set_size_inches(w / dpi, h / dpi, forward=True)
            if getattr(self, 'canvas', None):
                try:
                    self.canvas.draw_idle()
                except Exception:
                    pass
        except Exception:
            pass

    def _safe_canvas_draw(self, idle=False):
        """Schedule a draw_idle once; avoid immediate draw/update storms."""
        try:
            if getattr(self, 'canvas', None) is None:
                return
            try:
                if getattr(self, '_draw_job', None):
                    self.root.after_cancel(self._draw_job)
            except Exception:
                pass
            def _do():
                try:
                    self._draw_job = None
                    if hasattr(self.canvas, 'draw_idle'):
                        self.canvas.draw_idle()
                    else:
                        self.canvas.draw()
                except Exception:
                    pass
            try:
                self._draw_job = self.root.after(60, _do)
            except Exception:
                _do()
        except Exception:
            pass

    
    def on_draw(self, idle=False):
        """Compatibility proxy used by ZoomManager and internal after-calls."""
        return self._safe_canvas_draw(idle=idle)

    def _safe_status(self, msg):
            """Safe status setter — updates status bar and never raises."""
            try:
                # Update Tk status label if available
                if hasattr(self, "status") and self.status is not None:
                    try:
                        self.status.set(str(msg))
                    except Exception:
                        pass
                # Keep UI responsive
                if hasattr(self, "root") and self.root is not None:
                    try:
                        self.root.update_idletasks()
                    except Exception:
                        pass
            except Exception:
                pass
            # Always echo to console for debugging
            try:
                print(str(msg))
            except Exception:
                pass
    

    def _ensure_canvas_visible(self):
        w = self.canvas.get_tk_widget()
        try:
            w.update_idletasks()
        except Exception:
            pass
        if w.winfo_height() < 80:
            try:
                w.configure(height=500)
            except Exception:
                pass
        try:
            self.on_draw(idle=False)
        except Exception:
            pass
        self._ensure_vis_after_id = self.root.after(200, lambda: self._ensure_canvas_visible() if getattr(self, '_alive', True) else None)
        # Preflight: ensure interactive TkAgg is available
        if NON_INTERACTIVE:
            msg = (
                'This app requires Tkinter + Matplotlib TkAgg.\n'
                f'Boot error: {BOOT_ERROR}\n\n'
                'Fixes by OS:\n'
                '- Windows: use Python from python.org (includes Tk).\n'
                '- macOS: prefer python.org Python. If using Homebrew: `brew install python-tk@3`.\n'
                '- Ubuntu/Debian: `sudo apt-get install python3-tk`.\n'
            )
            print(msg)
            raise SystemExit(1)

    def __init__(self, root):
        self.root = root

        # Liveness flag and clean shutdown for scheduled callbacks
        self._alive = True
        try:
            self.root.protocol("WM_DELETE_WINDOW", self._on_close_safe)
        except Exception:
            pass


        # ---- Early container creation: must exist before any try/except touches them ----
        try:
            self.main_frame
        except AttributeError:
            self.main_frame = ttk.Frame(root)
        try:
            self.plot_frame
        except AttributeError:
            self.plot_frame = ttk.Frame(self.main_frame)
        try:
            self.bottom_bar
        except AttributeError:
            self.bottom_bar = ttk.Frame(self.main_frame)
        # Idempotent layout wiring
        try:
            self.main_frame.grid(row=1, column=0, sticky='nsew')
            self.main_frame.grid_columnconfigure(0, weight=1)
            self.main_frame.grid_rowconfigure(0, weight=1)
            root.rowconfigure(1, weight=1)
            root.columnconfigure(0, weight=1)
            self.plot_frame.grid(row=0, column=0, sticky='nsew')
            self.bottom_bar.grid(row=1, column=0, sticky='ew')
        except Exception:
            pass

        # Bottom Save bar (always visible across width)
        try:
            if not hasattr(self, 'bottom_bar') or self.bottom_bar is None:
                self.bottom_bar = ttk.Frame(self.main_frame)
                self.bottom_bar.grid(row=1, column=0, sticky='ew')
            # Clear previous children if any
            for w in list(self.bottom_bar.children.values()):
                w.destroy()
            # Single Save ALL button
            self.btn_save_all = ttk.Button(self.bottom_bar, text='Save ALL',
                                       command=getattr(self, 'on_save_results', lambda: None),
                                       state=tk.DISABLED)
            self.btn_save_all.pack(side=tk.LEFT, padx=12, pady=4)
            # Stretch spacer so button stays left but bar fills width
            ttk.Label(self.bottom_bar, text='').pack(side=tk.LEFT, expand=True, fill=tk.X)
        except Exception:
            pass

        # ---- Bind save handlers whether defined at top-level or nested ---- whether defined at top-level or nested ----
        try:
            from types import MethodType as _MethodType
            self.on_save_results = _MethodType(on_save_results, self)  # nested or top-level
        except Exception:
            pass
        try:
            from types import MethodType as _MethodType
            self._save_kind = _MethodType(_save_kind, self)  # nested or top-level
        except Exception:
            pass
        self.app = self  # compatibility alias for code that expects `app.app`
        self.master = root  # Tk-style alias some code expects
        # Initialize plotting attributes early
        self.fig = None
        self.ax = None
        self.canvas = None
        try:
            from matplotlib.figure import Figure as _EarlyFigure
            self.fig = _EarlyFigure(figsize=(1, 1), dpi=72)
            self.ax = self.fig.add_subplot(111)
        except Exception:
            pass

        # Ensure plotting attributes exist early to avoid AttributeError during early callbacks
        try:
            # Create a minimal placeholder Figure/Axes; replaced later by the real plot
            from matplotlib.figure import Figure as _EarlyFigure
            self.fig = _EarlyFigure(figsize=(1, 1), dpi=72)
            self.ax = self.fig.add_subplot(111)
        except Exception:
            # Fallback placeholders
            self.fig = None
            self.ax = None

        self.status = tk.StringVar(value="Ready.")
        try:
            self.root.minsize(1024, 700)
        except Exception:
            pass
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.root.title(APP_TITLE)
        root.geometry("1500x850")

        # --- fixed indentation block ---
        self.csv_path: Optional[str] = None
        self.df: Optional[pd.DataFrame] = None
        self.result: Optional[Dict[str,Any]] = None
        self.output_dir: str = os.getcwd()

        # Top controls
        self.toolbar = HScrollFrame(root)
        try:
            self.toolbar.grid(row=0, column=0, sticky='ew')
            root.columnconfigure(0, weight=1)
        except Exception:
            pass
        # Visible status bar at the bottom
        try:
            self.status_label = ttk.Label(self.bottom_bar if hasattr(self, 'bottom_bar') else root, textvariable=self.status, anchor='w')
            try:
                # Prefer grid inside bottom_bar; fallback to root.pack only if bottom_bar missing
                if hasattr(self, 'bottom_bar'):
                    self.status_label.grid(row=0, column=0, sticky='ew', padx=8, pady=4)
                    try: self.bottom_bar.columnconfigure(0, weight=1)
                    except Exception: pass
                else:
                    self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
            except Exception:
                pass
        except Exception:
            pass
        top = self.toolbar.inner

        self.path_label = ttk.Label(top, text="Drop a CSV here" if DND_AVAILABLE else "No file loaded")
        self.path_label.pack(side=tk.LEFT, padx=(0,10))

        # Optional drag-and-drop: register drop target on label and root
        try:
            if _have_dnd():
                self.path_label.drop_target_register(DND_FILES)
                self.path_label.dnd_bind('<<Drop>>', self.on_drop_files)
                self.root.drop_target_register(DND_FILES)
                self.root.dnd_bind('<<Drop>>', self.on_drop_files)
        except Exception:
            pass

        ttk.Button(top, text="Load CSV…", command=self.on_load_csv).pack(side=tk.LEFT)

        # Run button placed early to keep visible on smaller screens

        ttk.Label(top, text="min_height_frac:").pack(side=tk.LEFT, padx=(20,4))
        self.min_height_var = tk.StringVar(value=str(DEFAULT_MIN_HEIGHT_FRAC))
        ttk.Entry(top, textvariable=self.min_height_var, width=6).pack(side=tk.LEFT)

        ttk.Label(top, text="window_factor:").pack(side=tk.LEFT, padx=(12,4))
        self.window_factor_var = tk.StringVar(value=str(DEFAULT_WINDOW_FACTOR))
        ttk.Entry(top, textvariable=self.window_factor_var, width=6).pack(side=tk.LEFT)
        ttk.Label(top, text="prom:").pack(side=tk.LEFT, padx=(12,4))
        self.prominence_var = tk.StringVar(value="")
        ttk.Entry(top, textvariable=self.prominence_var, width=6).pack(side=tk.LEFT)
        ttk.Label(top, text="dist:").pack(side=tk.LEFT, padx=(12,4))
        self.distance_var = tk.StringVar(value="")
        ttk.Entry(top, textvariable=self.distance_var, width=6).pack(side=tk.LEFT)

        self.global_fit_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Global fit", variable=self.global_fit_var).pack(side=tk.LEFT, padx=(12,4))

        ttk.Label(top, text="Profile:").pack(side=tk.LEFT, padx=(12,4))
        self.lineshape_var = tk.StringVar(value="Gaussian")
        self.lineshape_cb = ttk.Combobox(top, textvariable=self.lineshape_var, width=14, state="readonly")
        self.lineshape_cb["values"] = ("Auto", "Gaussian", "Pseudo-Voigt", "EMG")
        self.lineshape_cb.current(0)
        self.lineshape_cb.pack(side=tk.LEFT, padx=(2,8))

        # Marker mode toggle
        self.marker_nearest_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Red dot = nearest data point", variable=self.marker_nearest_var).pack(side=tk.LEFT, padx=(12,4))
        self.grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Grid", variable=self.grid_var).pack(side=tk.LEFT, padx=(8,4))
        # Toggle to show highest peak marker/label
        self.show_highest_peak = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Show highest peak", variable=self.show_highest_peak,
                        command=lambda: (self._update_highest_peak_annotation(), self.canvas.draw_idle())).pack(side=tk.LEFT, padx=(8,4))

        self.btn_run = ttk.Button(top, text="Run Fit", command=self.on_run_fit, state=tk.DISABLED)
        self.btn_run.pack(side=tk.LEFT, padx=(20,0))

        ttk.Label(top, text="Output:").pack(side=tk.LEFT, padx=(16,4))
        self.output_dir_var = tk.StringVar(value=self.output_dir)
        ttk.Entry(top, textvariable=self.output_dir_var, width=35).pack(side=tk.LEFT)
        ttk.Button(top, text="Choose…", command=self.on_choose_output_dir).pack(side=tk.LEFT, padx=(4,0))

        # Save buttons row (inside scrollable toolbar)
        save_row = ttk.Frame(self.toolbar.inner, padding=(8,0,8,8))
        save_row.pack(side=tk.TOP, fill=tk.X)
        # Ensure the scroll-canvas height expands to show BOTH rows on macOS
        try:
            self.toolbar.inner.update_idletasks()
            self.toolbar.canvas.configure(height=self.toolbar.inner.winfo_reqheight())
            self.toolbar.canvas.configure(scrollregion=self.toolbar.canvas.bbox('all'))
        except Exception:
            pass





        # Initialize Save buttons disabled
        self._update_save_buttons(False, False)


        def _setup_menubar(self):
            import sys
            accel = "Command" if sys.platform == "darwin" else "Control"
            m = tk.Menu(self.root)
    
            filem = tk.Menu(m, tearoff=0)
            filem.add_command(label=f"Open…\t{accel}+O", command=getattr(self, "on_open", getattr(self, "on_load_csv", lambda: None)))
            filem.add_separator()
            filem.add_command(label=f"Save ALL\t{accel}+S", command=getattr(self, "on_save_results", lambda: None))
            filem.add_command(label="Save Annotated", command=lambda: self.save_plot_kind("main"))
            filem.add_command(label="Save Residuals", command=lambda: self.save_plot_kind("residuals"))
            filem.add_command(label="Save Zoom", command=lambda: self.save_plot_kind("zoom"))
            filem.add_command(label="Save Baseline Corrected", command=lambda: self.save_plot_kind("baseline_corrected"))
            filem.add_separator()
            filem.add_command(label="Save Alt Fit Plots", command=getattr(self, "on_save_alt_fit", lambda: None))
            filem.add_separator()
            filem.add_command(label="Exit", command=self.root.quit)
            m.add_cascade(label="File", menu=filem)
    
            fitm = tk.Menu(m, tearoff=0)
            fitm.add_command(label=f"Run Fit\t{accel}+R", command=getattr(self, "on_run_fit", lambda: None))
            m.add_cascade(label="Fit", menu=fitm)
    
            viewm = tk.Menu(m, tearoff=0)
            viewm.add_checkbutton(label="Grid", onvalue=True, offvalue=False,
                                  variable=getattr(self, "grid_var", tk.BooleanVar(value=True)),
                                  command=lambda: getattr(self, "refresh_plot", lambda: None)())
            m.add_cascade(label="View", menu=viewm)
    
            self.root.config(menu=m)
            self.root.bind_all(f"<{accel}-o>", lambda e: getattr(self, "on_open", getattr(self, "on_load_csv", lambda: None))())
            self.root.bind_all(f"<{accel}-s>", lambda e: getattr(self, "on_save_results", lambda: None)())
            self.root.bind_all(f"<{accel}-r>", lambda e: getattr(self, "on_run_fit", lambda: None)())
    
            import sys
            accel = "Command" if sys.platform == "darwin" else "Control"
            m = tk.Menu(self.root)
    
            filem = tk.Menu(m, tearoff=0)
            filem.add_command(label=f"Open…\t{accel}+O", command=getattr(self, "on_open", getattr(self, "on_load_csv", lambda: None)))
            filem.add_separator()
            filem.add_command(label=f"Save ALL\t{accel}+S", command=getattr(self, "on_save_results", lambda: None))
            filem.add_command(label="Save Annotated", command=lambda: self.save_plot_kind("main"))
            filem.add_command(label="Save Residuals", command=lambda: self.save_plot_kind("residuals"))
            filem.add_command(label="Save Zoom", command=lambda: self.save_plot_kind("zoom"))
            filem.add_command(label="Save Baseline Corrected", command=lambda: self.save_plot_kind("baseline_corrected"))
            filem.add_separator()
            filem.add_command(label="Save Alt Fit Plots", command=getattr(self, "on_save_alt_fit", lambda: None))
            filem.add_separator()
            filem.add_command(label="Exit", command=self.root.quit)
            m.add_cascade(label="File", menu=filem)
    
            fitm = tk.Menu(m, tearoff=0)
            fitm.add_command(label=f"Run Fit\t{accel}+R", command=getattr(self, "on_run_fit", lambda: None))
            m.add_cascade(label="Fit", menu=fitm)
    
            viewm = tk.Menu(m, tearoff=0)
            viewm.add_checkbutton(label="Grid", onvalue=True, offvalue=False,
                                  variable=getattr(self, "grid_var", tk.BooleanVar(value=True)),
                                  command=lambda: getattr(self, "refresh_plot", lambda: None)())
            m.add_cascade(label="View", menu=viewm)
    
            self.root.config(menu=m)
            self.root.bind_all(f"<{accel}-o>", lambda e: getattr(self, "on_open", getattr(self, "on_load_csv", lambda: None))())
            self.root.bind_all(f"<{accel}-s>", lambda e: getattr(self, "on_save_results", lambda: None)())
            self.root.bind_all(f"<{accel}-r>", lambda e: getattr(self, "on_run_fit", lambda: None)())
    
            import sys
            accel = "Command" if sys.platform == "darwin" else "Control"
            m = tk.Menu(self.root)
    
            filem = tk.Menu(m, tearoff=0)
            filem.add_command(label=f"Open…\t{accel}+O", command=getattr(self, "on_open", getattr(self, "on_load_csv", lambda: None)))
            filem.add_separator()
            filem.add_command(label=f"Save ALL\t{accel}+S", command=getattr(self, "on_save_results", lambda: None))
            filem.add_command(label="Save Annotated", command=lambda: self.save_plot_kind("main"))
            filem.add_command(label="Save Residuals", command=lambda: self.save_plot_kind("residuals"))
            filem.add_command(label="Save Zoom", command=lambda: self.save_plot_kind("zoom"))
            filem.add_command(label="Save Baseline Corrected", command=lambda: self.save_plot_kind("baseline_corrected"))
            filem.add_command(label="Save Alt Fit Plots", command=getattr(self, "on_save_alt_fit", lambda: None))
            filem.add_separator()
            filem.add_command(label="Exit", command=self.root.quit)
            m.add_cascade(label="File", menu=filem)
    
            fitm = tk.Menu(m, tearoff=0)
            fitm.add_command(label=f"Run Fit\t{accel}+R", command=getattr(self, "on_run_fit", lambda: None))
            m.add_cascade(label="Fit", menu=fitm)
    
            viewm = tk.Menu(m, tearoff=0)
            viewm.add_checkbutton(label="Grid", onvalue=True, offvalue=False,
                                  variable=getattr(self, "grid_var", tk.BooleanVar(value=True)),
                                  command=lambda: getattr(self, "refresh_plot", lambda: None)())
            m.add_cascade(label="View", menu=viewm)
    
            self.root.config(menu=m)
            self.root.bind_all(f"<{accel}-o>", lambda e: getattr(self, "on_open", getattr(self, "on_load_csv", lambda: None))())
            self.root.bind_all(f"<{accel}-s>", lambda e: getattr(self, "on_save_results", lambda: None)())
            self.root.bind_all(f"<{accel}-r>", lambda e: getattr(self, "on_run_fit", lambda: None)())
    
        def _ensure_toolbar_minimum(self):
            """If the scrollable toolbar is empty on some systems, build a minimal, cross-platform control row."""
            try:
                parent = self.toolbar.inner if hasattr(self, "toolbar") and self.toolbar else self.root
            except Exception:
                parent = self.root
            try:
                children = parent.winfo_children()
            except Exception:
                children = []
            if children:
                return
    
            import tkinter as tk
            from tkinter import ttk
            if not hasattr(self, "min_height_var"): self.min_height_var = tk.StringVar(value="0.5")
            if not hasattr(self, "window_factor_var"): self.window_factor_var = tk.StringVar(value="1.5")
            if not hasattr(self, "prominence_var"): self.prominence_var = tk.StringVar(value="")
            if not hasattr(self, "distance_var"): self.distance_var = tk.StringVar(value="")
            if not hasattr(self, "global_fit_var"): self.global_fit_var = tk.BooleanVar(value=True)
            if not hasattr(self, "lineshape_var"): self.lineshape_var = tk.StringVar(value="Auto")
            if not hasattr(self, "k_var"): self.k_var = tk.StringVar(value="2")
            if not hasattr(self, "grid_var"): self.grid_var = tk.BooleanVar(value=True)
    
            row = ttk.Frame(parent); row.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)
            ttk.Button(row, text="Load CSV…", command=getattr(self, "on_load_csv", lambda: None)).pack(side=tk.LEFT, padx=4)
            ttk.Button(row, text="Run Fit", command=getattr(self, "on_run_fit", lambda: None)).pack(side=tk.LEFT, padx=4)
            ttk.Label(row, text="min_height_frac:").pack(side=tk.LEFT, padx=(12,4))
            ttk.Entry(row, width=5, textvariable=self.min_height_var).pack(side=tk.LEFT)
            ttk.Label(row, text="window_factor:").pack(side=tk.LEFT, padx=(12,4))
            ttk.Entry(row, width=5, textvariable=self.window_factor_var).pack(side=tk.LEFT)
            ttk.Label(row, text="prom:").pack(side=tk.LEFT, padx=(12,4))
            ttk.Entry(row, width=5, textvariable=self.prominence_var).pack(side=tk.LEFT)
            ttk.Label(row, text="dist:").pack(side=tk.LEFT, padx=(12,4))
            ttk.Entry(row, width=5, textvariable=self.distance_var).pack(side=tk.LEFT)
            ttk.Checkbutton(row, text="Global fit", variable=self.global_fit_var).pack(side=tk.LEFT, padx=(12,4))
            ttk.Label(row, text="Profile:").pack(side=tk.LEFT, padx=(12,4))
            ttk.Combobox(row, width=9, textvariable=self.lineshape_var, values=("Auto","Gaussian","PVoigt","EMG")).pack(side=tk.LEFT, padx=(0,4))
    
            save_row = ttk.Frame(parent); save_row.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)

            ttk.Combobox(save_row, width=3, textvariable=self.k_var, values=[str(i) for i in range(1,10)]).pack(side=tk.LEFT, padx=(0,8))
# Always-accessible dropdown with mirrored actions
            self.more = ttk.Menubutton(save_row, text="More ▾")
            _menu = tk.Menu(self.more, tearoff=0)
            self.more["menu"] = _menu
            _menu.add_command(label="Save ALL", command=getattr(self, "on_save_results", lambda: None))
            _menu.add_separator()
            _menu.add_command(label="Save Annotated", command=lambda: self.save_plot_kind("main"))
            _menu.add_command(label="Save Residuals", command=lambda: self.save_plot_kind("residuals"))
            _menu.add_command(label="Save Zoom", command=lambda: self.save_plot_kind("zoom"))
            _menu.add_command(label="Save Baseline Corrected", command=lambda: self.save_plot_kind("baseline_corrected"))
            _menu.add_separator()
            _menu.add_command(label="Save Alt Fit Plots", command=getattr(self, "on_save_alt_fit", lambda: None))
            self.more.pack(side=tk.RIGHT, padx=(8, 2))
    
            # Main content area (use a simple Frame for reliability on macOS)
            if getattr(self, 'main_frame', None) is None:
                self.main_frame = ttk.Frame(root)

                # Status bar at bottom
                sb = ttk.Frame(root)
                ttk.Separator(sb, orient=tk.HORIZONTAL).pack(fill=tk.X)
                ttk.Label(sb, textvariable=self.status, anchor='w').pack(fill=tk.X, padx=8, pady=4)
                sb.pack(side=tk.BOTTOM, fill=tk.X)
    
            # Plot panel
            self.plot_frame = ttk.Frame(self.main_frame)
            try:
                self.main_frame.rowconfigure(0, weight=1)
                self.main_frame.columnconfigure(0, weight=1)
            except Exception:
                pass
                self.plot_frame.rowconfigure(0, weight=1)
                self.plot_frame.columnconfigure(0, weight=1)
                self.plot_frame.configure(height=500)
                self.plot_frame.pack_propagate(False)
            except Exception:
                pass
    
            self.fig = Figure(figsize=(9,6), dpi=110, constrained_layout=True)
        try:
            _scale = float(self.root.tk.call('tk', 'scaling'))
            if _scale and _scale > 0.0:
                self.fig.set_dpi(96.0 * _scale)
        except Exception:
            pass
            self.ax = self.fig.add_subplot(111)
            # Make plot area readable regardless of OS dark mode
            try:
                self.fig.patch.set_facecolor('white')
                self.ax.set_facecolor('white')
                for _s in self.ax.spines.values():
                    _s.set_color('black')
                self.ax.tick_params(axis='both', colors='black')
                self.ax.title.set_color('black')
                self.ax.xaxis.label.set_color('black')
                self.ax.yaxis.label.set_color('black')
            except Exception:
                pass
    
            self.ax.set_title("No data")
            self.ax.set_xlabel("wavelength_nm", fontsize=11)
            self.ax.set_ylabel("voltage_V", fontsize=11)
            self.ax.tick_params(axis="both", labelsize=10)
            self.ax.grid(True, which="both", alpha=0.25)
            # draw a placeholder line so canvas always paints
            try:
                self.ax.plot([], [])
            except Exception:
                pass
        # Create TkAgg canvas in the plot frame
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        if getattr(self, 'plot_frame', None) is None:
            self.plot_frame = ttk.Frame(self.main_frame)
            try:
                self.main_frame.rowconfigure(0, weight=1)
                self.main_frame.columnconfigure(0, weight=1)
                self.plot_frame.rowconfigure(0, weight=1)
                self.plot_frame.columnconfigure(0, weight=1)
            except Exception:
                pass
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        # Bind resize events to keep canvas filling the frame
        try:
            self.plot_frame.bind("<Configure>", self._force_canvas_fullsize)
            self.root.bind("<Configure>", self._force_canvas_fullsize)
        except Exception:
            pass
        # Kick an initial sizing pass after layout settles
        try:
            self.root.after(150, self._force_canvas_fullsize)
            self.root.after(300, self._force_canvas_fullsize)
        except Exception:
            pass
        # Keep figure sized to frame
        try:
            self.plot_frame.bind("<Configure>", self._resize_figure_to_frame)
        except Exception:
            pass
        # Do an initial size sync after layout
        try:
            self.root.after(50, self._resize_figure_to_frame)
        except Exception:
            pass
        try:
            self.plot_frame.rowconfigure(1, weight=0)
            self.plot_frame.columnconfigure(0, weight=1)
        except Exception:
            pass
        try:
            self.plot_frame.rowconfigure(0, weight=1)
            self.plot_frame.columnconfigure(0, weight=1)
        except Exception:
            pass
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        # Matplotlib navigation toolbar (Zoom/Pan/Save)
        try:
            self.mpl_toolbar = NavigationToolbar2Tk(self.canvas, self.bottom_bar, pack_toolbar=False)
            self.mpl_toolbar.update()
            # Place toolbar above the canvas
            self.mpl_toolbar.pack(side=tk.TOP, fill=tk.X)
        except Exception as _e:
            pass
        # --- macOS TkAgg visibility fix ---
        try:
            self.fig.set_size_inches(9.0, 6.0, forward=True)
        except Exception:
            pass
        try:
            self._ensure_vis_after_id = self.root.after(150, lambda: self._ensure_canvas_visible() if getattr(self, '_alive', True) else None)
            self._ensure_vis_after_id = self.root.after(300, lambda: self._ensure_canvas_visible() if getattr(self, '_alive', True) else None)
        except Exception:
            pass
        # --- end visibility fix ---
    
            try:
                _w = self.canvas.get_tk_widget()
                _w.configure(background='white', highlightthickness=0, borderwidth=0)
                _w.grid(row=0, column=0, sticky='nsew')
                _w.update_idletasks()
            except Exception:
                pass
            try:
                # make sure the embedded Tk widget isn't dark-gray on macOS themes
                _w = self.canvas.get_tk_widget()
                _w.configure(background='white', highlightthickness=0, borderwidth=0)
            except Exception:
                pass
            self.on_draw(idle=False)
            self.on_draw(idle=False)
            # Ensure canvas gets a visible height and redraw shortly after launch
            try:
                self.root.update_idletasks()
                self.on_draw(idle=False)
            except Exception:
                pass
    
            # Zoom controls
            zoombar = ttk.Frame(self.plot_frame)
            zoombar.grid(row=1, column=0, sticky='w', pady=4)
            self.zoom = ZoomManager(self.ax, self.canvas, on_draw=self._safe_canvas_draw)
            zoom_btn  = ttk.Button(zoombar, text="Zoom (drag)", command=self.zoom.toggle)
            back_btn  = ttk.Button(zoombar, text="Back",        command=self.zoom.back)
            reset_btn = ttk.Button(zoombar, text="Reset",       command=self.zoom.reset_to_full)
            zoom_btn.pack(side="left", padx=(0,6))
            back_btn.pack(side="left", padx=6)
            reset_btn.pack(side="left", padx=6)
    
        def _ensure_output_dir(self) -> str:
            dvar = getattr(self, "output_dir_var", None)
            outdir = dvar.get() if dvar is not None else getattr(self, "output_dir", os.getcwd())
            outdir = outdir or os.getcwd()
            try:
                os.makedirs(outdir, exist_ok=True)
            except Exception:
                pass
            return outdir
    
    def _save_kind(self, kind: str, prefix: str, x, y, res):
        outdir = self._ensure_output_dir()
        tag = self._current_model_tag(res)
        outpath = os.path.join(outdir, f"{prefix}_{tag}_{kind}.png")
        g = res.get("global_fit") if isinstance(res, dict) else None
        if kind == "main":
            try:
                _save_plot_main(x, y, g, outpath, peaks_idx=res.get("peaks_idx"), peaks_pos=res.get("peaks_pos"),
                                marker_nearest=bool(getattr(self, "marker_nearest_var", tk.BooleanVar(value=True)).get()))
            except NameError:
                _save_plot_baseline_corrected(x, y, g, outpath)
        elif kind == "residuals":
            _save_plot_residuals(x, y, g, outpath)
        elif kind == "zoom":
            try:
                _save_plot_zoom(x, y, g, outpath)
            except NameError:
                _save_plot_baseline_corrected(x, y, g, outpath)
        elif kind == "baseline_corrected":
            _save_plot_baseline_corrected(x, y, g, outpath)
        else:
            return None
        return outpath
    
    def on_save_results(self):
        if self.result is None or self.df is None:
            return
        from datetime import datetime
        x_col, y_col = pick_xy_columns(self.df)
        x = self.df[x_col].to_numpy(dtype=float)
        y = self.df[y_col].to_numpy(dtype=float)
        prefix = f"fit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        saved = []
        for kind in ("main", "residuals", "zoom", "baseline_corrected"):
            p = self._save_kind(kind, prefix, x, y, self.result)
            if p:
                saved.append(p)
        outdir = self._ensure_output_dir()
        try:
            import numpy as np, pandas as pd, os
            g = self.result.get("global_fit")
            y_fit = None
            if g is not None and hasattr(g, "predict"):
                try:
                    y_fit = g.predict(x)
                except Exception:
                    pass
            if y_fit is None and "y_fit" in self.result:
                try:
                    y_fit = np.asarray(self.result["y_fit"], dtype=float)
                    if y_fit.shape != x.shape:
                        y_fit = None
                except Exception:
                    y_fit = None
            if y_fit is not None:
                p_fit = os.path.join(outdir, f"{prefix}_fitted_curve.csv")
                pd.DataFrame({"x": x, "y_fit": y_fit}).to_csv(p_fit, index=False)
                saved.append(p_fit)
                resid = y - y_fit
                p_comb = os.path.join(outdir, f"{prefix}_data_and_fit.csv")
                pd.DataFrame({"x": x, "y": y, "y_fit": y_fit, "residual": resid}).to_csv(p_comb, index=False)
                saved.append(p_comb)
        except Exception:
            pass
        try:
            import pandas as pd, os, numpy as np
            rows = []
            g = self.result.get("global_fit") if isinstance(self.result, dict) else None
            if g is not None and hasattr(g, "params"):
                for i, pf in enumerate(getattr(g, "params", []), 1):
                    rows.append({
                        "which": f"global_peak_{i}",
                        "mu": getattr(pf, "mu", float("nan")),
                        "A": getattr(pf, "A", float("nan")),
                        "sigma": getattr(pf, "sigma", float("nan")),
                        "gamma": getattr(pf, "gamma", float("nan")),
                        "FWHM": getattr(pf, "FWHM", float("nan")),
                        "area": getattr(pf, "area", float("nan"))
                    })
            else:
                for i, pf in enumerate(self.result.get("peaks_fitted", []), 1):
                    rows.append({
                        "which": f"local_peak_{i}",
                        "mu": getattr(pf, "mu", float("nan")),
                        "A": getattr(pf, "A", float("nan")),
                        "sigma": getattr(pf, "sigma", float("nan")),
                        "gamma": getattr(pf, "gamma", float("nan")),
                        "FWHM": getattr(pf, "FWHM", float("nan")),
                        "area": getattr(pf, "area", float("nan"))
                    })
            if rows:
                csvp = os.path.join(outdir, f"{prefix}_peaks.csv")
                df_out = pd.DataFrame(rows)
                if 'A' in df_out.columns and 'voltage' not in df_out.columns:
                    df_out['voltage'] = df_out['A']
                df_out.to_csv(csvp, index=False); saved.append(csvp)
        except Exception:
            pass
        try:
            import pandas as pd, os
            ms = self.result.get("model_selection", [])
            if ms:
                csvp = os.path.join(outdir, f"{prefix}_model_selection.csv")
                pd.DataFrame(ms).to_csv(csvp, index=False); saved.append(csvp)
        except Exception:
            pass
        
        
        # NEW: save full fitted curve CSV (x, y_data, y_fit, baseline, y_bc, yfit_bc, residual)
        try:
            import numpy as np, pandas as pd, os
            y_fit_curve = None
            g = self.result.get("global_fit")
            # Prefer GlobalFit.y_fit if present
            if g is not None and hasattr(g, "y_fit"):
                try:
                    y_fit_curve = np.asarray(getattr(g, "y_fit"), dtype=float)
                except Exception:
                    y_fit_curve = None
            # If not, try predictor interface
            if y_fit_curve is None and g is not None and hasattr(g, "predict"):
                try:
                    y_fit_curve = np.asarray(g.predict(x), dtype=float)
                except Exception:
                    try:
                        y_fit_curve = np.asarray(g.predict(np.asarray(x).reshape(-1,1)), dtype=float)
                    except Exception:
                        y_fit_curve = None
            # Fallback to stored vector in result
            if y_fit_curve is None and isinstance(self.result.get("y_fit"), (list, tuple, np.ndarray)):
                try:
                    y_fit_curve = np.asarray(self.result["y_fit"], dtype=float)
                except Exception:
                    y_fit_curve = None

            baseline = None
            if "baseline" in self.result:
                try:
                    baseline = np.asarray(self.result["baseline"], dtype=float)
                except Exception:
                    baseline = None

            # If still missing, write minimal CSV (x,y)
            if y_fit_curve is None or y_fit_curve.shape != np.asarray(x).shape:
                df_curve = pd.DataFrame({"wavelength_nm": x, "y_data": y})
                csvp_curve = os.path.join(outdir, f"{prefix}_fit_curve.csv")
                df_curve.to_csv(csvp_curve, index=False); saved.append(csvp_curve)
            else:
                y_bc = y - baseline if baseline is not None and baseline.shape == np.asarray(x).shape else None
                yfit_bc = y_fit_curve - baseline if baseline is not None and baseline.shape == np.asarray(x).shape else None
                residual = y - y_fit_curve
                data = {
                    "wavelength_nm": x,
                    "y_data": y,
                    "y_fit": y_fit_curve,
                    "baseline": (baseline if baseline is not None and baseline.shape == np.asarray(x).shape else None),
                    "y_bc": (y_bc if y_bc is not None else None),
                    "yfit_bc": (yfit_bc if yfit_bc is not None else None),
                    "residual": residual
                }
                df_curve = pd.DataFrame({k:v for k,v in data.items() if v is not None})
                csvp_curve = os.path.join(outdir, f"{prefix}_fit_curve.csv")
                df_curve.to_csv(csvp_curve, index=False); saved.append(csvp_curve)
        except Exception:
            pass



        if saved:
            try:
                from tkinter import messagebox
                messagebox.showinfo("Saved", "Saved files:\n" + "\n".join(saved))
            except Exception:
                pass
            self._safe_status(f"Saved {len(saved)} files to {outdir}")

    
    def _current_model_tag(self, res=None):
        self._update_highest_peak_annotation()
        """Return normalized short model tag for filenames (gaussian|voigt|emg|auto)."""
        try:
            name = None
            if res and isinstance(res, dict):
                name = res.get('line_shape_used')
            if not name and hasattr(self, 'result') and isinstance(self.result, dict):
                name = self.result.get('line_shape_used')
            if not name and hasattr(self, 'lineshape_var'):
                name = str(self.lineshape_var.get())
            n = str(name or 'Gaussian').strip().lower()
            if 'voigt' in n: return 'voigt'
            if 'emg' in n or 'exp' in n: return 'emg'
            if 'gauss' in n: return 'gaussian'
            if 'auto' in n: return 'auto'
            return re.sub(r'[^a-z0-9]+', '', n) or 'model'
        except Exception:
            return 'model'

    def _update_highest_peak_annotation(self):
        # Remove previous markers
        try:
            if hasattr(self, '_peak_point') and self._peak_point is not None:
                self._peak_point.remove(); self._peak_point = None
        except Exception:
            self._peak_point = None
        try:
            if hasattr(self, '_peak_annot') and self._peak_annot is not None:
                self._peak_annot.remove(); self._peak_annot = None
        except Exception:
            self._peak_annot = None
        # Bail if disabled or no data
        if not getattr(self, 'show_highest_peak', tk.BooleanVar(value=False)).get():
            return
        if getattr(self, 'df', None) is None:
            return
        try:
            import numpy as np
            x_col, y_col = pick_xy_columns(self.df)
            x = np.asarray(self.df[x_col].to_numpy(dtype=float))
            y = np.asarray(self.df[y_col].to_numpy(dtype=float))
            if len(y) == 0 or not np.any(np.isfinite(y)):
                return
            idx = int(np.nanargmax(y))
            xpk, ypk = float(x[idx]), float(y[idx])
            self._peak_point = self.ax.scatter([xpk], [ypk], s=36, zorder=6)
            label = f"{xpk:.2f}"
            self._peak_annot = self.ax.annotate(label, xy=(xpk, ypk), xytext=(6,6),
                                               textcoords='offset points', fontsize=9, zorder=6)
            try:
                self._safe_status(f"Highest peak → x={xpk:.4g}, y={ypk:.4g}")
            except Exception:
                pass
        except Exception:
            pass

    def draw_plot(self, x: np.ndarray, y: np.ndarray, res: dict):
        # Clear and plot data; force visible paint on TkAgg.
        try:
            self.ax.clear()
        except Exception:
            return
        try:
            xd, yd, idx = _decimate_for_plot(x, y, max_points=4000)
        except Exception:
            import numpy as _np
            xd = _np.asarray(x, dtype=float); yd = _np.asarray(y, dtype=float)
            idx = _np.arange(len(xd), dtype=int)
        try:
            self.ax.plot(xd, yd, label='data', lw=1.4)
        except Exception:
            pass
        # Overlay global fit if present
        g = res.get('global_fit') if isinstance(res, dict) else None
        try:
            if g is not None and hasattr(g, 'y_fit'):
                import numpy as _np
                yfit = _np.asarray(g.y_fit, dtype=float)
                if len(yfit) == len(y):
                    yfitd = yfit[idx]; xf = xd
                else:
                    yfitd = yfit; xf = _np.asarray(getattr(g, 'x_fit', x), dtype=float)
                label = f"{_resolve_model_display_name(getattr(g,'model_name','Gaussian'))} fit (R2={getattr(g,'R2', float('nan')):.3f})"
                self.ax.plot(xf[:len(yfitd)], yfitd, label=label, linewidth=2.0)
        except Exception:
            pass
        # Visible marker and watermark
        try:
            if len(xd) and len(yd):
                import numpy as _np
                j = int(_np.nanargmax(yd))
                self.ax.plot([xd[j]], [yd[j]], marker='o', markersize=6, zorder=5)
        except Exception:
            pass
        try:
            self.ax.text(0.02, 0.98, f'Rendered {len(xd)} pts', transform=self.ax.transAxes, va='top', ha='left', alpha=0.6)
        except Exception:
            pass
        # Labels/grid/limits
        try:
            self.ax.set_xlabel('wavelength_nm'); self.ax.set_ylabel('voltage_V')
            self.ax.grid(True, which='both', alpha=0.25)
        except Exception:
            pass
        try:
            import numpy as _np
            if len(xd) and len(yd):
                xmn = float(_np.nanmin(xd)); xmx = float(_np.nanmax(xd))
                ymn = float(_np.nanmin(yd)); ymx = float(_np.nanmax(yd))
                ypad = (ymx-ymn)*0.05 if ymx>ymn else 1.0
                self.ax.set_xlim(xmn, xmx); self.ax.set_ylim(ymn-ypad, ymx+ypad)
        except Exception:
            pass
        try:
            self.ax.legend(loc='best')
        except Exception:
            pass
        # Tight layout best-effort
        try: self.fig.tight_layout()
        except Exception: pass
        
        # Final draw + ensure Tk paints and widget is realized
        self.on_draw(idle=False)
        try:
            self._ensure_canvas_visible()
        except Exception:
            pass
                # Force immediate redraw and Tk widget refresh
        try:
            self.on_draw(idle=False)
            _w = self.canvas.get_tk_widget()
            _w.update_idletasks(); _w.update()
        except Exception:
            pass
        self._update_highest_peak_annotation()
        try: self.canvas.draw_idle()
        except Exception: pass
    def on_choose_output_dir(self):
        d = filedialog.askdirectory(title="Select Output Folder", mustexist=True)
        if d:
            self.output_dir = d
            self.output_dir_var.set(d)

    def on_drop_files(self, event):
        paths = self.root.splitlist(event.data)
        if not paths:
            return
        first = paths[0]
        if os.path.isdir(first):
            messagebox.showinfo("Drop", "Please drop a CSV file, not a folder.")
            return
        self.load_csv(first)

    def on_load_csv(self):
        p = filedialog.askopenfilename(title="Select CSV", filetypes=[("CSV files","*.csv"),("All files","*.*")])
        if p:
            self.load_csv(p)

    
    def load_csv(self, p: str):
            """
            Robust CSV loader: picks x,y columns, coerces to numeric, drops NaNs,
            updates labels, enables Run, and draws data immediately.
            Any failure is shown in a message box and status bar (no crash).
            """
            try:
                df = parse_csv(p)
                x_col, y_col = pick_xy_columns(df)
                self.csv_path = p
                self.df = df
                try:
                    self.path_label.configure(text=os.path.basename(p) + f"  ({x_col} vs {y_col})")
                except Exception:
                    pass
                # Status early
                self._safe_status(f"Loading {p} ...")
    
                # Coerce numerics safely
                x = pd.to_numeric(df[x_col], errors="coerce")
                y = pd.to_numeric(df[y_col], errors="coerce")
                mask = x.notna() & y.notna()
                x = x[mask].astype(float).to_numpy()
                y = y[mask].astype(float).to_numpy()
    
                if x.size == 0 or y.size == 0:
                    raise ValueError("No numeric data found after parsing first two columns.")
    
                # Enable run if present
                try:
                    self.btn_run.configure(state=tk.NORMAL)
                except Exception:
                    pass
    
                # Draw immediately
                self.draw_plot(x, y, {})
                self._update_save_buttons(True, False)
                self._update_highest_peak_annotation()
                
                self._safe_status(f"Loaded {p}")
    
            except Exception as e:
                import traceback
                traceback.print_exc()
                try:
                    messagebox.showerror("Load error", str(e))
                except Exception:
                    pass
                self._safe_status(f"Load failed: {e}")

    
    def on_run_fit(self):
            if self.df is None:
                return
            try:
                min_height = float(self.min_height_var.get())
                window_factor = float(self.window_factor_var.get())
                if not (0 < min_height <= 1.0):
                    raise ValueError("min_height_frac must be in (0,1].")
                if window_factor < 1.0:
                    window_factor = 1.0
                # Clamp to sensible ranges
                if not (0 < min_height <= 1.0):
                    raise ValueError("min_height_frac must be in (0,1].")
                if window_factor < 1.0:
                    window_factor = 1.0
            except Exception:
                messagebox.showerror("Input error", "min_height_frac and window_factor must be numbers.")
                return
    
            x_col, y_col = pick_xy_columns(self.df)
            x = self.df[x_col].to_numpy(dtype=float)
            y = self.df[y_col].to_numpy(dtype=float)
            # Optional peak detection parameters
            prominence = None
            distance = None
            try:
                s = (self.prominence_var.get() or '').strip()
                if s:
                    prominence = float(s)
            except Exception:
                prominence = None
            try:
                s = (self.distance_var.get() or '').strip()
                if s:
                    distance = int(float(s))
            except Exception:
                distance = None
    
            do_global = bool(self.global_fit_var.get())
            line_shape = self.lineshape_var.get()
    
            try:
                self.btn_run.configure(state=tk.DISABLED)
            except Exception:
                pass
            self._safe_status("Running fit…")
    
            def task():
                ls = str(line_shape).strip().lower()
                if ls in ("auto","automatic"):
                    return analyze_spectrum_auto(x=x, y=y, min_height_frac=min_height, window_factor=window_factor, distance=distance, prominence=prominence)
                else:
                    return analyze_spectrum(
                        x=x, y=y,
                        pre_baseline=True,
                        line_shape=line_shape,
                        min_height_frac=min_height,
                        distance=distance,
                        prominence=prominence,
                        window_factor=window_factor,
                        do_global_fit=do_global,
                        plot=False,
                        save_plots=False,
                    )
    
            
            def done(fut):
                try:
                    res = fut.result()
                except Exception as e:
                    self.root.after(0, lambda e=e: (
                        self._safe_status(f"Fit failed: {e}"),
                        self.btn_run.configure(state=tk.NORMAL)
                    ))
                    return
                def ui_update():
                    self.result = res
                    self.populate_tables(res)
                    self.draw_plot(x, y, res)
                    self._update_save_buttons(True, True)
                    self._safe_status("Fit complete.")
                    self._update_save_buttons(True, True)
                    self.btn_run.configure(state=tk.NORMAL)
                self.root.after(0, ui_update)
            fut = self.executor.submit(task)
            fut.add_done_callback(done)

    def populate_tables(self, res):
        """Fill results tables if present; no-op safely otherwise."""
        try:
            trees = [getattr(self, "tree", None), getattr(self, "tree_ms", None)]
            if not any(trees):
                return
            # Build rows from `res`
            rows = []
            try:
                g = res.get("global_fit")
                if g is not None and hasattr(g, "params"):
                    for i, pf in enumerate(getattr(g, "params", []), 1):
                        try:
                            rows.append(("global_peak_"+str(i),
                                         f"{pf.mu:.6f}", f"{pf.FWHM:.6f}", f"{pf.A:.6g}", f"{pf.area:.6g}"))
                        except Exception:
                            pass
                peaks = res.get("peaks", [])
                for p in peaks:
                    try:
                        rows.append(("peak",
                                     f"{p.get('mu', float('nan')):.6f}",
                                     f"{p.get('FWHM', float('nan')):.6f}",
                                     f"{p.get('A', float('nan')):.6g}",
                                     f"{p.get('area', float('nan')):.6g}"))
                    except Exception:
                        pass
            except Exception:
                pass

            for tv in trees:
                if not tv:
                    continue
                try:
                    # Clear items
                    try:
                        for iid in tv.get_children():
                            tv.delete(iid)
                    except Exception:
                        pass
                    # Insert rows
                    for r in rows:
                        try:
                            tv.insert("", "end", values=r)
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass
    

    

    def save_plot_kind(self, kind: str, **kwargs):
    

    

        if getattr(self, 'df', None) is None:
    

    

            return
    

    

        x_col, y_col = pick_xy_columns(self.df)
    

    

        x = self.df[x_col].to_numpy(dtype=float)
    

    

        y = self.df[y_col].to_numpy(dtype=float)
    

    

        res = self.result if getattr(self, 'result', None) is not None else {}
    

    

        if kind != 'main' and not res:
    

    

            try:
    

    

                from tkinter import messagebox
    

    

                messagebox.showinfo('Save', 'Run fit first to save this view.')
    

    

            except Exception:
    

    

                pass
    

    

            return
    

    

        p = self._save_kind(kind, 'fit', x, y, res)
    

    

        if p:
    

    

            self._safe_status(f'Saved: {p}')
    

    

            try:
    

    

                from tkinter import messagebox
    

    

                messagebox.showinfo('Saved', f'Saved {kind} plot:\n{p}')
    

    

            except Exception:
    

    

                pass

# ------------------------
# Robust launcher
# ------------------------
def _bootstrap():
    import traceback
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
    except Exception as e:
        print("Tkinter import failed:", e)
        raise

    try:
        root = tk.Tk()
        root.title(APP_TITLE)
        try:
            root.geometry("1200x800")
        except Exception:
            pass
        app = App(root)
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        try:
            messagebox.showerror("Startup error", tb)
        except Exception:
            pass
    try:
        root.deiconify()
        root.lift()
        root.focus_force()
    except Exception:
        pass
    try:
        root.mainloop()
    except Exception as e:
        print("Mainloop exited with error:", e)

if __name__ == "__main__":
    _bootstrap()