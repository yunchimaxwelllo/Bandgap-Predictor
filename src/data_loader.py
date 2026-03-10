import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.signal import find_peaks

# XRD Helpers
GRID = np.arange(10.0, 80.0 + 1e-9, 0.02, dtype=np.float32)
XRDF_MIN = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0, 0.0], dtype=np.float32)
XRDF_MAX = np.array([1000., 20.0, 80.0, 35., 1.0, 1.0, 80.0, 9.0], dtype=np.float32)

def fwhm_to_sigma(fwhm): return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

def shift_1d(y, shift_bins):
    if shift_bins == 0: return y
    out = np.roll(y, shift_bins).astype(np.float32)
    if shift_bins > 0: out[:shift_bins] = 0.0
    else: out[shift_bins:] = 0.0
    return out

def gaussian_blur_approx(y, sigma_bins):
    if sigma_bins <= 0.01: return y
    r = int(max(2, 3 * sigma_bins))
    k = np.exp(-0.5 * (np.arange(-r, r + 1, dtype=np.float32) / sigma_bins) ** 2)
    return np.convolve(y, k / np.sum(k), mode="same").astype(np.float32)

def xrd_peak_stats_features(y, tt, eps=1e-12):
    y0 = y - float(np.min(y))
    if y0.max() > 0: y0 = (y0 / float(np.max(y0))).astype(np.float32)
    dist_samples = max(1, int(np.round(0.10 / max(float(np.median(np.diff(tt))), eps))))
    peak_idx, _ = find_peaks(y0, prominence=0.02, height=0.02, distance=dist_samples)
    p_tt, p_y = tt[peak_idx], y0[peak_idx]
    cnt = float(len(peak_idx))
    dens = cnt / max(float(tt[-1] - tt[0]), eps)
    m_pos, s_pos = (float(np.mean(p_tt)), float(np.std(p_tt))) if cnt > 0 else (0.0, 0.0)
    m_int, s_int = (float(np.mean(p_y)), float(np.std(p_y))) if cnt > 0 else (0.0, 0.0)
    y_pos = np.clip(y0, 0.0, None)
    denom = float(np.sum(y_pos)) + eps
    iw_avg = float(np.sum(tt * y_pos) / denom)
    p = y_pos / denom
    entropy = float(-np.sum(p * np.log(p + eps)))
    feats = np.array([cnt, dens, m_pos, s_pos, m_int, s_int, iw_avg, entropy], dtype=np.float32)
    return np.clip(feats, XRDF_MIN, XRDF_MAX)

# PyTorch Dataset 
class AblationDataset(Dataset):
    def __init__(self, ids, mag_source, struct_scaled, geo_scaled, xrd_scaled, id2bg, train=True):
        self.mag_source = mag_source
        self.struct_scaled = struct_scaled
        self.geo_scaled = geo_scaled
        self.xrd_scaled = xrd_scaled
        self.id2bg = id2bg
        self.train = train
        
        self.ids = [i for i in ids if (i in mag_source) and (i in struct_scaled) and (i in geo_scaled) and (i in xrd_scaled)]

    def __len__(self): return len(self.ids)
    
    def _augment(self, y):
        if 0.05 > 0: y = shift_1d(y, int(np.round(np.random.uniform(-0.05, 0.05) / 0.02)))
        if 0.02 > 0: y = gaussian_blur_approx(y, fwhm_to_sigma(max(0.05, 0.28 + np.random.uniform(-0.02, 0.02))) / 0.02)
        if 0.02 > 0: y += np.random.uniform(0, 0.02) + np.random.uniform(-0.02, 0.02) * np.linspace(0, 1, len(y))
        if 0.005 > 0: y += np.random.normal(0, 0.005, size=y.shape)
        y = np.clip(y, 0.0, None).astype(np.float32)
        return (y / y.max()).astype(np.float32) if y.max() > 0 else y

    def __getitem__(self, idx):
        mid = self.ids[idx]
        y = self.xrd_scaled[mid].astype(np.float32)
        if self.train: y = self._augment(y)
        x_xrd = torch.nan_to_num(torch.from_numpy(y[None, :]))
        x_xrdf = torch.nan_to_num(torch.from_numpy(xrd_peak_stats_features(y, GRID)))
        x_g1, x_g2 = [torch.nan_to_num(torch.from_numpy(g).float()) for g in self.mag_source[mid]]
        x_symm = torch.nan_to_num(torch.from_numpy(self.struct_scaled[mid]["symm"]).float())
        x_vol  = torch.nan_to_num(torch.from_numpy(self.struct_scaled[mid]["volpa"]).float())
        x_geo  = torch.nan_to_num(torch.from_numpy(self.geo_scaled[mid]).float())
        y_log = torch.tensor([np.float32(np.log1p(self.id2bg[mid]))], dtype=torch.float32)

        return x_xrd, x_xrdf, x_g1, x_g2, x_symm, x_vol, x_geo, y_log, mid

def get_loaders(split, mag_source, struct_scaled, geo_scaled, xrd_scaled, id2bg):
    tr = DataLoader(AblationDataset(split["train"], mag_source, struct_scaled, geo_scaled, xrd_scaled, id2bg, train=True), batch_size=64, shuffle=True, drop_last=True)
    va = DataLoader(AblationDataset(split["val"], mag_source, struct_scaled, geo_scaled, xrd_scaled, id2bg, train=False), batch_size=64, shuffle=False)
    te = DataLoader(AblationDataset(split["test"], mag_source, struct_scaled, geo_scaled, xrd_scaled, id2bg, train=False), batch_size=64, shuffle=False)
    return tr, va, te

# Scikit-Learn Data Builder 
def build_separated_dataset(mids, magpie_scaled, struct_scaled, geo_scaled, id2bg):
    X_m1, X_m2, X_oth, y = [], [], [], []
    for mid in mids:
        if mid in magpie_scaled and mid in id2bg:
            X_m1.append(magpie_scaled[mid][0])
            X_m2.append(magpie_scaled[mid][1])
            
            other = np.concatenate([
                struct_scaled[mid]["symm"], 
                struct_scaled[mid]["volpa"], 
                struct_scaled[mid]["lattice"],
                geo_scaled[mid]
            ])
            X_oth.append(other)
            y.append(id2bg[mid])
            
    return (np.nan_to_num(np.array(X_m1, dtype=np.float32)), 
            np.nan_to_num(np.array(X_m2, dtype=np.float32)), 
            np.nan_to_num(np.array(X_oth, dtype=np.float32)), 
            np.nan_to_num(np.array(y, dtype=np.float32)))