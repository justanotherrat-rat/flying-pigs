"""Shared test helpers: an in-memory DEM, a GeoTIFF writer, a placement-score
oracle and an independent Python reference of the ns-3 two-ray sea model."""
from types import SimpleNamespace

import numpy as np

EARTH_R = 6371000.0  # same constant the module uses


class FakeDEM:
    """In-memory EPSG:4326 stand-in for a rasterio dataset.

    Implements exactly the surface losGenerationBands uses:
    index(lon, lat), read(band, window=...), height, width, nodata, bounds, res.
    Starts as flat sea (0 m). Add land with set_box_* helpers.
    """

    def __init__(self, lat_ref, lon_ref, half_deg=0.8, res=0.002, fill=0.0):
        self.res = (res, res)
        self.width = self.height = int(round(2 * half_deg / res))
        self.left, self.right = lon_ref - half_deg, lon_ref + half_deg
        self.bottom, self.top = lat_ref - half_deg, lat_ref + half_deg
        self.bounds = SimpleNamespace(left=self.left, right=self.right,
                                      bottom=self.bottom, top=self.top)
        self.nodata = None
        self.crs = "EPSG:4326"
        self.data = np.full((self.height, self.width), fill, dtype=np.float32)

    def index(self, lon, lat):
        return (int(np.floor((self.top - lat) / self.res[1])),
                int(np.floor((lon - self.left) / self.res[0])))

    def read(self, band=1, window=None):
        if window is None:
            return self.data
        r, c = int(window.row_off), int(window.col_off)
        return self.data[r:r + int(window.height), c:c + int(window.width)]

    def set_box_lonlat(self, lon_min, lon_max, lat_min, lat_max, value):
        r0, c0 = self.index(lon_min, lat_max)
        r1, c1 = self.index(lon_max, lat_min)
        self.data[max(r0, 0):max(r1, 0) + 1, max(c0, 0):max(c1, 0) + 1] = value

    def set_box_enu(self, lat_ref, lon_ref, e_min, e_max, n_min, n_max, value):
        """Same flat-earth ENU->degrees mapping as losGenerationBands.enu_to_lla."""
        k = EARTH_R * np.cos(np.radians(lat_ref))
        self.set_box_lonlat(lon_ref + np.degrees(e_min / k), lon_ref + np.degrees(e_max / k),
                            lat_ref + np.degrees(n_min / EARTH_R),
                            lat_ref + np.degrees(n_max / EARTH_R), value)


def write_geotiff(path, lat_ref, lon_ref, half_deg=0.5, res=0.002, land_east_of_lon=None,
                  land_height=50.0):
    """Write a small GeoTIFF (sea everywhere; optional land strip east of a longitude)."""
    import rasterio
    from rasterio.transform import from_origin
    n = int(round(2 * half_deg / res))
    data = np.zeros((n, n), dtype=np.float32)
    left, top = lon_ref - half_deg, lat_ref + half_deg
    if land_east_of_lon is not None:
        col = int(np.floor((land_east_of_lon - left) / res))
        data[:, max(col, 0):] = land_height
    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(left, top, res, res)) as dst:
        dst.write(data, 1)
    return path


def placement_score(lgb, graph_dense):
    """Re-implementation of the connectivity part of place_uavs' objective
    (no travel penalty, no link-length term) used as an independent oracle."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    adj = np.asarray(graph_dense)
    metrics = lgb._network_metrics(adj)
    missing_two_hop = metrics["total_pairs"] - metrics["pairs_within_two_hops"]
    dist = shortest_path(csr_matrix(adj), directed=False, unweighted=True)
    pair = dist[np.triu_indices(len(adj), k=1)]
    disconnected = int(np.count_nonzero(~np.isfinite(pair)))
    return 100000.0 * disconnected + 10000.0 * missing_two_hop


# --------------------------------------------------------------------------
# Independent reference of uav_mesh_eval.cc's maritime two-ray model
# (ITU-R P.527-6 seawater permittivity, Fresnel reflection, Ament roughness)
# --------------------------------------------------------------------------
SEA_RMS_HEIGHT = [0.0, 0.0125, 0.075, 0.21875, 0.46875, 0.8125, 1.25, 1.875, 2.875, 4.25]


def fspl_db(d_m, f_ghz):
    return 20 * np.log10(4 * np.pi * d_m * f_ghz * 1e9 / 299792458.0)


def sea_permittivity(f, T=20.0, S=35.0):
    th = 300 / (T + 273.15) - 1
    es = 77.66 + 103.3 * th
    e1 = 0.0671 * es
    einf = 3.52 - 7.52 * th
    f1 = 20.20 - 146.4 * th + 316.0 * th * th
    f2 = 39.8 * f1
    ess = es * np.exp(-3.3333e-3 * S + 4.74868e-6 * S * S)
    f1s = f1 * (1 + S * (2.3232e-3 - 7.9208e-5 * T + 3.6764e-6 * T**2 + 3.5594e-7 * T**3
                         + 8.9795e-9 * T**4))
    e1s = e1 * np.exp(-6.28908e-3 * S + 1.76032e-4 * S * S - 9.22144e-5 * T * S)
    f2s = f2 * (1 + S * (-1.99723e-2 + 1.81176e-4 * T))
    einfs = einf * (1 + S * (-2.04265e-3 + 1.57883e-4 * T))
    s35 = 2.903602 + 8.607e-2 * T + 4.738817e-4 * T**2 - 2.991e-6 * T**3 + 4.3047e-9 * T**4
    r15 = S * (37.5109 + 5.45216 * S + 1.4409e-2 * S * S) / (1004.75 + 182.283 * S + S * S)
    a0 = (6.9431 + 3.2841 * S - 9.9486e-2 * S * S) / (84.850 + 69.024 * S + S * S)
    a1 = 49.843 - 0.2276 * S + 0.198e-2 * S * S
    sigma = s35 * r15 * (1 + a0 * (T - 15) / (a1 + T))
    x1, x2 = f / f1s, f / f2s
    re = (ess - e1s) / (1 + x1**2) + (e1s - einfs) / (1 + x2**2) + einfs
    im = x1 * (ess - e1s) / (1 + x1**2) + x2 * (e1s - einfs) / (1 + x2**2) + 18 * sigma / f
    return re, im


def two_ray_db(d, h1, h2, f_ghz, sea_state=3, vertical=True, temp_c=20.0, salinity=35.0):
    d, h1, h2 = max(d, 1.0), max(h1, 0.1), max(h2, 0.1)
    r1, r2 = np.hypot(d, h1 - h2), np.hypot(d, h1 + h2)
    grazing = np.arctan((h1 + h2) / d)
    lam = 299792458.0 / (f_ghz * 1e9)
    dphi = (4 * np.pi * h1 * h2 / (lam * d) if d > 20 * (h1 + h2)
            else 2 * np.pi * (r2 - r1) / lam)
    re, im = sea_permittivity(f_ghz, temp_c, salinity)
    eps = complex(re, -im)
    sp, cp = np.sin(grazing), np.cos(grazing)
    root = np.sqrt(eps - cp * cp)
    refl = (eps * sp - root) / (eps * sp + root) if vertical else (sp - root) / (sp + root)
    rough = np.exp(-2 * (2 * np.pi * SEA_RMS_HEIGHT[min(max(sea_state, 0), 9)] * sp / lam) ** 2)
    total = 1 + refl * rough * np.exp(-1j * dphi) * (r1 / r2)
    return fspl_db(r1, f_ghz) - 20 * np.log10(max(abs(total), 1e-3))
