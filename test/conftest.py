"""pytest configuration: loads losGenerationBands.py as a module.

Use a different/fixed copy of the module with either
    pytest --lgb-module=path/to/losGenerationBands.py  (note the "=": with a space pytest
    mistakes the path for a test directory and skips this conftest)
or  LGB_MODULE=path/to/losGenerationBands pytest
"""
import importlib.util
import os
from importlib.machinery import SourceFileLoader
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # never open plot windows

import pytest

from helpers import FakeDEM

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MODULE = REPO / "simulation" / "losGenerationBands.py"
REAL_DEM = REPO / "simulation" / "output_hh.tif"
REF = (1.4, 103.3)  # reference lat/lon used by repo's own placement test


def pytest_addoption(parser):
    parser.addoption("--lgb-module", default=os.environ.get("LGB_MODULE", str(DEFAULT_MODULE)),
                     help="path to the losGenerationBands module under test")


@pytest.fixture(scope="session")
def lgb(request):
    path = Path(request.config.getoption("--lgb-module")).expanduser().resolve()
    if not path.is_file():
        pytest.exit(f"module under test not found: {path}", returncode=2)
    loader = SourceFileLoader("losGenerationBands_under_test", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


@pytest.fixture
def ref():
    return REF


@pytest.fixture
def flat_dem():
    """Flat open sea (0 m) covering lat 0.6-2.2, lon 102.5-104.1."""
    return FakeDEM(*REF, half_deg=0.8)


@pytest.fixture(scope="session")
def real_dem():
    rasterio = pytest.importorskip("rasterio")
    if not REAL_DEM.is_file():
        pytest.skip(f"real DEM not found at {REAL_DEM}")
    ds = rasterio.open(REAL_DEM)
    yield ds
    ds.close()


@pytest.fixture
def real_dem_path():
    if not REAL_DEM.is_file():
        pytest.skip(f"real DEM not found at {REAL_DEM}")
    return REAL_DEM
