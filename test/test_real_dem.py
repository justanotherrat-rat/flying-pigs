"""Sanity checks on the bundled DEM (simulation/output_hh.tif)."""
import numpy as np
import pytest

pytestmark = pytest.mark.slow


def test_dem_is_wgs84_and_covers_the_scenario(real_dem):
    assert real_dem.crs.to_epsg() == 4326
    b = real_dem.bounds
    assert b.left <= 103.0 and b.right >= 104.0 and b.bottom <= 0.6 and b.top >= 2.0


def test_dem_resolution_is_about_one_arcsecond(real_dem):
    assert real_dem.res[0] == pytest.approx(1 / 3600, rel=0.01)


def test_known_sea_point_is_water(lgb, real_dem):
    assert not lgb.is_on_land(103.5, 0.6, real_dem)


def test_known_hill_is_land_and_high(lgb, real_dem):
    # Bukit Timah Hill, Singapore (~163 m)
    assert lgb.get_elevation_at_lonlat(103.7764, 1.3546, real_dem) > 100
    assert lgb.is_on_land(103.7764, 1.3546, real_dem)


def test_default_convoy_box_is_mostly_inside_dem(real_dem):
    # generate_convoy_lla samples lon 102.8-103.6; the DEM starts at 103.0, so the
    # western fifth is outside coverage and silently rejected as "land". Documented here.
    b = real_dem.bounds
    covered = (103.6 - max(b.left, 102.8)) / 0.8
    assert covered == pytest.approx(0.75, abs=0.01)


def test_dem_has_no_nodata_flag_and_no_nan(real_dem):
    data = real_dem.read(1, window=((0, 600), (0, 600)))
    assert real_dem.nodata is None
    assert np.isfinite(data).all()


def test_elevation_outside_dem_is_none(lgb, real_dem):
    assert lgb.get_elevation_at_lonlat(50.0, 10.0, real_dem) is None
