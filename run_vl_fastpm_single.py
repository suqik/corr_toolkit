"""Measure FastPM mock void lensing for a single simulation.

Run with: python run_vl_fastpm_single.py
Only cosmo_000001 is processed, using the boss_lowz_e2_e3 north 3-tomo shape
catalog for all 10 HOD samples. The radial scale is the median Rv of the
selected, footprint-matched voids, as implemented by GGLCalculator (not a
separate radial scale for each void).
"""

from pathlib import Path

import numpy as np

from container import SurveyData
from calculator import GGLCalculator, GGLConfig


NJOBS = 1
NGAL = 3.5e-4  # Fixed tracer number density in (h/Mpc)^3 for all samples.
ICOSMO = 0  # The only FastPM simulation processed by this script.

wdir = Path(__file__).resolve().parent
data_dirbase = Path("/Users/suqikuai777/Dataspace/FastPM/MockCatalogs")
lens_dir = data_dirbase / "Voids"
lens_fmt = "cosmo_{:06d}_realization_0000_HOD_{:d}_boss_lowz_north_2dflens_south.fits"

srcs_dir = data_dirbase / "Shape"
srcs_fmt = "cosmo_{:06d}_realization_0002_boss_cmass_north_3tomos.fits"
# srcs_fmt = "kids1000_north_3tomos/cosmo_{:06d}_realization_0002_kids1000_north_3tomos.fits"

# rand_dir = data_dirbase / "Rand"
# rand_fmt = "DS20/boss_cmasslowztot_ngc_z0.2_0.4_official.fits"

out_dir = wdir / "results/vl/bosslowz_north_full"
out_fmt = "cosmo{:06d}_HOD{:d}_vl_update_v1.2.3.fits"
cosmo_param_info_file = data_dirbase.parent / "cosmo_list.txt"


def load_cosmo_params(fname):
    """Read the fixed hubble parameter and row-indexed FastPM cosmologies."""
    with open(fname) as f:
        fixed = dict(token.split("=", 1) for token in f.readline().lstrip("#").split())
    hubble = float(fixed["hubble"])
    rows = np.loadtxt(fname, ndmin=2)
    return {
        icosmo: {"Om0": float(row[0]), "H0": 100.0 * hubble, "w0": -1.0}
        for icosmo, row in enumerate(rows)
    }


def main():
    cosmo_params = load_cosmo_params(cosmo_param_info_file)

    ggl_config = GGLConfig(
        rp_min=0.1,
        rp_max=3.0,
        rp_bins=13,
        rp_unit="Rv",
        bin_type="linear",
        flip_g1=False,
        flip_g2=True,  # Same FastPM shear convention as run_ggl_fastpm.py.
        wRSD=False,  # Void catalogs already contain the void-center redshift.
        wSN=False,
        wPhZ=False,
    )
    ggl_instance = GGLCalculator(config=ggl_config)
    out_dir.mkdir(parents=True, exist_ok=True)

    icosmo = ICOSMO
    cosmo_dict = cosmo_params[icosmo]
    
    mock_shape = SurveyData.load_cosmogrid_shape(srcs_dir / srcs_fmt.format(icosmo))
    if len(mock_shape) == 0:
        raise ValueError(f"Empty shape catalog for cosmology {icosmo}")
    
    mock_shape = mock_shape[mock_shape.tomo == 4]
    mock_shape = mock_shape[::10]
    srcs_table = ggl_instance.mk_srcs_cat(mock_shape)

    # mock_rand = SurveyData.load_cosmogrid_rand(rand_dir / rand_fmt)
    
    ihod = 0
    
    mock_void = SurveyData.load_cosmogrid_void(lens_dir / lens_fmt.format(icosmo, ihod))
    mock_void_boss = mock_void[mock_void.survey != 3]
    mock_void_boss_curr = mock_void_boss[((mock_void_boss.Rv > 15) & (mock_void_boss.Rv < 16))]
    # mock_void_boss_curr = mock_void_boss[((mock_void_boss.Rv > 15) & (mock_void_boss.Rv < 25))]
    mock_void_boss_curr = mock_void_boss_curr
    mock_void_boss_matched = mock_void_boss_curr.match_to_reference(
        mock_shape, nside=256, in_place=False
    )
    if len(mock_void_boss_matched) == 0:
        raise ValueError(f"No selected voids overlap the shapes: cosmo={icosmo}, HOD={ihod}")
    
    # mock_rand_matched = mock_rand.match_to_reference(
    #     mock_shape, nside=256, in_place=False
    # )

    lens_table = ggl_instance.mk_lens_cat(mock_void_boss_matched)
    # rand_table = ggl_instance.mk_lens_cat(mock_rand_matched)
    # rand_table = rand_table[::20]
    
    ggl_instance.get_Rv_mean_mpch(mock_void_boss_matched)
    # Rv (and hence physical bin edges) changes with each void sample.
    # Start from fresh randoms and recompute pairs for every HOD.
    lens_table = ggl_instance.compute_pairs(
        cosmo_dict, lens_table, srcs_table, n_jobs=NJOBS
    )
    # rand_table = ggl_instance.compute_pairs(
    #     cosmo_dict, rand_table, srcs_table, n_jobs=NJOBS
    # )
    esd = ggl_instance.stack_signals(lens_table,
                                    # rand_table
                                     )
    esd.meta["NGAL"] = NGAL
    esd.meta["RV_MED"] = float(ggl_instance.Rv_mean_mpch)
    esd.write(out_dir / out_fmt.format(icosmo, ihod), overwrite=True)

if __name__ == "__main__":
    main()
