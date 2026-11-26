"""Measure FastPM mock void lensing for a single simulation.

Run with: python run_vl_fastpm_single.py
Only cosmo_000001 is processed, using the boss_lowz_e2_e3 north 3-tomo shape
catalog for all 10 HOD samples. The radial scale is the median Rv of the
selected, footprint-matched voids, as implemented by GGLCalculator (not a
separate radial scale for each void).
"""

from pathlib import Path

import numpy as np
from loguru import logger

from container import FastPMSurveyData
from calculator import GGLCalculator, GGLConfig


NJOBS = 12
NGAL = 3.5e-4  # Fixed tracer number density in (h/Mpc)^3 for all samples.
ICOSMO = 0  # The only FastPM simulation processed by this script.
IHOD = 0
IROT = 0

wdir = Path(__file__).resolve().parent
data_dirbase = Path("/Users/suqikuai777/Dataspace/FastPM/MockCatalogs")
lens_dir = data_dirbase / "Voids"
lens_fmt = "cosmo_{:04d}_run0_HOD{:d}_allrots_boss_match_kids1000.fits"

srcs_dir = data_dirbase / "Shapes"
srcs_fmt = "cosmo_{:04d}_run_0000_kids1000_3tomos_allrots.fits"
# srcs_fmt = "kids1000_north_3tomos/cosmo_{:06d}_realization_0002_kids1000_north_3tomos.fits"

RAND = True
rand_dir = data_dirbase / "Rand"
rand_fmt = "bossngc_2dflens_z0.2_0.4_match_kids1000_rots.fits"

# rvmin = 15
# rvmax = 17.5
rvmins = [15.0, 17.5, 20.5]
rvmaxs = [17.5, 20.5, 25.0]

wSN = True

out_dir = wdir / "results/vl/bosslowz_2dflens_kids1000"
out_fmt = "cosmo{:04d}_HOD{:d}_rot{:d}_vl_3tomos_rv{:.1f}{:.1f}"
if RAND:
    out_fmt += "_wrand"
if wSN:
    out_fmt += "_wSN"
out_fmt += ".fits"

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
        rp_max=2.9,
        rp_bins=14,
        rp_unit="Rv",
        bin_type="linear",
        flip_g1=False,
        flip_g2=True,  # Same FastPM shear convention as run_ggl_fastpm.py.
        wRSD=False,  # Void catalogs already contain the void-center redshift.
        wSN=wSN,
        wPhZ=False,
        njk=128
    )
    ggl_instance = GGLCalculator(config=ggl_config)
    out_dir.mkdir(parents=True, exist_ok=True)

    icosmo = ICOSMO
    cosmo_dict = cosmo_params[icosmo]

    for (rvmin, rvmax) in zip(rvmins, rvmaxs):

        for irot in range(1,2):

            Flag_rand_calc = False

            logger.info("Load shape catalog")
            
            mock_shape = FastPMSurveyData.load_shape(srcs_dir / srcs_fmt.format(icosmo), irot=irot)
            if len(mock_shape) == 0:
                raise ValueError(f"Empty shape catalog for cosmology {icosmo}")

            srcs_table = ggl_instance.mk_srcs_cat(mock_shape)
            
            for ihod in range(0,1):

                logger.info("Load lens catalog")
                
                mock_void = FastPMSurveyData.load_void(lens_dir / lens_fmt.format(icosmo, ihod), irot=irot)

                # mock_void = mock_void[((mock_void.Rv > 15.0) & (mock_void.Rv < 25.0))]
                # print(np.quantile(mock_void.Rv, [1./3., 2./3.]))

                mock_void_boss_curr = mock_void[((mock_void.Rv) > rvmin) & (mock_void.Rv < rvmax)]
                ds_rate = len(mock_void_boss_curr) / len(mock_void)

                mock_void_boss_curr = mock_void_boss_curr
                # mock_void_boss_matched = mock_void_boss_curr.match_to_reference(
                #     mock_shape, nside=256, in_place=False
                # )
                mock_void_boss_matched = mock_void_boss_curr

                if len(mock_void_boss_matched) == 0:
                    raise ValueError(f"No selected voids overlap the shapes: cosmo={icosmo}, HOD={ihod}")

                lens_table = ggl_instance.mk_lens_cat(mock_void_boss_matched)
                rvmean = ggl_instance.get_Rv_mean_mpch(mock_void_boss_matched)

                if RAND:
                    if not Flag_rand_calc:
                        logger.info("Load random catalog")

                        mock_rand = FastPMSurveyData.load_rand(rand_dir / rand_fmt, irot=irot)
                        Nrand_target = int(len(mock_rand)*ds_rate)
                        mock_rand = mock_rand[np.random.choice(np.arange(len(mock_rand)), Nrand_target)]
                        rand_table = ggl_instance.mk_lens_cat(mock_rand)

                        rand_table = ggl_instance.compute_pairs(
                            cosmo_dict, rand_table, srcs_table, n_jobs=NJOBS
                        )
                        Flag_rand_calc = True
                    else:
                        pass

                else:
                    rand_table = None
            
                # print(rvmean)
                # Rv (and hence physical bin edges) changes with each void sample.
                # Start from fresh randoms and recompute pairs for every HOD.

                logger.info("compute pairs")

                lens_table = ggl_instance.compute_pairs(
                    cosmo_dict, lens_table, srcs_table, n_jobs=NJOBS
                )

                esd = ggl_instance.stack_signals(lens_table,
                                                rand_table
                                                )
                esd.meta["NGAL"] = NGAL
                esd.meta["RV_MED"] = float(ggl_instance.Rv_mean_mpch)

                jk_cov = ggl_instance.estimate_jackknife_cov(lens_table,
                                                            rand_table,
                                                            )
                esd = ggl_instance.add_errorbar(esd, jk_cov)

                esd.write(out_dir / out_fmt.format(icosmo, ihod, irot, rvmin, rvmax), overwrite=True)

if __name__ == "__main__":
    main()
