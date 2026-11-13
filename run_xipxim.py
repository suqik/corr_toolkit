from container import *
from calculator import *
import numpy as np
from loguru import logger

if __name__ == "__main__":
    import os
    os.environ['OMP_NUM_THREADS'] = '8'

    ddir = "/Users/suqikuai777/Dataspace/FastPM/MockCatalogs/Shape"

    logger.info("Load shape catalog")

    shapecat = SurveyData.load_cosmogrid_shape(f"{ddir}/cosmo_000000_realization_0002_boss_lowz_e2_e3_north_3tomos.fits")

    shapecat = shapecat[shapecat.tomo == 3]

    cs_config = Ang2PTConfig(
        theta_min=10.0,
        theta_max=1000.0,
        theta_bins=15,
        theta_unit='arcmin',
        bin_type='log',
        flip_g1=False,
        flip_g2=True
    )

    cs_instance = Ang2PTCalculator(config=cs_config)

    tc_shape = cs_instance.mk_GG_cat(shapecat)

    logger.info("Measuring shear-shear tpcf")

    rnom, xip, xim, varxip, varxim = cs_instance.compute_gg_auto(tc_shape)

    logger.info("Save to file")

    np.savetxt("test_xipxim.txt", np.c_[rnom, xip, xim, np.sqrt(varxip), np.sqrt(varxim)])