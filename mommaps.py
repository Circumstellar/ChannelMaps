from astropy.io import fits
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from common import get_coords


# matplotlib.rc("contour", negative_linestyle="dashed")
# matplotlib.rc("axes", linewidth=0.5)
# matplotlib.rc("xtick.major", size=2)
# matplotlib.rc("ytick.major", size=2)
from matplotlib.ticker import FormatStrFormatter as FSF
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MultipleLocator


# AKSco.12CO.display.image.fits   AKSco.C18O.display.image.fits
# AKSco.12CO.display.mom0.fits    AKSco.C18O.display.mom0.fits
# AKSco.12CO.display.mom1.fits    AKSco.C18O.display.mom1.fits
# AKSco.13CO.display.image.fits   model.12CO.display.image.fits
# AKSco.13CO.display.mom0.fits    Moment Maps.ipynb
# AKSco.13CO.display.mom1.fits    resid.12CO.display.image.fits
# AKSco_band6cont_nat.image.fits


hdu_list = fits.open("data/AKSco.12CO.display.mom1.fits")
hdu = hdu_list[0]
header = hdu.header

radius = 2/3600.
dict = get_coords(header, radius)
RA = 3600 * dict["RA"]
DEC = 3600 * dict["DEC"]
decl, decr = dict["DEC_slice"]
ral, rar = dict["RA_slice"]

ext = (RA[0], RA[-1], DEC[0], DEC[-1]) # [arcsec]

cont = fits.open("data/AKSco_band6cont_nat.image.fits")[0].data[0,0, decl:decr, ral:rar]

mom_12CO_1 = hdu.data[0,0, decl:decr, ral:rar]
mom_12CO_0 = fits.open("data/AKSco.12CO.display.mom0.fits")[0].data[0,0, decl:decr, ral:rar]

hdu_list.close()
mom_13CO_1 = fits.open("data/AKSco.13CO.display.mom1.fits")[0].data[0,0, decl:decr, ral:rar]
mom_13CO_0 = fits.open("data/AKSco.13CO.display.mom0.fits")[0].data[0,0, decl:decr, ral:rar]

mom_C18O_1 = fits.open("data/AKSco.C18O.display.mom1.fits")[0].data[0,0, decl:decr, ral:rar]
mom_C18O_0 = fits.open("data/AKSco.C18O.display.mom0.fits")[0].data[0,0, decl:decr, ral:rar]

fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(6.5, 2.5))

ax[0].imshow(cont, origin="lower", interpolation="none", extent=ext)


ax[1].imshow(mom_12CO_1, origin="lower", interpolation="none", extent=ext)
ax[1].contour(mom_12CO_0, origin="lower", extent=ext, linewidths=0.2, colors="black")

ax[2].imshow(mom_13CO_1, origin="lower", interpolation="none", extent=ext)
ax[2].contour(mom_13CO_0, origin="lower", extent=ext, linewidths=0.2, colors="black")

ax[3].imshow(mom_C18O_1, origin="lower", interpolation="none", extent=ext)
ax[3].contour(mom_C18O_0, origin="lower", extent=ext, linewidths=0.2, colors="black")

labels = [r"cont", r"${}^{12}$CO", r"${}^{13}$CO", r"C${}^{18}$O"]

for a, label in zip(ax, labels):
    a.xaxis.set_major_formatter(FSF("%.0f"))
    a.yaxis.set_major_formatter(FSF("%.0f"))
    a.xaxis.set_major_locator(MultipleLocator(1.))
    a.yaxis.set_major_locator(MultipleLocator(1.))
    a.annotate(label, (0.1, 0.9), xycoords="axes fraction", size=6)

for a in ax[1:]:
    a.xaxis.set_ticklabels([])
    a.yaxis.set_ticklabels([])

ax[0].set_xlabel(r"$\Delta \alpha$ ['']")
ax[0].set_ylabel(r"$\Delta \delta$ ['']")
ax[0].tick_params(axis='both', which='major', labelsize=8)


fig.subplots_adjust(left=0.1, right=0.9, top=0.97, wspace=0.0)
fig.savefig("mom.png")