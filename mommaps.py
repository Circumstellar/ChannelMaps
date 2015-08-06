from astropy.io import fits
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as LSC
import math
from common import get_coords, plot_beam, get_levels, get_geom_levels


matplotlib.rc("contour", negative_linestyle="dashed")
# matplotlib.rc("axes", linewidth=0.5)
# matplotlib.rc("xtick.major", size=2)
# matplotlib.rc("ytick.major", size=2)
from matplotlib.ticker import FormatStrFormatter as FSF
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MultipleLocator

hdu_list = fits.open("data/AKSco.12CO.display.mom1.fits")
hdu = hdu_list[0]
header = hdu.header

radius = 2.1/3600.
dict = get_coords(header, radius)
RA = 3600 * dict["RA"]
DEC = 3600 * dict["DEC"]
decl, decr = dict["DEC_slice"]
ral, rar = dict["RA_slice"]

ext = (RA[0], RA[-1], DEC[0], DEC[-1]) # [arcsec]

hdu_cont = fits.open("data/AKSco_band6cont_nat.image.fits")[0]
cont = hdu_cont.data[0,0]

mom_12CO_1 = hdu.data[0,0, decl:decr, ral:rar]

hdu_12CO_0 = fits.open("data/AKSco.12CO.display.mom0.fits")[0]
mom_12CO_0 = hdu_12CO_0.data[0,0] #,

hdu_list.close()
mom_13CO_1 = fits.open("data/AKSco.13CO.display.mom1.fits")[0].data[0,0, decl:decr, ral:rar]

hdu_13CO_0 = fits.open("data/AKSco.13CO.display.mom0.fits")[0]
mom_13CO_0 = hdu_13CO_0.data[0,0] #, decl:decr, ral:rar]

mom_C18O_1 = fits.open("data/AKSco.C18O.display.mom1.fits")[0].data[0,0, decl:decr, ral:rar]
hdu_C18O_0 = fits.open("data/AKSco.C18O.display.mom0.fits")[0]
mom_C18O_0 = hdu_C18O_0.data[0,0] #, decl:decr, ral:rar]

hdu_list = [hdu_cont, hdu_12CO_0, hdu_13CO_0, hdu_C18O_0]

# Before truncation, try to measure an average RMS away from the emission.
region=40
for data in [cont, mom_12CO_0, mom_13CO_0, mom_C18O_0]:
    print("RMS", np.std(data[0:region,0:region]))
    print("RMS", np.std(data[0:region,-region:-1]))
    print("RMS", np.std(data[-region:-1,-region:-1]))
    print("RMS", np.std(data[-region:-1,0:region]))
    print("Min", np.min(data))
    print("Max", np.max(data))
    print()

cont = cont[decl:decr, ral:rar]
mom_12CO_0 = mom_12CO_0[decl:decr, ral:rar]
mom_13CO_0 = mom_13CO_0[decl:decr, ral:rar]
mom_C18O_0 = mom_C18O_0[decl:decr, ral:rar]

for data in [mom_12CO_1, mom_13CO_1, mom_C18O_1]:
    print("Min", np.nanmin(data))
    print("Max", np.nanmax(data))
    print()

# Calculate the list of two-sigma contours to feed to the contour routine
rms = 0.018

all_data = np.concatenate((mom_12CO_0, mom_13CO_0))
vmin = np.min(all_data)
vmax = np.max(all_data)
levels_12CO = get_levels(rms, vmin, vmax, 10)
levels_13CO = get_levels(rms, vmin, vmax)
levels_C18O = get_levels(0.01, vmin, vmax)

rms_cont = 4.39435e-05
vmin = np.min(cont)
vmax = np.max(cont)
levels_cont = get_geom_levels(rms_cont, vmin, vmax, 2)

fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(6.5, 2.2))

vvmax = np.max(np.abs(cont))
# norm = matplotlib.colors.Normalize(-vvmax, vvmax)
norm = matplotlib.colors.SymLogNorm(linthresh=3 * 5.0e-5, linscale=0.0, vmin=-vvmax, vmax=vvmax)
ax[0].imshow(-cont, origin="lower", interpolation="none", extent=ext, cmap="bwr", norm=norm)
ax[0].contour(cont, origin="lower", extent=ext, linewidths=0.4, colors="black", levels=levels_cont)

vsys = 5.49

# vvmax = np.nanmax(np.abs(np.concatenate((mom_12CO_1, mom_13CO_1, mom_C18O_1))) - vsys)
# Use the same range within each moment map
vvmax = np.nanmax(np.abs(mom_12CO_1 - vsys))
norm = matplotlib.colors.Normalize(-vvmax + vsys, vvmax + vsys)
print("12CO", -vvmax + vsys, vvmax + vsys)

ax[1].imshow(mom_12CO_1, origin="lower", interpolation="none", extent=ext, cmap="bwr", norm=norm)
ax[1].contour(mom_12CO_0, origin="lower", extent=ext, linewidths=0.4, colors="black", levels=levels_12CO)

vvmax = np.nanmax(np.abs(mom_13CO_1 - vsys))
norm = matplotlib.colors.Normalize(-vvmax + vsys, vvmax + vsys)
ax[2].imshow(mom_13CO_1, origin="lower", interpolation="none", extent=ext, cmap="bwr", norm=norm)
ax[2].contour(mom_13CO_0, origin="lower", extent=ext, linewidths=0.4, colors="black", levels=levels_13CO)

vvmax = np.nanmax(np.abs(mom_C18O_1 - vsys))
norm = matplotlib.colors.Normalize(-vvmax + vsys, vvmax + vsys)
ax[3].imshow(mom_C18O_1, origin="lower", interpolation="none", extent=ext, cmap="bwr", norm=norm)
ax[3].contour(mom_C18O_0, origin="lower", extent=ext, linewidths=0.4, colors="black", levels=levels_C18O)

labels = [r"continuum", r"${}^{12}$CO", r"${}^{13}$CO", r"C${}^{18}$O"]
alabels = ["(a)", "(b)", "(c)", "(d)"]

for a, label, alabel, hdu in zip(ax, labels, alabels, hdu_list):
    a.xaxis.set_major_formatter(FSF("%.0f"))
    a.yaxis.set_major_formatter(FSF("%.0f"))
    a.xaxis.set_major_locator(MultipleLocator(1.))
    a.yaxis.set_major_locator(MultipleLocator(1.))
    a.annotate(label, (0.95, 0.05), xycoords="axes fraction", size=6, ha="right")
    a.annotate(alabel, (0.05, 0.9), xycoords="axes fraction", size=6)
    plot_beam(a, hdu.header, xy=(1.55,-1.55))

for a in ax[1:]:
    a.xaxis.set_ticklabels([])
    a.yaxis.set_ticklabels([])

ax[0].set_xlabel(r"$\Delta \alpha$ [${}^{\prime\prime}$]", fontsize=8)
ax[0].set_ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]", fontsize=8)
ax[0].tick_params(axis='both', which='major', labelsize=8)


fig.subplots_adjust(left=0.08, right=0.92, top=1.0, bottom=0.00, wspace=0.0)
fig.savefig("mom.pdf")
fig.savefig("mom.svg")
fig.savefig("mom.png")
