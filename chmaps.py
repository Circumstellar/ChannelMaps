from astropy.io import fits
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from common import get_coords, plot_beam, get_levels, make_cmap

matplotlib.rc("contour", negative_linestyle="dashed")
matplotlib.rc("axes", linewidth=0.5)
matplotlib.rc("xtick.major", size=2)
matplotlib.rc("ytick.major", size=2)
from matplotlib.ticker import FormatStrFormatter as FSF
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MultipleLocator

# Read the number of channels
hdu_list = fits.open("data/AKSco.12CO.display.image.fits")
hdu = hdu_list[0]
header = hdu.header

# Because DEC increases, this means that we should be using imshow with origin="lower" set.

# Determine the frequencies. Goes in increasing order.
# freq = header["CRVAL3"] + np.arange(header["NAXIS3"]) * header["CDELT3"]
# print(freq)
#
# c_kms = 2.99792458e5 # [km s^-1]
# f0 = 230.538e9 # [Hz]
# # f0 = header["RESTFRQ"] # [Hz]
#
# # Convert all frequencies to velocity [km/s]
# vs = c_kms * (f0 - freq)/f0

vs = vels = (np.arange(header["NAXIS3"])*header["CDELT3"] + header["CRVAL3"]) * 1e-3 # [km/s]

# Using the systemic velocity, normalize these to the interval [0, 1], with 0.5 being the middle.
vsys = 5.49
v_cent = vs - vsys
vrange = np.max(np.abs(v_cent))
vel_min = -vrange
vel_max = vrange

print("vrange", vrange)

vel_fracs = (v_cent - vel_min)/(2 * vrange)

print(vel_fracs)

data = hdu.data[0] #,triml:-trimr]
model_hdu_list = fits.open("data/model.12CO.display.image.fits")
model = model_hdu_list[0].data[0] # , triml:-trimr]
model_hdu_list.close()

resid_hdu_list = fits.open("data/resid.12CO.display.image.fits")
resid = resid_hdu_list[0].data[0] #, triml:-trimr]
resid_hdu_list.close()

# vs = vs[triml:-trimr]

# Reverse the channels so blueshifted channels appear first
# data = data[::-1]
# model = model[::-1]
# resid = resid[::-1]
# vs = vs[::-1]


# Data is now (nchan, ny, nx)
# Assuming BSCALE=1.0 and BZERO=0.0

radius = 2.1/3600.
dict = get_coords(header, radius)
RA = 3600 * dict["RA"]
DEC = 3600 * dict["DEC"]
decl, decr = dict["DEC_slice"]
ral, rar = dict["RA_slice"]

ext = (RA[0], RA[-1], DEC[0], DEC[-1]) # [arcsec]

# print(np.sum(ind_DEC))
# print(np.sum(ind_RA))
# We want to shift to phase center (with an offset) and also truncate the
# RA and DEC which are outside of our range.
# For some reason, truncating both dimensions at the same time doesn't work properly.
# so we have to do them separately
# data = data[:, ind_DEC, :]
# data = data[:, :, ind_RA]
# data = data[:, mx-slice_DEC:mx+slice_DEC, mx-slice_RA:mx+slice_RA]
# model = model[:, mx-slice_DEC:mx+slice_DEC, mx-slice_RA:mx+slice_RA]
# resid = resid[:, mx-slice_DEC:mx+slice_DEC, mx-slice_RA:mx+slice_RA]

# Before truncation, try to measure an average RMS away from the emission.
region=40
print("RMS", np.std(data[:,0:region,0:region]))
print("RMS", np.std(data[:,0:region,-region:-1]))
print("RMS", np.std(data[:,-region:-1,-region:-1]))
print("RMS", np.std(data[:,-region:-1,0:region]))

data = data[:, decl:decr, ral:rar]
model = model[:, decl:decr, ral:rar]
resid = resid[:, decl:decr, ral:rar]

# Determine vmin and vmax from all of the channels, all of the datasets, excluding the trimmed ones
all_data = np.concatenate((data, model, resid))

vmin = np.min(all_data)
vmax = np.max(all_data)

vvmax = np.max(np.abs(all_data))

norm = matplotlib.colors.Normalize(-vvmax, vvmax)

# Calculate the list of three-sigma contours to feed to the contour routine
# rms   = 3 * 0.00373
rms = 0.0045
levels = get_levels(rms, vmin, vmax)
print("3 sigma contour levels:", levels)

hdu_list.close()

ax_size = 5.5/12 # in
margin = 0.5 # in. sides and bottom
middle_margin = 0.1 # separating panels
top_margin = 0.1
nrows = 5
ncols = 12

panel_width = ax_size * ncols # in
panel_height = ax_size * nrows # in

fig_width = panel_width + 2 * margin # in
fig_height = 3 * panel_height + 2 * middle_margin + top_margin + margin# in

dx = ax_size / fig_width
dy = ax_size / fig_height

fig = plt.figure(figsize=(fig_width, fig_height))

incl = 109.5 # deg
PA = 141.4 # deg

major = 2 # [arcsec]
minor = major * math.cos(incl * np.pi/180.)
slope = math.tan(PA * np.pi/180.)

x1s = np.linspace(-minor/2 * math.cos(PA * np.pi/180.), minor/2 * math.cos(PA * np.pi/180.))
y1s = x1s * slope

x2s = np.linspace(-major/2 * math.sin(PA * np.pi/180.), major/2 * math.sin(PA * np.pi/180.))
y2s = -x2s / slope

def plot(data, panel):
    '''
    data being the (nchan, ny, nx) dataset

    panel being the top, middle, or bottom (0, 1, or 2)
    '''
    panel_offset = (middle_margin + panel_height) * panel / fig_height # [figure fraction]

    for row in range(nrows):
        for col in range(ncols):
            chan = row * ncols + col

            xmin = (margin + ax_size * col)/fig_width
            ymin = 1.0 - ((top_margin + ax_size * (row + 1))/fig_height + panel_offset)
            rect = [xmin, ymin, dx, dy]

            ax = fig.add_axes(rect)

            cmap = make_cmap(vel_fracs[chan])

            # Plot image
            ax.imshow(data[chan], cmap=cmap, norm=norm, origin="lower", extent=ext)

            # Plot contours
            ax.contour(data[chan], origin="lower", levels=levels, linewidths=0.2, colors="black", extent=ext)

            # Annotate the velocity
            ax.annotate("{:.1f}".format(vs[chan]), (0.1, 0.8), xycoords="axes fraction", size=5)
            if row == 0 and col == 0 and panel==0:
                ax.annotate(r"$\textrm{km s}^{-1}$", (0.15, 0.6), xycoords="axes fraction", size=5)

            # Set ticks for every arcsec
            ax.xaxis.set_major_formatter(FSF("%.0f"))
            ax.yaxis.set_major_formatter(FSF("%.0f"))
            ax.xaxis.set_major_locator(MultipleLocator(1.))
            ax.yaxis.set_major_locator(MultipleLocator(1.))

            # Plot the crosses
            ax.plot(x1s, y1s, lw=0.1, ls=":", color="0.5")
            ax.plot(x2s, y2s, lw=0.1, ls=":", color="0.5")

            # Plot the beam
            if row == (nrows - 1) and col == 0:
                plot_beam(ax, header)
                # ax.add_artist(Ellipse(xy=(1, -1), width=BMIN, height=BMAJ, angle=BPA,
                    # facecolor="0.8", linewidth=0.2))

            if row == (nrows - 1) and col == 0 and panel==2:
                # Actually create axis labels
                ax.set_xlabel(r"$\Delta \alpha$ [${}^{\prime\prime}$]", fontsize=8)
                ax.set_ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]", fontsize=8)

                ax.tick_params(axis='both', which='major', labelsize=8)

            else:
                # Hide axis label and tick labels
                ax.xaxis.set_ticklabels([])
                ax.yaxis.set_ticklabels([])


plot(data, 0)
plot(model, 1)
plot(resid, 2)

left = 0.03
fig.text(left, 0.85, "data", rotation="vertical")
fig.text(left, 0.54, "model", rotation="vertical")
fig.text(left, 0.24, "residual", rotation="vertical")

# Add the side labels

fig.savefig("chmaps.pdf")
fig.savefig("chmaps.svg")
