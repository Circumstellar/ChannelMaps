
from astropy.io import fits
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

matplotlib.rc("contour", negative_linestyle="dashed")
matplotlib.rc("axes", linewidth=0.5)
matplotlib.rc("xtick.major", size=2)
matplotlib.rc("ytick.major", size=2)
from matplotlib.ticker import FormatStrFormatter as FSF
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Ellipse

# Read the number of channels
hdu_list = fits.open("data.fits")
hdu = hdu_list[0]
header = hdu.header
nx = header["NAXIS1"]
ny = header["NAXIS2"]
nchan = header["NAXIS3"]

assert nx % 2 == 0 and ny % 2 == 0 , "We don't have an even number of pixels, assumptions in the routine are violated."

# Midpoints
mx = nx/2
my = ny/2

# Phase center in decimal degrees
RA = header["CRVAL1"]
DEC = header["CRVAL2"]

# from: http://tdc-www.harvard.edu/wcstools/wcstools.wcs.html
# CRPIX1 and CRPIX2 are the pixel coordinates of the reference point to which
# the projection and rotation refer, i.e, integers
# Indexed from 1, I'm assuming
# If the image is 256 x 256, CRPIX = 129, then it must be indexed from 1.

# CRVAL1 and CRVAL2 give the center coordinate as right ascension or longitude
# in decimal degrees.

# CDELT1 and CDELT2 indicate the plate scale in degrees per pixel
# This means that to get proper RA, DEC coordinates you would need to add a
# cos(DEC) in there or use some WCS tools. But since we just care about offsets,
# this should be fine


xx = np.arange(header["NAXIS1"])
yy = np.arange(header["NAXIS2"])

# RA coordinates
CDELT1 = header["CDELT1"]
CRPIX1 = header["CRPIX1"] - 1. # Now indexed from 0

# DEC coordinates
CDELT2 = header["CDELT2"]
CRPIX2 = header["CRPIX2"] - 1. # Now indexed from 0

# Lay down relative coordinates spanning the image
# Note that RAs actually goes in decreasing order because of sky convention
dRAs = (np.arange(nx) - nx/2) * CDELT1
dDECs = (np.arange(ny) - ny/2) * CDELT2

# Truncation radius
radius = 2/3600.

# Because DEC increases, this means that we should be using imshow with origin="lower" set.

# Determine the frequencies. Goes in increasing order.
freq = header["CRVAL3"] + np.arange(header["NAXIS3"]) * header["CDELT3"]
print(freq)

c_kms = 2.99792458e5 # [km s^-1]
f0 = 230.538e9 # [Hz]
# f0 = header["RESTFRQ"] # [Hz]

# Convert all frequencies to velocity [km/s]
vs = c_kms * (f0 - freq)/f0

# Subtract the systemic velocity (in this case subtract a negative)
# vs -= -26.09

# Read the beam size and position angle from the header
BMAJ = 3600. * header["BMAJ"] # [arcsec]
BMIN = 3600. * header["BMIN"] # [arcsec]
BPA =  header["BPA"] # degrees East of North

# If the dataset is 82 channels big, trim 11 from either side to get 60 channels
triml = 11
trimr = 11

data = hdu.data[0,triml:-trimr]
model_hdu_list = fits.open("model.fits")
model = model_hdu_list[0].data[0, triml:-trimr]
model_hdu_list.close()

resid_hdu_list = fits.open("resid.fits")
resid = resid_hdu_list[0].data[0, triml:-trimr]
resid_hdu_list.close()

vs = vs[triml:-trimr]

# Reverse the channels so blueshifted channels appear first
data = data[::-1]
model = model[::-1]
resid = resid[::-1]
vs = vs[::-1]


# Data is now (nchan, ny, nx)
# Assuming BSCALE=1.0 and BZERO=0.0

# Alternatively
slice_RA = math.floor(math.fabs(radius/CDELT1))
slice_DEC = math.floor(math.fabs(radius/CDELT2))

assert slice_RA < mx and slice_DEC < my, "Selected radius must be smaller than half size of image."

# print(np.sum(ind_DEC))
# print(np.sum(ind_RA))
# We want to shift to phase center (with an offset) and also truncate the
# RA and DEC which are outside of our range.
# For some reason, truncating both dimensions at the same time doesn't work properly.
# so we have to do them separately
# data = data[:, ind_DEC, :]
# data = data[:, :, ind_RA]
data = data[:, mx-slice_DEC:mx+slice_DEC, mx-slice_RA:mx+slice_RA]
model = model[:, mx-slice_DEC:mx+slice_DEC, mx-slice_RA:mx+slice_RA]
resid = resid[:, mx-slice_DEC:mx+slice_DEC, mx-slice_RA:mx+slice_RA]

dRAs = 3600 * dRAs[mx-slice_RA:mx+slice_RA] # [arcsec]
dDECs = 3600 * dDECs[mx-slice_DEC:mx+slice_DEC] # [arcsec]

ext = (dRAs[0], dRAs[-1], dDECs[0], dDECs[-1]) # [arcsec]

# Determine vmin and vmax from all of the channels, all of the datasets, excluding the trimmed ones
all_data = np.concatenate((data, model, resid))

vmin = np.min(all_data)
vmax = np.max(all_data)

vvmax = np.max(np.abs(all_data))

norm = matplotlib.colors.Normalize(-vvmax, vvmax)

# Calculate the list of three-sigma contours to feed to the contour routine
rms   = 3 * 0.00425

levels = []
# Add contours from rms to vmin, then reverse
# We don't want a 0-level contour
val = -rms
while val > vmin:
    levels.append(val)
    val -= rms

# Reverse in place
levels.reverse()
val = rms
while val < vmax:
    levels.append(val)
    val += rms

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

#
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

            # Plot image
            ax.imshow(data[chan], cmap="RdBu", norm=norm, origin="lower", extent=ext)

            # Plot contours
            ax.contour(data[chan], origin="lower", levels=levels, linewidths=0.2, colors="black", extent=ext)

            # Annotate the velocity
            ax.annotate("{:.1f}".format(vs[chan]), (0.1, 0.8), xycoords="axes fraction", size=5)
            if row == 0 and col == 0 and panel == 0:
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
                ax.add_artist(Ellipse(xy=(1, -1), width=BMIN, height=BMAJ, angle=BPA,
                    facecolor="0.8", linewidth=0.2))

            if row == (nrows - 1) and col == 0 and panel==2:
                # Actually create axis labels
                ax.set_xlabel(r"$\Delta \alpha$ ['']")
                ax.set_ylabel(r"$\Delta \delta$ ['']")
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
