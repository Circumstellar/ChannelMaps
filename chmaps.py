import argparse
parser = argparse.ArgumentParser(description="Plot channel maps.")
parser.add_argument("file", help="FITS file containing the data to plot.")
parser.add_argument("nrows", type=int, help="The number of rows to plot.")
parser.add_argument("ncols", type=int, help="The numeber of columns to plot.")
parser.add_argument("--arcsec", type=float, default=2.0, help="Half-width of the image.")
parser.add_argument("--trimchan", type=int, nargs=2, help="Which channels to trim from the ends, if any. Indexed from 0, inclusive. Separated by WHITESPACE.")
args = parser.parse_args()

nrows = args.nrows
ncols = args.ncols

# Convert from arcsec to degrees
radius = args.arcsec/3600.

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import aplpy

import matplotlib
matplotlib.rc("contour", negative_linestyle="dashed")

# Read the number of channels
hdu_list = fits.open(args.file)
hdu = hdu_list[0]
nchan = hdu.header["NAXIS3"]

RA = hdu.header["OBSRA"]
DEC = hdu.header["OBSDEC"]

# Make sure we have the right number of spaces to plot
if args.trimchan:
    tlow, thigh = args.trimchan
    ntrim = tlow + 1 + nchan - thigh
    assert nrows * ncols >= (nchan - ntrim), "Must specify more axes (nrows * ncols) (={}) than number of channels ({})".format(nrows * ncols, nchan - ntrim)
    # numpy array is apparently ordered (nstokes, nchan, nx, ny)
    data = hdu.data[:,tlow:thigh, :, :]
else:
    assert nrows * ncols >= nchan, "Must specify more axes (nrows * ncols) (={}) than number of channels ({})".format(nrows * ncols, nchan)
    data = hdu.data[:,:,:]

# Determine vmin and vmax from all of the channels, excluding the trimmed ones
vmin = np.min(data)
vmax = np.max(data)

# Calculate the list of three-sigma contours to feed to the contour routine
rms   = 2 * 0.0075

levels = []
# Add contours from rms to vmin, then reverse
val = 0
while val > vmin:
    levels.append(val)
    val -= rms

# Reverse in place
levels.reverse()
val = rms
while val < vmax:
    levels.append(val)
    val += rms

print(levels)

hdu_list.close()

# Maybe, for now, just specify axes as a multiple of individual plot size.
ax_size = 1 # in
margin = 1 # in on all sides

fig_width = ax_size * ncols + 2 * margin # in
fig_height = ax_size * nrows + 2 * margin # in

dx = ax_size / fig_width
dy = ax_size / fig_height

# APLPy takes subplot parameters in terms of figure fractions. That means to keep
# plots square, we will have to adjust the amount by which we shift.
fig = plt.figure(figsize=(fig_width, fig_height))

chan_offset = 0

print("Plotting channel: ")

# Using DEC, determine the appropriate spacing for 1arcsec in RA
xspace = 2./np.cos(DEC * np.pi/180.)/3600

for row in range(nrows):
    for col in range(ncols):
        chan = row * ncols + col + chan_offset
        if chan <= tlow or chan >= thigh:
            chan_offset += 1
            chan += 1

        print(chan)

        xmin = (margin + ax_size * col)/fig_width
        ymin = (margin + ax_size * row)/fig_height
        splot = [xmin, ymin, dx, dy]

        f = aplpy.FITSFigure(args.file, figure=fig, dimensions=[0,1],
            slices=[chan, 0], subplot=splot)
        f.set_auto_refresh(False)
        f.show_colorscale(vmin=vmin, vmax=vmax)
        f.show_contour(args.file, dimensions=[0,1], slices=[chan, 0],
            levels=levels, colors="black", linewidths=0.2)
        f.recenter(RA, DEC, radius)


        f.ticks.set_xspacing(6/3600.) # degrees
        f.ticks.set_yspacing(2/3600.) # degrees
        

        if row == 0 and col == 0:
            f.tick_labels.set_font(size='xx-small')
            f.axis_labels.set_font(size='x-small')
        else:
            f.hide_xaxis_label()
            f.hide_yaxis_label()
            f.hide_xtick_labels()
            f.hide_ytick_labels()


fig.savefig("chmaps.pdf")
