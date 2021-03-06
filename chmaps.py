#!/usr/bin/env python

# Reads everything from a config.yaml file located in the same directory.

import argparse

parser = argparse.ArgumentParser(description="Plot channel maps.")
parser.add_argument("--config", default="config.yaml", help="The configuration file specifying defaults.")
parser.add_argument("--measure", action="store_true", help="Just measure the basic properties, like the RMS and number of channels, so that you may add them into the config.yaml file.")
parser.add_argument("--fmt", default="pdf", help="What file format to save the maps in.")
args = parser.parse_args()

import yaml
f = open(args.config)
config = yaml.load(f)
f.close()

# Do the measureing operations here
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import chmaps_common as common

matplotlib.rc("contour", negative_linestyle="dashed")
matplotlib.rc("axes", linewidth=0.5)
matplotlib.rc("xtick.major", size=2)
matplotlib.rc("ytick.major", size=2)
from matplotlib.ticker import FormatStrFormatter as FSF
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MultipleLocator
import matplotlib.patheffects as path_effects

if config["telescope"] == "ALMA":
    readfn = common.read_ALMA
elif config["telescope"] == "SMA":
    readfn = common.read_SMA

if args.measure:
    # Use the dataset to actually measure quantities like the maximum intensity and RMS
    # so that they can be replicated for the other types of measurements and put on the same scale.
    vs, data, header = readfn(config["data"])

    # Now, think about truncating the channels
    low = config["trim"]["blue"]
    high = config["trim"]["red"]

    if high != 0:
        vs = vs[low:-high]
        data = data[low:-high]
    else:
        vs = vs[low:]
        data = data[low:]

    print("Data shape", data.shape)

    print("Average velocity", np.average(vs))

    # Before truncation, try to measure an average RMS away from the emission.
    region = config["RMS_region"]

    print("RMS", np.std(data[:,0:region,0:region]), "Jy/beam")
    print("RMS", np.std(data[:,0:region,-region:-1]), "Jy/beam")
    print("RMS", np.std(data[:,-region:-1,-region:-1]), "Jy/beam")
    print("RMS", np.std(data[:,-region:-1,0:region]), "Jy/beam")

    # If we have the data, model, and residual specified, try determining the scaling factors from all of the channels, all of the datasets
    # else, just use the data
    try:
        all_data = []
        for kw in ["data", "model", "resid"]:
            fname = config[kw]
            vs, data, header = readfn(fname)
            all_data.append(data)
        all_data = np.concatenate(all_data)
    except FileNotFoundError as e:
        vs, data, header = readfn(config["data"])
        all_data = data

    vmin = np.min(all_data)
    vmax = np.max(all_data)
    vvmax = np.max(np.abs(all_data))
    print("vvmax {:.4f}, vmin {:.4f}, vmax {:.4f}".format(vvmax, vmin, vmax))

    import sys
    sys.exit()

ax_size = 1.0 # in
margin = 0.5 # in. sides and bottom
nrows = config["nrows"]
ncols = config["ncols"]
radius = config["radius"]/3600. # [degrees]
rms = config["RMS"]

panel_width = ax_size * ncols # in
panel_height = ax_size * nrows # in

fig_width = panel_width + 2 * margin # in
fig_height = panel_height + 2 * margin # in

dx = ax_size / fig_width
dy = ax_size / fig_height

vvmax = config["vvmax"]
norm = matplotlib.colors.Normalize(-vvmax, vvmax)

cmap = plt.get_cmap("RdBu")

mu_RA = config["mu_RA"]
mu_DEC = config["mu_DEC"]

# If the ellipse is specified, read it and plot it
crosshairs = config.get("crosshairs", None)
if crosshairs:
    a = crosshairs["major"]
    incl = crosshairs["incl"]
    PA = crosshairs["PA"]


    b = a * math.cos(incl * np.pi/180.)

    slope = math.tan(PA * np.pi/180.)
    print("slope", slope)

    # Because the crosshairs will be plotted on the channel map, where RA runs positive to the
    # left, we will use a separate coordinate for RA plotting.

    # if (PA > 0) and (PA < 90):
    x_a = np.linspace(-a * math.cos(PA * np.pi/180.), a * math.cos(PA * np.pi/180.))

    RA_a = -x_a
    DEC_a = x_a * slope

    x_b = np.linspace(-b * math.sin(PA * np.pi/180.), b * math.sin(PA * np.pi/180.))
    RA_b = -x_b
    DEC_b = -x_b / slope


def plot_maps(fits_name, fname):
    try:
        vs, data, header = readfn(fits_name)
    except FileNotFoundError as e:
        print("Cannot load {}, continuing.".format(fits_name))
        return

    if vs[-1] < vs[0]:
        print("Reversing velocity to be increasing.")
        vs = vs[::-1]
        data = data[::-1]

    # Now, think about truncating the channels
    low = config["trim"]["blue"]
    high = config["trim"]["red"]

    if high != 0:
        vs = vs[low:-high]
        data = data[low:-high]
    else:
        vs = vs[low:]
        data = data[low:]

    nchan = data.shape[0]

    dict = common.get_coords(data, header, radius, mu_RA, mu_DEC)
    RA = 3600 * dict["RA"] # [arcsec]
    DEC = 3600 * dict["DEC"] # [arcsec]
    decl, decr = dict["DEC_slice"]
    ral, rar = dict["RA_slice"]
    data = dict["data"]

    ext = (RA[0], RA[-1], DEC[0], DEC[-1]) # [arcsec]

    print("Extent", ext)
    # data = data[:, decl:decr, ral:rar]

    # Using the systemic velocity, normalize  these to the interval [0, 1], with 0.5 being the middle corresponding to the systemic velocity. Note that in this case, it is not likely that v_min = 0.0 && v_max = 1.0. Either v_min or v_max will be greater than 0.0 or less than 1.0, respectively, unless the systemic velocity is perfectly centered.
    vsys = config["vsys"]
    v_cent = vs - vsys
    vrange = np.max(np.abs(v_cent))

    vel_min = -vrange
    vel_max = vrange
    vel_fracs = 1 - (v_cent - vel_min)/(2 * vrange)

    # Get contour levels specific to this data/model/resid set
    vmin = np.min(data)
    vmax = np.max(data)
    levels = common.get_levels(rms, vmin, vmax)
    # print("3 sigma contour levels:", levels)

    fig = plt.figure(figsize=(fig_width, fig_height))

    for row in range(nrows):
        for col in range(ncols):
            chan = row * ncols + col

            if chan >= nchan:
                continue

            xmin = (margin + ax_size * col)/fig_width
            ymin = 1.0 - ((margin + ax_size * (row + 1))/fig_height)
            rect = [xmin, ymin, dx, dy]

            ax = fig.add_axes(rect)

            # cmap = make_cmap(vel_fracs[chan])
            # print("Color", cmap(vel_fracs[chan]))

            # Plot image
            ax.imshow(data[chan], cmap=cmap, norm=norm, origin="lower", extent=ext)

            # Plot contours
            ax.contour(data[chan], origin="lower", levels=levels, linewidths=0.2, colors="black", extent=ext)

            if crosshairs:
                ax.plot(RA_a, DEC_a, lw=0.2, ls=":", color="0.2")
                ax.plot(RA_b, DEC_b, lw=0.2, ls="-", color="0.2")

            # Annotate the velocity
            text = ax.annotate("{:.1f}".format(vs[chan]), (0.1, 0.8), xycoords="axes fraction", size=5, color=cmap(vel_fracs[chan]))
            text.set_path_effects([path_effects.Stroke(linewidth=0.2, foreground='black'),
                       path_effects.Normal()])
            if row == 0 and col == 0:
                ax.annotate(r"$\textrm{km s}^{-1}$", (0.15, 0.6), xycoords="axes fraction", size=5)


            # Plot the beam
            if row == (nrows - 1) and col == 0:
                common.plot_beam(ax, header, xy=(0.75 * 3600 * radius,-0.75 * 3600 * radius))

                # Actually create axis labels
                ax.set_xlabel(r"$\Delta \alpha$ [${}^{\prime\prime}$]", fontsize=8)
                ax.set_ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]", fontsize=8)

                ax.tick_params(axis='both', which='major', labelsize=8)
                #
                # ax.xaxis.set_major_formatter(FSF("%.0f"))
                # ax.yaxis.set_major_formatter(FSF("%.0f"))
                # ax.xaxis.set_major_locator(MultipleLocator(1.))
                # ax.yaxis.set_major_locator(MultipleLocator(1.))

            else:
                # Hide axis label and tick labels
                ax.xaxis.set_ticklabels([])
                ax.yaxis.set_ticklabels([])



    fig.savefig(fname)

def plot_spectrum(fits_name, fname):
    '''
    Sum up all of the flux in each image to make a spatially-integrated line spectrum.
    '''
    try:
        vs, data, header = readfn(fits_name)
    except FileNotFoundError as e:
        print("Cannot load {}, continuing.".format(fits_name))
        return

    # We always want to display the data from blueshifted to redshifted. This means we want to
    # find out if the velocities are decreasing, and if so, switch them to increasing

    if vs[-1] < vs[0]:
        print("Reversing velocity to be increasing.")
        vs = vs[::-1]
        data = data[::-1]

    nchan = data.shape[0]

    dict = common.get_coords(data, header, radius, mu_RA, mu_DEC)
    RA = 3600 * dict["RA"] # [arcsec]
    DEC = 3600 * dict["DEC"] # [arcsec]
    decl, decr = dict["DEC_slice"]
    ral, rar = dict["RA_slice"]
    data = dict["data"]

    # Need to divide by how many pixels are in a beam, so that we get Jy/pixel


    vsys = config["vsys"]
    v_cent = vs - vsys

    print(data.shape)

    flux = np.sum(data, axis=(1,2))

    fig, ax = plt.subplots(nrows=1)
    ax.plot(v_cent, flux, ls="steps-mid")
    ax.set_xlabel(r"$v$ [km/s]")
    ax.set_ylabel("Flux [Jy]")


    fig.savefig(fname)



# Go through and plot data, model, and residuals. If the file doesn't exist, the routine will skip.
plot_maps(config["data"], "data." + args.fmt)
plot_maps(config["model"], "model." + args.fmt)
plot_maps(config["resid"], "resid." + args.fmt)

# plot_spectrum(config["data"], "spec_data.pdf")
# plot_spectrum(config["model"], "spec_model.pdf")
# plot_spectrum(config["resid"], "spec_resid.pdf")
