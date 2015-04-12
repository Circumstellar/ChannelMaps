import math
from astropy.io import fits
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def get_coords(header, radius):
    '''
    Given a FITS header object, return the RA and DEC vectors.

    Alternatively, also specify a truncation radius and this routine will return the slice indices.
    '''
    nx = header["NAXIS1"]
    ny = header["NAXIS2"]
    nchan = header["NAXIS3"]

    assert nx % 2 == 0 and ny % 2 == 0 , "We don't have an even number of pixels, assumptions in the routine are violated."

    # Midpoints
    mx = int(nx/2)
    my = int(ny/2)

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

    # Alternatively
    slice_RA = math.floor(math.fabs(radius/CDELT1))
    slice_DEC = math.floor(math.fabs(radius/CDELT2))

    assert slice_RA < mx and slice_DEC < my, "Selected radius must be smaller than half size of image."

    decl = mx-slice_DEC
    decr = mx+slice_DEC

    ral = mx-slice_RA
    rar = mx+slice_RA

    # Lay down relative coordinates spanning the image
    # Note that RAs actually goes in decreasing order because of sky convention
    dRAs = (np.arange(nx) - nx/2)[decl:decr] * CDELT1
    dDECs = (np.arange(ny) - ny/2)[ral:rar] * CDELT2

    # Return the coordinates and the indices used to slice the dataset
    return {"RA":dRAs, "DEC":dDECs, "DEC_slice":(decl, decr), "RA_slice":(ral, rar)}


def plot_beam(ax, header, xy=(1,-1)):
    BMAJ = 3600. * header["BMAJ"] # [arcsec]
    BMIN = 3600. * header["BMIN"] # [arcsec]
    BPA =  header["BPA"] # degrees East of North
    ax.add_artist(Ellipse(xy=xy, width=BMIN, height=BMAJ, angle=BPA, facecolor="0.8", linewidth=0.2))

def get_levels(rms, vmin, vmax, spacing=3):
    '''

    First contour is at 3 sigma, and then contours go up (or down) in multiples of spacing
    '''

    levels = []
    # Add contours from rms to vmin, then reverse
    # We don't want a 0-level contour
    val = -(3 * rms)
    while val > vmin:
        levels.append(val)
        # After the first level, go down in increments of spacing

        val -= rms * spacing

    # Reverse in place
    levels.reverse()
    val = 3 * rms
    while val < vmax:
        levels.append(val)
        val += rms * spacing

    return levels
