import math
from astropy.io import fits
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap as LSC
from scipy.ndimage.interpolation import shift

c_kms = 2.99792458e5 # [km s^-1]

def read_SMA(fname):
    hdu_list = fits.open(fname)
    hdu = hdu_list[0]
    header = hdu.header

    vs = (np.arange(header["NAXIS3"])*header["CDELT3"] + header["CRVAL3"]) * 1e-3 # [km/s]

    data = hdu.data[0] # Get rid of the zombie first dimension.

    return (vs, data, header)

def read_ALMA(fname):
    hdu_list = fits.open(fname)
    hdu = hdu_list[0]
    header = hdu.header

    # Determine the frequencies.
    freq = header["CRVAL3"] + np.arange(header["NAXIS3"]) * header["CDELT3"]

    f0 = header["RESTFRQ"] # [Hz]

    # Convert all frequencies to velocity [km/s]
    vs = c_kms * (f0 - freq)/f0

    data = hdu.data[0] # Get rid of the zombie first dimension.

    return (vs, data, header)


def get_coords(data, header, radius, mu_RA=0.0, mu_DEC=0.0):
    '''
    Given a FITS header object, return the RA and DEC vectors.

    Alternatively, also specify a truncation radius and this routine will return the slice indices.

    If mu_RA, mu_DEC specified, shift the image by this amount.
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

    # Based upon the size of the image, convert mu_RA and mu_DEC into integer pixel shifts,
    # reporting the error in the centroiding (in pixels and arcsec).
    pix_x = mu_RA / 3600 / CDELT1
    pix_y = mu_DEC / 3600 / CDELT2
    print("Pixel shifts: RA: {:.2f}, DEC: {:.2f}".format(pix_x, pix_y))

    # Now, attempt the shifting using scipy
    data = shift(data, [0, -pix_y, -pix_x]) # 0 is don't shift in wavelength

    # # Truncate these
    # pix_x = int(pix_x)
    # pix_y = int(pix_y)
    #
    # # Then, add these pixel shifts to decl, decr, ral, and rar.
    # ral += pix_x
    # rar += pix_x
    # decl += pix_y
    # decr += pix_y
    #
    # # So now the dRAs and dDECs are centered around 0.0, but the ral,rar, etc used will
    # truncate the image such that it is shifted.

    data = data[:, decl:decr, ral:rar]

    # Lay down relative coordinates spanning the image
    # Note that RAs actually goes in decreasing order because of sky convention
    dRAs = (np.arange(nx) - nx/2)[decl:decr] * CDELT1
    dDECs = (np.arange(ny) - ny/2)[ral:rar] * CDELT2

    # Return the coordinates and the indices used to slice the dataset
    return {"RA":dRAs, "DEC":dDECs, "DEC_slice":(decl, decr), "RA_slice":(ral, rar), "data":data}


def plot_beam(ax, header, xy=(1,-1)):
    BMAJ = 3600. * header["BMAJ"] # [arcsec]
    BMIN = 3600. * header["BMIN"] # [arcsec]
    BPA =  header["BPA"] # degrees East of North
    print('BMAJ: {:.3f}", BMIN: {:.3f}", BPA: {:.2f} deg'.format(BMAJ, BMIN, BPA))
    # However, to plot it we need to negate the BPA since the rotation is the opposite direction
    # due to flipping RA.
    ax.add_artist(Ellipse(xy=xy, width=BMIN, height=BMAJ, angle=-BPA, facecolor="0.8", linewidth=0.2))

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


def get_geom_levels(rms, vmin, vmax, factor=np.sqrt(2)):
    levels = []
    val = -(3 * rms)
    while val > vmin:
        levels.append(val)

        val = val * factor

    # Reverse in place
    levels.reverse()
    val = 3 * rms
    while val < vmax:
        levels.append(val)
        val = val * factor

    return levels

# Make our custom intensity scale
dict_BuRd = {'red':   [(0.0,  0.0, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  1.0, 1.0)],

         'green': [(0.0,  0.0, 0.0),
                   (0.5, 1.0, 1.0),
                   (1.0,  0.0, 0.0)],

         'blue':  [(0.0,  1.0, 1.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.0, 0.0)]}

BuRd = LSC("BuRd", dict_BuRd)

dict_YlGr = {'red':   [(0.0,  1.0, 1.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.0, 0.0)],

         'green': [(0.0,  1.0, 1.0),
                   (0.5, 1.0, 1.0),
                   (1.0,  1.0, 1.0)],

         'blue':  [(0.0,  0.0, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.0, 0.0)]}

YlGr = LSC("YlGr", dict_YlGr)

def make_cmap(vel_frac):
    '''Given a symmetric velocity fraction between 0 (minimum) and 1.0 (maximum),
    with 0.5 being the middle, create a colormap to scale the emission in this channel.'''
    assert vel_frac >= 0.0 and vel_frac <= 1.0, "vel_frac must be in the range [0, 1]"

    # negative values
    r, g, b, a = YlGr(vel_frac)

    # positive values
    R, G, B, A = BuRd(vel_frac)

    # Take these rgba values and construct a new colorscheme
    cdict = {'red':   [(0.0,  r, r),
                   (0.5,  1.0, 1.0),
                   (1.0,  R, R)],

         'green': [(0.0,  g, g),
                   (0.5, 1.0, 1.0),
                   (1.0,  G, G)],

         'blue':  [(0.0,  b, b),
                   (0.5,  1.0, 1.0),
                   (1.0,  B, B)]}

    return  LSC("new", cdict)
