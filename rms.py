import numpy


def rms_invalid(rms, noise, low_bound=1, high_bound=50):
    """
    Is the RMS value of an image outside the plausible range?

    :param rms: RMS value of an image, can be computed with
                tkp.quality.statistics.rms
    :param noise: Theoretical noise level of instrument, can be calculated with
                  tkp.lofar.noise.noise_level
    :param low_bound: multiplied with noise to define lower threshold
    :param high_bound: multiplied with noise to define upper threshold
    :returns: True/False
    """
    if (rms < noise * low_bound) or (rms > noise * high_bound):
        ratio = rms / noise
        return "rms value (%s) is %s times theoretical noise (%s)" % \
                    (rms, ratio, noise)
    else:
        return False


def rms(data):
    """Returns the RMS of the data about the median.
    Args:
        data: a numpy array
    """
    data -= numpy.median(data)
    return numpy.sqrt(numpy.power(data, 2).sum()/len(data))


def clip(data, sigma=3):
    """Remove all values above a threshold from the array.
    Uses iterative clipping at sigma value until nothing more is getting clipped.
    Args:
        data: a numpy array
    """
    raveled = data.ravel()
    median = numpy.median(raveled)
    std = numpy.std(raveled)
    newdata = raveled[numpy.abs(raveled-median) <= sigma*std]
    if len(newdata) and len(newdata) != len(raveled):
        return clip(newdata, sigma)
    else:
        return newdata


def subregion(data, f=4):
    """Returns the inner region of a image, according to f.

    Resulting area is 4/(f*f) of the original.
    Args:
        data: a numpy array
    """
    x, y = data.shape
    return data[(x/2 - x/f):(x/2 + x/f), (y/2 - y/f):(y/2 + y/f)]


def rms_with_clipped_subregion(data, rms_est_sigma=3, rms_est_fraction=4):
    """
    RMS for quality-control.

    Root mean square value calculated from central region of an image.
    We sigma-clip the input-data in an attempt to exclude source-pixels
    and keep only background-pixels.

    Args:
        data: A numpy array
        rms_est_sigma: sigma value used for clipping
        rms_est_fraction: determines size of subsection, result will be
            1/fth of the image size where f=rms_est_fraction
    returns the rms value of a iterative sigma clipped subsection of an image
    """
    return rms(clip(subregion(data, rms_est_fraction), rms_est_sigma))


def sigmaclip(X, n, mask=None, iters=5):
    """Sigmaclip a numpy array.
    
    Keyword arguments:
    X -- the data
    n -- number of sigmas
    mask -- initial mask (default None)
    iters -- max number of iterations (default 5)
    """
    shape = X.shape
    if not mask:
        mask = np.zeros(shape, dtype=int)
        
    mx = ma.masked_array(X, mask=mask)
   
    for i in range(iters):
        centroid = np.median(mx.compressed())
        sigma = mx.std()
        masked_sum = np.sum(mask)
        mask |= (mx.data < centroid-n*sigma) | (mx.data > centroid+n*sigma)

        if masked_sum == np.sum(mask):
            break
            
        mx.mask = mask
    
    return np.sqrt(np.square(mx.compressed()).mean()), mx.min(), mx.max()
