#!/usr/bin/env python

import numpy as np
import multiprocessing
import struct
import os
import datetime
import pytz
import argparse
import logging
import gfft
import errno
import math
import rms
from astropy.io import fits
from astropy.time import Time, TimeDelta

# Python logging format in similar style to googles c++ glog format
LOG_FORMAT = "%(levelname)s %(asctime)s %(process)d %(filename)s:%(lineno)d] %(message)s"

VERSION = '1.1'
NUM_ANT = 288
LEN_HDR = 512
LEN_BDY = NUM_ANT**2 * 8
HDR_MAGIC = 0x4141525446414143
LOFAR_CS002_LONG = '6.869837540d'
LOFAR_CS002_LAT  = '52.915122495d'

def get_configuration():
    """
    Returns a populated configuration
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('files', metavar='FILE', type=argparse.FileType('r'), nargs='+',
            help="Files containing calibrated visibilities, supports glob patterns")
    parser.add_argument('--res', type=int, default=1024,
            help="Image output resolution (default: %(default)s)")
    parser.add_argument('--window', type=int, default=6,
            help="Kaiser window size (default: %(default)s)")
    parser.add_argument('--alpha', type=float, default=1.5,
            help="Kaiser window alpha param (default: %(default)s)")
    parser.add_argument('--inttime', type=int, default=1,
            help="Integration time (default: %(default)s)")
    parser.add_argument('--antpos', type=str, default='/usr/local/share/aartfaac/antennasets/lba_outer.dat',
            help="Location of the antenna positions file (default: %(default)s)")
    parser.add_argument('--nthreads', type=int, default=multiprocessing.cpu_count(),
            help="Number of threads to use for imaging (default: %(default)s)")
    parser.add_argument('--output', type=str, default=os.getcwd(),
            help="Output directory (default: %(default)s)")

    return parser.parse_args()


def parse_header(hdr):
    """
    Parse aartfaac header for calibrated data

    struct output_header_t
    {
      uint64_t magic;                   ///< magic to determine header                (  8 B)
      double start_time;                ///< start time (unix)                        (  8 B)
      double end_time;                  ///< end time (unix)                          (  8 B)
      int32_t subband;                  ///< lofar subband                            (  4 B)
      int32_t num_dipoles;              ///< number of dipoles (288 or 576)           (  4 B)
      int32_t polarization;             ///< XX=0, YY=1                               (  4 B)
      int32_t num_channels;             ///< number of channels (<= 64)               (  4 B)
      float ateam_flux[5];              ///< Ateam fluxes (CasA, CygA, Tau, Vir, Sun) ( 24 B)
      std::bitset<5> ateam;             ///< Ateam active                             (  8 B)
      std::bitset<64> flagged_channels; ///< bitset of flagged channels               (  8 B)
      std::bitset<576> flagged_dipoles; ///< bitset of flagged dipoles                ( 72 B)
      uint32_t weights[78];             ///< stationweights n*(n+1)/2, n in {6, 12}   (312 B)
      uint8_t pad[48];                  ///< 512 byte block                           ( 48 B)
    };
    """
    m, t0, t1, s, d, p, c = struct.unpack("<Qddiiii", hdr[0:40])
    f = np.frombuffer(hdr[80:152], dtype=np.uint64)
    assert(m == HDR_MAGIC)
    return (m, t0, t1, s, d, p, c, f)


def parse_data(data):
    """
    Parse aartfaac ACM
    """
    return np.fromstring(data, dtype=np.complex64).reshape(NUM_ANT, NUM_ANT)


def image_fits(metadata):
    """
    Create and write fits image
    """
    img = create_img(metadata)
    write_fits(img, metadata, fitshdu)


def write_fits(img, metadata, fitsobj):
    imgtime = Time(metadata[0][0] + config.inttime*0.5, scale='utc', format='unix', location=(LOFAR_CS002_LONG, LOFAR_CS002_LAT))

    imgtime.format='isot'
    imgtime.out_subfmt = 'date_hms'
    filename = '%s_S%0.1f_I%ix%i_W%i_A%0.1f.fits' % (imgtime.datetime.strftime("%Y%m%d%H%M%SUTC"), np.mean(subbands), len(subbands), config.inttime, config.window, config.alpha)
    filename = os.path.join(config.output, filename)

    if os.path.exists(filename):
        logger.info("'%s' exists - skipping", filename)
        return

    # CRVAL1 should hold RA in degrees. sidereal_time returns hour angle in
    # hours.
    fitsobj.header['CRVAL1'] = imgtime.sidereal_time(kind='apparent').value  *  15
    fitsobj.header['DATE-OBS'] = str(imgtime)
    imgtime_end = imgtime + TimeDelta(config.inttime, format='sec')
    fitsobj.header['END_UTC'] = str(imgtime_end)
    t = Time.now();
    t.format = 'isot'
    fitsobj.header['DATE'] = str(t)
    fitsobj.data[0, 0, :, :] = img
    data = img[np.logical_not(np.isnan(img))]
    quality = rms.rms(rms.clip(data))
    high = data.max()
    low = data.min()
    fitsobj.header['DATAMAX'] = high
    fitsobj.header['DATAMIN'] = low
    fitsobj.header['HISTORY'][0] = 'AARTFAAC 6 stations superterp'
    fitsobj.header['HISTORY'][1] = 'RMS {}'.format(quality)
    fitsobj.header['HISTORY'][2] = 'DYNAMIC RANGE {}:{}'.format(int(round(high)), int(round(quality)))
    fitsobj.writeto(filename)
    logger.info("%s %0.3f %i:%i", filename, quality, int(round(high)), int(round(quality)))


def create_empty_fits():
    """
    See http://heasarc.gsfc.nasa.gov/docs/fcg/standard_dict.html for details
    """
    hdu = fits.PrimaryHDU()
    hdu.header['AUTHOR'  ] = 'pysquasher.py - https://github.com/transientskp/aartfaac-pysquasher'
    hdu.header['REFERENC'] = 'http://aartfaac.org/'
    hdu.header['BSCALE'  ] =  1.
    hdu.header['BZERO'   ] =  0.
    hdu.header['BMAJ'    ] =  1.
    hdu.header['BMIN'    ] =  1.
    hdu.header['BPA'     ] =  0.
    hdu.header['BTYPE'   ] = 'Intensity'
    hdu.header['OBJECT'  ] = 'Aartfaac image'
    hdu.header['BUNIT'   ] = 'Jy/beam'
    hdu.header['EQUINOX' ] = 2000.
    hdu.header['RADESYS' ] = 'FK5'
    hdu.header['LONPOLE' ] = 180.
    hdu.header['LATPOLE' ] = float(LOFAR_CS002_LAT[0:-1]) # Latitude of LOFAR
    hdu.header['PC01_01' ] = 1.
    hdu.header['PC02_01' ] = 0.
    hdu.header['PC03_01' ] = 0.
    hdu.header['PC04_01' ] = 0.
    hdu.header['PC01_02' ] = 0.
    hdu.header['PC02_02' ] = 1.
    hdu.header['PC03_02' ] = 0.
    hdu.header['PC04_02' ] = 0.
    hdu.header['PC01_03' ] = 0.
    hdu.header['PC02_03' ] = 0.
    hdu.header['PC03_03' ] = 1.
    hdu.header['PC04_03' ] = 0.
    hdu.header['PC01_04' ] = 0.
    hdu.header['PC02_04' ] = 0.
    hdu.header['PC03_04' ] = 0.
    hdu.header['PC04_04' ] = 1.
    hdu.header['CTYPE1'  ] = 'RA---SIN'
    hdu.header['CRVAL1'  ] = 0. # Will be filled by imaging thread
    hdu.header['CDELT1'  ] = -math.asin(1./float(config.res/2)) * (180/math.pi)
    hdu.header['CRPIX1'  ] = config.res/2. + 1.
    hdu.header['CUNIT1'  ] = 'deg'
    hdu.header['CTYPE2'  ] = 'DEC--SIN'
    hdu.header['CRVAL2'  ] = float(LOFAR_CS002_LAT[0:-1])
    hdu.header['CDELT2'  ] = math.asin(1./float(config.res/2)) * (180/math.pi)
    hdu.header['CRPIX2'  ] = config.res/2. + 1.
    hdu.header['CUNIT2'  ] = 'deg'
    hdu.header['CTYPE3'  ] = 'FREQ'
    hdu.header['CRVAL3'  ] = freq_hz
    hdu.header['CDELT3'  ] = bw_hz
    hdu.header['CRPIX3'  ] = 1.
    hdu.header['CUNIT3'  ] = 'Hz'
    hdu.header['CTYPE4'  ] = 'STOKES'
    hdu.header['CRVAL4'  ] = 1.
    hdu.header['CDELT4'  ] = 1.
    hdu.header['CRPIX4'  ] = 1.
    hdu.header['CUNIT4'  ] = 'stokes-unit'
    hdu.header['PV2_1'   ] = 0.
    hdu.header['PV2_2'   ] = 0.
    hdu.header['RESTFRQ' ] = freq_hz
    hdu.header['RESTBW'  ] = bw_hz
    hdu.header['SPECSYS' ] = 'LSRK'
    hdu.header['ALTRVAL' ] = 0.
    hdu.header['ALTRPIX' ] = 1.
    hdu.header['VELREF'  ] = 257.
    hdu.header['TELESCOP'] = 'LOFAR'
    hdu.header['INSTRUME'] = 'AARTFAAC'
    hdu.header['OBSERVER'] = 'AARTFAAC Project'
    hdu.header['DATE-OBS'] = ''
    hdu.header['TIMESYS' ] = 'UTC'
    hdu.header['OBSRA'   ] = 0. # Will be filled by imaging thread
    hdu.header['OBSDEC'  ] = float(LOFAR_CS002_LAT[0:-1])
    hdu.header['OBSGEO-X'] = 3.8266e+06 # CS002 center ITRF location
    hdu.header['OBSGEO-Y'] = 4.6102e+05
    hdu.header['OBSGEO-Z'] = 5.0649e+06
    hdu.header['DATE'    ] = '' # Will be filled by imaging thread
    hdu.header['ORIGIN'  ] = 'Anton Pannekoek Institute'
    hdu.header['HISTORY' ] = '_'
    hdu.header['HISTORY' ] = '_'
    hdu.header['HISTORY' ] = '_'
    hdu.data = np.zeros( (1, 1, config.res, config.res) , dtype=np.float32)

    return hdu


# Convert visibilities to image
def create_img(metadata):
    """
    Constructs a single image
    """
    metadata.sort(key=lambda x: x[1])
    data = []
    for i in range(len(subbands)):
        sb = []
        for v in metadata[i*config.inttime*2:(i+1)*config.inttime*2]:
            f = open(v[3], 'r')
            f.seek(v[4])
            sb.append(parse_data(f.read(LEN_BDY)))
        data.append(np.array(sb).mean(axis=0))

    return np.rot90(np.real(gfft.gfft(np.ravel(data), in_ax, out_ax, verbose=False, W=config.window, alpha=config.alpha)), 3)*mask



def load(filename, subbands):
    """
    Load antenna positions and compute U,V coordinates
    """
    A = np.loadtxt(filename)
    R = np.array([[-0.1195950000, -0.7919540000, 0.5987530000],
                  [ 0.9928230000, -0.0954190000, 0.0720990000],
                  [ 0.0000330000,  0.6030780000, 0.7976820000]])
    L = A.dot(R)

    u, v = [], []

    for a1 in range(0, NUM_ANT):
        for a2 in range(0, NUM_ANT):
            u.append(L[a1, 0] - L[a2, 0])
            v.append(L[a1, 1] - L[a2, 1])

    c = 299792458.0
    return [np.ravel([(np.array(u)/(c/(s*(2e8/1024))/2.0)) for s in subbands]),
            np.ravel([(np.array(v)/(c/(s*(2e8/1024))/2.0)) for s in subbands])]


if __name__ == "__main__":
    formatter = logging.Formatter(LOG_FORMAT)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    config = get_configuration()
    try:
        os.makedirs(config.output)
        logger.info('Created directory \'%s\'', config.output)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(config.output):
            pass
        else:
            raise

    logger.info('pysquasher v%s (%i threads)', VERSION, config.nthreads)

    metadata = []
    subbands = []

    for f in config.files:
        _, t0, _, s, _, _, _, _ = parse_header(f.read(LEN_HDR))
        utc_first = datetime.datetime.utcfromtimestamp(t0).replace(tzinfo=pytz.utc)
        size = os.path.getsize(f.name)
        n = size/(LEN_BDY+LEN_HDR)
        f.seek((n-1)*(LEN_BDY+LEN_HDR))
        _, t0, _, s, _, _, _, _ = parse_header(f.read(LEN_HDR))
        utc_last = datetime.datetime.utcfromtimestamp(t0).replace(tzinfo=pytz.utc)
        logger.info('parsing \'%s\' (%i bytes) %s - %s',
                    os.path.basename(f.name), size,
                    datetime.datetime.strftime(utc_first, '%Y-%d-%m %H:%M:%S %Z'),
                    datetime.datetime.strftime(utc_last, '%Y-%d-%m %H:%M:%S %Z'))
        subbands.append(s)
        f.seek(0)

    subbands = list(set(subbands))
    subbands.sort()

    dx = 1.0 / config.res
    in_ax = load(config.antpos, subbands)
    out_ax = [(dx, config.res), (dx, config.res)]
    freq_hz = np.mean(subbands)*(2e8/1024)
    bw_hz = (np.max(subbands) - np.min(subbands) + 1)*(2e8/1024)

    logger.info('{} subbands: {}'.format(len(subbands), subbands))
    logger.info('%0.2f MHz central frequency', freq_hz*1e-6)
    logger.info('%0.2f MHz bandwidth', bw_hz*1e-6)
    logger.info('%i seconds integration time', config.inttime)
    logger.info('%ix%i pixel resolution', config.res, config.res)
    logger.info('%i Kaiser window size', config.window)
    logger.info('%0.1f Kaiser alpha parameter', config.alpha)

    for f in config.files:
        size = os.path.getsize(f.name)
        N = size/(LEN_BDY+LEN_HDR)

        for i in range(N):
            m, t0, t1, s, d, p, c, fl = parse_header(f.read(LEN_HDR))
            flagged = []
            for j,v in enumerate(fl):
                for k in range(64):
                    if np.bitwise_and(v, np.uint64(1<<k)):
                        flagged.append(j*64+k)

            metadata.append((t0, s, p, f.name, f.tell(), flagged))
            f.seek(f.tell()+LEN_BDY)

        f.close()

    metadata.sort()
    n = len(metadata)
    m = (len(subbands)*config.inttime)*2 # XX, YY pol
    skip = 0
    valid = []
    times = np.zeros((m, 1), dtype=np.uint32)
    for i in range(n-m):
        if i < skip:
            continue

        for j in range(m):
            times[j] = np.uint32(metadata[i+j][0])

        if (times[m-1] - times[0]) == (config.inttime - 1):
            valid.append(metadata[i:i+m])
            skip = i + m

    L = np.linspace(-1, 1, config.res)
    M = np.linspace(-1, 1, config.res)
    mask = np.ones((config.res, config.res))
    xv, yv = np.meshgrid(L, M)
    mask[np.sqrt(np.array(xv**2 + yv**2)) > 1] = np.NaN
    fitshdu = create_empty_fits()

    logging.info('Imaging %i images using %i threads', len(valid), config.nthreads)
    pool = multiprocessing.Pool(config.nthreads)
    pool.map(image_fits, valid)
