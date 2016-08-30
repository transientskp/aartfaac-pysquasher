#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import multiprocessing
import struct
import sys
import os
import datetime
import pytz
import argparse
import logging
import gfft
import errno
from astropy.io import fits

LOG_FORMAT = "%(levelname)s %(asctime)s %(process)d %(filename)s:%(lineno)d] %(message)s"

VERSION = '1.0'
NUM_ANT = 288
LEN_HDR = 512
LEN_BDY = NUM_ANT**2 * 8
HDR_MAGIC = 0x4141525446414143

def get_configuration():
    """
    Returns a populated configuration
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('files', metavar='FILE', type=argparse.FileType('r'), nargs='+',
            help="Files containing calibrated visibilities")
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
      uint64_t magic;                   ///< magic to determine header
      double start_time;                ///< start time (unix)
      double end_time;                  ///< end time (unix)
      int32_t subband;                  ///< lofar subband
      int32_t num_dipoles;              ///< number of dipoles (288 or 576)
      int32_t polarization;             ///< XX=0, YY=1
      int32_t num_channels;             ///< number of channels (<= 64)
      float ateam_flux[5];              ///< Ateam fluxes (CasA, CygA, Tau, Vir, Sun)
      std::bitset<5> ateam;             ///< Ateam active
      std::bitset<64> flagged_channels; ///< bitset of flagged channels (8 byte)
      std::bitset<576> flagged_dipoles; ///< bitset of flagged dipoles (72 byte)
      uint32_t weights[78];             ///< stationweights n*(n+1)/2, n in {6, 12}
      uint8_t pad[48];                  ///< 512 byte block
    };
    """
    m, t0, t1, s, d, p, c = struct.unpack("<Qddiiii", hdr[0:40])
    assert(m == HDR_MAGIC)
    return (m, t0, t1, s, d, p, c)


def parse_data(data):
    """
    Parse aartfaac ACM
    """
    return np.fromstring(data, dtype=np.complex64).reshape(NUM_ANT, NUM_ANT)

# Create a png image using 'img' with metadata from 'metadata'
def write_png (img, metadata):

    # Create filename
    filename = '%s_S%0.1f_I%ix%i_W%i_A%0.1f.png' % (time.strftime("%Y%m%d%H%M%S%Z"), np.mean(subbands), len(subbands), config.inttime, config.window, config.alpha)
    plt.clf()
    plt.imshow(img*mask, interpolation='bilinear', cmap=plt.get_cmap('jet'), extent=[L[0], L[-1], M[0], M[-1]])
    plt.title('F %0.2fMHz - BW %0.2fMHz - T %is - %s' % (freq_hz/1e6, bw_hz/1e6, config.inttime, time.strftime("%Y-%m-%d %H:%M:%S %Z")), fontsize=9)
    plt.colorbar()
    plt.savefig(os.path.join(config.output, filename))
    logger.info(filename)

# Create a FITS image using 'img' with metadata from 'metadata'
def write_fits (img, metadata):

    # Create filename
    filename = '%s_S%0.1f_I%ix%i_W%i_A%0.1f.fits' % (time.strftime("%Y%m%d%H%M%S%Z"), np.mean(subbands), len(subbands), config.inttime, config.window, config.alpha)
    hdu = fits.PrimaryHDU()
    hdulist = fits.HDUList ([hdu])


def create_img (metadata):
    """
    Constructs a single image
    """
    time = datetime.datetime.utcfromtimestamp(metadata[0][0]+config.inttime*0.5).replace(tzinfo=pytz.utc)
    metadata.sort(key=lambda x: x[1])
    data = []
    for i in range(len(subbands)):
        sb = []
        for v in metadata[i*config.inttime*2:(i+1)*config.inttime*2]:
            f = open(v[3], 'r')
            f.seek(v[4])
            sb.append(parse_data(f.read(LEN_BDY)))
        data.append(np.array(sb).mean(axis=0))
              
    return np.fliplr(np.rot90(np.real(gfft.gfft(np.ravel(data), in_ax, out_ax, verbose=False, W=config.window, alpha=config.alpha))))



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
            u.append(L[a1,0] - L[a2,0])
            v.append(L[a1,1] - L[a2,1])

    c = 299792458.0
    return [np.ravel([(np.array(u)/(c/(s*(2e8/1024))/2.0)) for s in subbands]), \
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
        _, t0, _, s, _, _, _ = parse_header(f.read(LEN_HDR))
        utc_first = datetime.datetime.utcfromtimestamp(t0).replace(tzinfo=pytz.utc)
        size = os.path.getsize(f.name)
        n = size/(LEN_BDY+LEN_HDR)
        f.seek((n-1)*(LEN_BDY+LEN_HDR))
        _, t0, _, s, _, _, _ = parse_header(f.read(LEN_HDR))
        utc_last = datetime.datetime.utcfromtimestamp(t0).replace(tzinfo=pytz.utc)
        logger.info('parsing \'%s\' (%i bytes)', os.path.basename(f.name), size)
        logger.info('  %s start time', datetime.datetime.strftime(utc_first, '%Y-%d-%m %H:%M:%S %Z'))
        logger.info('  %s end time', datetime.datetime.strftime(utc_last, '%Y-%d-%m %H:%M:%S %Z'))
        subbands.append(s)
        f.seek(0)

    dx = 1.0 / config.res
    in_ax = load(config.antpos, subbands)
    out_ax = [(dx, config.res), (dx, config.res)]
    freq_hz = np.mean(subbands)*(2e8/1024)
    bw_hz = (np.max(subbands) - np.min(subbands) + 1)*(2e8/1024)

    logger.info('%i subbands', len(subbands))
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
            m, t0, t1, s, d, p, c = parse_header(f.read(LEN_HDR))
            metadata.append((t0, s, p, f.name, f.tell()))
            f.seek(f.tell()+LEN_BDY)

        f.close()

    metadata.sort()
    n = len(metadata)
    m = (len(subbands)*config.inttime)*2
    skip = 0
    valid = []
    times = np.zeros((m, 1), dtype=np.uint32)
    for i in range(n-m):
        if i < skip:
            continue

        for j in range(m):
            times[j] = np.uint32(metadata[i+j][0])

        if (times[0] <= times+config.inttime).all():
            valid.append(metadata[i:i+m])
            skip = i + m

    L = np.linspace(-1, 1, config.res);
    M = np.linspace(-1, 1, config.res);
    mask = np.ones((config.res, config.res));
    xv,yv = np.meshgrid(L,M);
    mask[np.sqrt(np.array(xv**2 + yv**2)) > 1] = np.NaN;

    logging.info('Imaging %i images using %i threads', len(valid), config.nthreads)
    pool = multiprocessing.Pool(config.nthreads)
    pool.map(image_png, valid)
