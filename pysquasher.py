#!/usr/bin/env python

from matplotlib import pyplot as plt
import numpy as np
import multiprocessing
import struct
import sys
import os
import datetime
import argparse
import gfft
import logging, logging.handlers

LOG_FORMAT = "%(levelname)s %(asctime)s %(process)d %(filename)s:%(lineno)d] %(message)s"

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
    parser.add_argument('--nthreads', type=int, default=multiprocessing.cpu_count(),
            help="Number of threads to use for imaging (default: %(default)s)")

    return parser.parse_args()


def parse_header(hdr):
    """
    Parse aartfaac header for calibrated data
    """
    m, t0, t1, s, d, p, c = struct.unpack("<Qddiiii", hdr[0:40])
    assert(m == HDR_MAGIC)
    return (m, t0, t1, s, d, p, c)


def parse_data(data):
    """
    Parse aartfaac ACM
    """
    return np.fromstring(data, dtype=np.complex64).reshape(NUM_ANT, NUM_ANT)


def process(sbs):
    """
    Constructs a single image
    """
    time = datetime.datetime.utcfromtimestamp(sbs[0][0])
    data = []
    for sb in sbs:
        f = open(sb[3], 'r')
        f.seek(sb[4])
        data.append(parse_data(f.read(LEN_BDY)))

    img = np.fliplr(np.rot90(np.real(gfft.gfft(np.ravel(data), in_ax, out_ax, verbose=False))))

    filename = '%s-S%i-I%i.png' % (time.strftime("%Y%m%d%H%M%S"), np.mean(subbands), len(subbands))
    plt.clf()
    plt.imshow(img*mask, interpolation='bilinear', cmap=plt.get_cmap('jet'), extent=[L[0], L[-1], M[0], M[-1]])
    plt.title('Stokes I - %0.2f - %s' % (freq_hz/1e6, time.strftime("%Y-%m-%d %H:%M:%S")))
    plt.savefig(filename)
    logger.info(filename)


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
    S = []
    # add subband twice for each pol
    for s in subbands:
        S.append(s)
        S.append(s)

    return [np.ravel([(np.array(u)/(c/(s*(2e8/1024))/2.0)) for s in S]), \
           np.ravel([(np.array(v)/(c/(s*(2e8/1024))/2.0)) for s in S])]


if __name__ == "__main__":
    formatter = logging.Formatter(LOG_FORMAT)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    config = get_configuration()
    metadata = []
    subbands = []

    for f in config.files:
        _, _, _, s, _, _, _ = parse_header(f.read(LEN_HDR))
        subbands.append(s)
        f.seek(0)

    logger.info('Imaging %i subbands', len(subbands))
    logger.info('  %0.2f MHz central frequency', np.array(subbands).mean()*2e2/1024)
    logger.info('  %0.2f MHz bandwidth', len(subbands)*(2e2/1024))
    logger.info('  %ix%i pixel resolution', config.res, config.res)

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
    m = len(subbands)*2
    valid = []
    times = np.zeros((m, 1), dtype=np.uint32)
    for i in range(n):
        if i+m >= n:
            break
        for j in range(m):
            times[j] = np.uint32(metadata[i+j][0])
        if (times[0] == times).all():
            valid.append(metadata[i:i+m])
            i += m

    L = np.linspace (-1, 1, config.res);
    M = np.linspace (-1, 1, config.res);
    mask = np.ones ( (config.res, config.res) );
    xv,yv = np.meshgrid (L,M);
    mask [np.sqrt(np.array(xv**2 + yv**2)) > 1] = np.NaN;
    freq_hz = np.array(subbands).mean()*(2e8/1024)
    dx = 1.0 / config.res

    in_ax = load('/usr/local/share/aartfaac/antennasets/lba_outer.dat', subbands)
    out_ax = [(dx, config.res), (dx, config.res)]

    subbands = np.array(subbands)
    pool = multiprocessing.Pool(config.nthreads)
    pool.map(process, valid)
