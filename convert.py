#!/usr/bin/env python

import os
import argparse
import logging
import time
import sys
import numpy as np

from pysquasher import align_to_magic, parse_header
from pysquasher import HDR_MAGIC, LOG_FORMAT, LEN_HDR, NUM_ANT
LEN_BDY = NUM_ANT * NUM_ANT * 8
LEN_TOT = LEN_HDR + LEN_BDY


def get_configuration():
    """
    Returns a populated configuration
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('files', metavar='FILE', type=argparse.FileType('r'), nargs='+',
            help="Files containing calibrated visibilities, supports glob patterns")
    parser.add_argument('--output', type=str, default=os.getcwd(),
            help="Output directory (default: %(default)s)")

    return parser.parse_args()


if __name__ == "__main__":
    formatter = logging.Formatter(LOG_FORMAT)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    cfg = get_configuration()
    i, j = np.tril_indices(NUM_ANT)
    indices = j*NUM_ANT + i

    if not os.path.exists(cfg.output):
        os.makedirs(cfg.output)
        logging.info('Created dir {}'.format(cfg.output))

    for f in cfg.files:
        start = align_to_magic(f)
        f.seek(LEN_HDR)
        
        if (align_to_magic(f) - start) != LEN_TOT:
            logging.error('Invalid format for {}, ignoring'.format(f.name))
            continue

        filename = os.path.join(cfg.output, os.path.basename(f.name))
        logging.info('{} -> {}'.format(os.path.basename(f.name), filename))
        f.seek(start)
        size = os.path.getsize(f.name)
        n = size/(LEN_TOT)

        with open(filename, 'wb') as nf:
            for i in range(n):
                data = f.read(LEN_TOT)
                hdr = parse_header(data[:LEN_HDR])
                assert(hdr[0] == HDR_MAGIC)
                acm = np.fromstring(data[LEN_HDR:], dtype=np.complex64).reshape(NUM_ANT, NUM_ANT)
                upp = np.take(acm, indices)
                nf.write(data[:LEN_HDR])
                nf.write(upp.tostring())
