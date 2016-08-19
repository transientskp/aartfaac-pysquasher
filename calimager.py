#!/usr/bin/env python

from matplotlib import pyplot as plt
import numpy as np
import multiprocessing
import struct
import sys, os
import datetime
import argparse
import gfft

def get_configuration():
    """Returns a populated configuration"""
    parser = argparse.ArgumentParser()
    parser.add_argument('files', metavar='FILE', type=argparse.FileType('r'), nargs='+',
            help="Files containing calibrated visibilities")
    parser.add_argument('--res', '-r', type=int, default=1024,
            help="Image output resolution (default: %(default)s)")

    return parser.parse_args()


class Acm:
    NUM_ANTS = 288
    LEN_HDR = 512
    LEN_BDY = NUM_ANTS * NUM_ANTS * 8
    HDR_MAGIC = 0x4141525446414143

    def __init__(self):
        self.start_time = 0.0
        self.end_time = 0.0
        self.subband = 0
        self.num_dipoles = 0
        self.polarization = 0
        self.num_channels = 0
        self.data = np.zeros((Acm.NUM_ANTS, Acm.NUM_ANTS), dtype=np.complex64)

    def add_header(self, hdr):
        """
        Add datablock header and check for correctness
        """
        if len(hdr) != Acm.LEN_HDR:
            sys.stderr.write("Invalid header size: expected %d, got %d.\n"%(Acm.LEN_HDR, len(hdr)))
            sys.exit(1)
        (magic, self.start_time, self.end_time, self.subband, self.num_dipoles, self.polarization, self.num_channels) = struct.unpack("<Qddiiii", hdr[0:40])
        if magic != Acm.HDR_MAGIC:
            sys.stderr.write("Invalid magic: expected %x, got %x.\n"% (Acm.HDR_MAGIC,magic))
            sys.exit(1)
        return True

    def add_body(self, body):
        """
        Add body data
        """
        self.data = np.fromstring(body, dtype=np.complex64)
        self.data = self.data.reshape(Acm.NUM_ANTS, Acm.NUM_ANTS)


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

    for a1 in range(0, Acm.NUM_ANTS):
        for a2 in range(0, Acm.NUM_ANTS):
            u.append(L[a1,0] - L[a2,0])
            v.append(L[a1,1] - L[a2,1])

    c = 299792458.0
    uv = [np.ravel([(np.array(u)/(c/(s*(2e8/1024))/2.0)) for s in subbands]), \
          np.ravel([(np.array(v)/(c/(s*(2e8/1024))/2.0)) for s in subbands])]

    return uv



if __name__ == "__main__":
    config = get_configuration()
    acm = Acm()
    metadata = []
    subbands = []

    for f in config.files:
        acm.add_header(f.read(Acm.LEN_HDR))
        subbands.append(acm.subband)
        f.seek(0)

    for f in config.files:
        size = os.path.getsize(f.name)
        N = size/(Acm.LEN_BDY+Acm.LEN_HDR)

        for i in range(N):
            acm.add_header(f.read(Acm.LEN_HDR))
            metadata.append((acm.start_time, acm.subband, acm.polarization, f, f.tell()))
            f.seek(f.tell()+Acm.LEN_BDY)

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
            valid.append(i)
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

    for i in valid:
        time = datetime.datetime.utcfromtimestamp(metadata[i][0])
        data = []
        for j in range(m):
            f = metadata[i+j][3]
            f.seek(metadata[i+j][4])
            acm.add_body(f.read(Acm.LEN_BDY))
            data.append(acm.data)

        img = np.fliplr(np.rot90(np.real(gfft.gfft(np.ravel(data), in_ax, out_ax))))

        plt.clf()
        plt.imshow(img*mask, interpolation='bilinear', cmap=plt.get_cmap('jet'), extent=[L[0], L[-1], M[0], M[-1]])
        plt.title('Stokes I - %0.2f - %s' % (freq_hz/1e6, time.strftime("%Y-%m-%d %H:%M:%S")))
        plt.savefig('StokesI-%i-%i-%s.png' % (np.array(subbands).mean(), len(subbands), time.strftime("%Y-%m-%d %H:%M:%S")))
        break
