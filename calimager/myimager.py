#!/usr/bin/env python

from matplotlib import pyplot as plt
import calimager.imager as img
import numpy as np
import struct
import sys, os
import ephem
import datetime
import argparse

IMAGE_RES = 1024

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

    L = np.linspace (-1, 1, IMAGE_RES);
    M = np.linspace (-1, 1, IMAGE_RES);
    mask = np.ones ( (IMAGE_RES, IMAGE_RES) );
    xv,yv = np.meshgrid (L,M);
    mask [np.sqrt(np.array(xv**2 + yv**2)) > 1] = np.NaN;
    freq_hz = np.array(subbands).mean()*(2e8/1024)
    imager = img.Imager('/usr/local/share/aartfaac/antennasets/lba_outer.dat', freq_hz, IMAGE_RES)
    for i in valid:
        imager.reset()
        time = datetime.datetime.utcfromtimestamp(metadata[i][0])
        for j in range(m):
            f = metadata[i+j][3]
            f.seek(metadata[i+j][4])
            acm.add_body(f.read(Acm.LEN_BDY))
            imager.addgrid(acm.data)
        img = imager.image()
        plt.clf()
        plt.imshow(img*mask, interpolation='bilinear', cmap=plt.get_cmap('jet'), extent=[L[0], L[-1], M[0], M[-1]])
        plt.title('Stokes I - %i - %s'%(int(freq_hz), time.strftime("%Y-%m-%d_%H:%M:%S")))
        plt.show()


"""
    with open(filename) as f:
        headers = []
        size = os.path.getsize(filename)
        N = size/(Acm.LEN_BDY+Acm.LEN_HDR)

        for i in range(N):
            chunk.add_header(f.read(Acm.LEN_HDR))
            headers.append((chunk.start_time, chunk.polarization, f.tell()))
            f.seek(f.tell()+Acm.LEN_BDY)

        headers.sort()
        print "Parsed {} headers, filesize {}".format(len(headers), size)

        xx = Acm()
        yy = Acm()
        assert(xx.subband == yy.subband)

        imager = img.Imager('/usr/local/share/aartfaac/antennasets/lba_outer.dat', chunk.subband*(2e8/1024), IMAGE_RES)

        l = np.linspace (-1, 1, IMAGE_RES);
        m = np.linspace (-1, 1, IMAGE_RES);
        mask = np.ones ( (IMAGE_RES, IMAGE_RES) );
        xv,yv = np.meshgrid (l,m);
        mask [np.sqrt(np.array(xv**2 + yv**2)) > 1] = np.NaN;

        
        for i in range(0, len(headers), 2):
            hdr = headers[i]
            f.seek(hdr[2])
            xx.add_body(f.read(Acm.LEN_BDY))
            hdr = headers[i+1]
            f.seek(hdr[2])
            yy.add_body(f.read(Acm.LEN_BDY))
            time = datetime.datetime.utcfromtimestamp(hdr[0])

            # annotations
            obs.date = datetime.datetime.utcfromtimestamp(hdr[0])

            # Compute local coordinates of all objects to be plotted.
            annotations = {}
            for k in obs_data.keys():
                obs_data[k].compute (obs);
                if obs_data[k].alt > 0:
                    u = -(np.cos(obs_data[k].alt) * np.sin(obs_data[k].az));
                    v =  (np.cos(obs_data[k].alt) * np.cos(obs_data[k].az)); 
                    annotations[k] = (u,v)

            imager.reset()
            imager.addgrid(xx.data)
            imager.addgrid(yy.data)
            img = imager.image()
            plt.clf()
            plt.imshow(img*mask, interpolation='bilinear', cmap=plt.get_cmap('jet'), extent=[l[0], l[-1], m[0], m[-1]])

            # add annotations
            for a in annotations.keys():
                plt.annotate(a, xy=annotations[a], xytext=(annotations[a][0]+0.1, annotations[a][1]+0.1), color='white', arrowprops=dict(facecolor='white', width=1, headwidth=4, shrink=0.15, edgecolor='white'),
                    horizontalalignment='left',
                    verticalalignment='bottom')
            plt.title('Stokes I - SB %d - %s'%(chunk.subband, time.strftime("%Y-%m-%d_%H:%M:%S")))
            plt.savefig('StokesI-SB%d-%s.png' % (chunk.subband, time.strftime("%Y-%m-%d_%H:%M:%S")))
            print "{}/{}".format(i,len(headers))

            plt.figure()
            plt.imshow(np.abs(imager.weights), interpolation='nearest', cmap=plt.get_cmap('jet'))
            plt.colorbar()
            plt.show()
"""



            


        
