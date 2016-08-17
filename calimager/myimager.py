#!/usr/bin/env python

from matplotlib import pyplot as plt
import calimager.imager as img
import numpy as np
import struct
import sys, os
import ephem
import datetime

IMAGE_RES = 1024

class Chunk:
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
        self.data = np.zeros((Chunk.NUM_ANTS, Chunk.NUM_ANTS), dtype=np.complex64)

    def add_header(self, hdr):
        """
        Add datablock header and check for correctness
        """
        if len(hdr) != Chunk.LEN_HDR:
            sys.stderr.write("Invalid header size: expected %d, got %d.\n"%(Chunk.LEN_HDR, len(hdr)))
            sys.exit(1)
        (magic, self.start_time, self.end_time, self.subband, self.num_dipoles, self.polarization, self.num_channels) = struct.unpack("<Qddiiii", hdr[0:40])
        if magic != Chunk.HDR_MAGIC:
            sys.stderr.write("Invalid magic: expected %x, got %x.\n"% (Chunk.HDR_MAGIC,magic))
            sys.exit(1)
        return True

    def add_body(self, body):
        """
        Add body data
        """
        self.data = np.fromstring(body, dtype=np.complex64)
        self.data = self.data.reshape(Chunk.NUM_ANTS, Chunk.NUM_ANTS)

if __name__ == "__main__":
    filename = '/media/fhuizing/data1/294-20160813135549.cal'
    filename = '/media/fhuizing/data1/300-20160813135547.cal'
    filename = '/media/fhuizing/data1/S294_C63_M9_T20131120-133700-CAL'

    obs = ephem.Observer();
    obs.pressure = 0; # To prevent refraction corrections.
    obs.lon, obs.lat = '6.869837540','52.915122495'; # CS002 on LOFAR
    obs_data = {};
    obs_data['Moon'] = ephem.Moon();
    obs_data['Jupiter'] = ephem.Jupiter();
    obs_data['Sun'] = ephem.Sun();
    obs_data['Cas.A'] = ephem.readdb('Cas-A,f|J, 23:23:26.0, 58:48:00,99.00,2000');
    obs_data['Cyg.A'] = ephem.readdb('Cyg-A,f|J, 19:59:28.35, 40:44:02,99.00,2000');
    obs_data['Tau.A'] = ephem.readdb('Tau-A,f|J, 05:34:31.94, 22:00:52.24,99.00,2000');
    obs_data['Vir.A'] = ephem.readdb('Vir-A,f|J, 12:30:49.42338, 12:23:28.0439,99.00,2000');
    obs_data['NCP'] = ephem.readdb('NCP,f|J, 0, 90:00:00,99.00,2000');
    obs_data['Gal. Center'] = ephem.readdb('Galactic Center,f|J, 17:45:40.0, -29:00:28.1,99.00, 2000'); 
    
    chunk = Chunk()
    with open(filename) as f:
        headers = []
        size = os.path.getsize(filename)
        N = size/(Chunk.LEN_BDY+Chunk.LEN_HDR)

        for i in range(N):
            chunk.add_header(f.read(Chunk.LEN_HDR))
            headers.append((chunk.start_time, chunk.polarization, f.tell()))
            f.seek(f.tell()+Chunk.LEN_BDY)

        headers.sort()
        print "Parsed {} headers, filesize {}".format(len(headers), size)

        xx = Chunk()
        yy = Chunk()
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
            xx.add_body(f.read(Chunk.LEN_BDY))
            hdr = headers[i+1]
            f.seek(hdr[2])
            yy.add_body(f.read(Chunk.LEN_BDY))
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



            


        
