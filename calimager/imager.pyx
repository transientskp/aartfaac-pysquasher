from capture import *
import cython
import numpy as np
cimport numpy as np

# Include complex arithmetic
cdef extern from "complex.h":
    pass

class Imager:
    C_MS = 299792458.0

    def __init__(self, antposfile, f_hz, size):
        """
        Initialize the imager vars and array structures
        """
        self.mDuv = Imager.C_MS / f_hz / 2.0
        self.size = size
        self.gridvis = np.zeros((size, size), np.complex64)
        self.img = np.zeros((size, size), np.float32)
        self.load(antposfile)
        self.dl = (Imager.C_MS/(f_hz * size * self.mDuv)); # dimensionless, in dir. cos. units
        self.lmax = self.dl * size/ 2;
        # self.l = np.linspace (-self.lmax, self.lmax, size);
        # self.m = np.linspace (-self.lmax, self.lmax, size);
        # self.mask = np.ones (size, size);
        # self.mask [np.sqrt(np.array(np.meshgrid (self.l))**2 + np.array(np.meshgrid(self.l)).conj().transpose()**2) > 1] = np.NaN;

    def reset(self):
        self.gridvis.fill(0)


    def addgrid(self, cm):
        self._grid(self.U, self.V, self.gridvis, cm)


    def image(self):
        """
        Create an image from the correlation matrix
        """
        self.gridvis = np.fft.fftshift(self.gridvis)
        self.gridvis = np.flipud(np.fliplr(self.gridvis))
        self.gridvis = np.conjugate(self.gridvis)
        self.img = np.real(np.fft.fftshift(np.fft.fft2(self.gridvis)))
        self.img = self.img[::-1, ::-1]
        return self.img


    def load(self, filename):
        """
        Load antenna positions and compute U,V coordinates
        """
        A = np.loadtxt(filename)
        R = np.array([[-0.1195950000, -0.7919540000, 0.5987530000],
                      [ 0.9928230000, -0.0954190000, 0.0720990000],
                      [ 0.0000330000,  0.6030780000, 0.7976820000]])
        L = A.dot(R)

        N = Chunk.NUM_ANTS
        self.U = np.zeros((N,N), dtype=np.float64)
        self.V = np.zeros((N,N), dtype=np.float64)

        for a1 in range(0, N):
            for a2 in range(0, N):
                self.U[a1, a2] = L[a1,0] - L[a2,0]
                self.V[a1, a2] = L[a1,1] - L[a2,1]


    def _grid(self, 
            np.ndarray[np.float64_t, ndim=2] U, 
            np.ndarray[np.float64_t, ndim=2] V, 
            np.ndarray[np.complex64_t, ndim=2] G, 
            np.ndarray[np.complex64_t, ndim=2] C):
        """
        Cythonified gridding function
        """
        cdef int w, e, s, n
        cdef float p, u, v
        cdef float west_power
        cdef float east_power
        cdef float south_power
        cdef float north_power
        cdef float south_west_power
        cdef float north_west_power
        cdef float south_east_power
        cdef float north_east_power
        cdef float duv = self.mDuv
        cdef float size = self.size
        cdef int N = Chunk.NUM_ANTS

        for a1 in range(N):
            for a2 in range(N):
                p = 1.0
                if a1 == a2:
                    p = 0.5

                u = U[a1, a2] / duv + size / 2 - 1
                v = V[a1, a2] / duv + size / 2 - 1

                w = int(np.floor(u))
                e = int(np.ceil(u))
                s = int(np.floor(v))
                n = int(np.ceil(v))

                west_power  = p - (u - w)
                east_power  = p - (e - u)
                south_power = p - (v - s)
                north_power = p - (n - v)

                south_west_power = south_power * west_power
                north_west_power = north_power * west_power
                south_east_power = south_power * east_power
                north_east_power = north_power * east_power

                G[s, w] += south_west_power * C[a1, a2]
                G[n, w] += north_west_power * C[a1, a2]
                G[s, e] += south_east_power * C[a1, a2]
                G[n, e] += north_east_power * C[a1, a2]
