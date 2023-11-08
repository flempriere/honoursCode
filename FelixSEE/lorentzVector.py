import os
import pickle
import numpy as np


class LorentzVector():
    def __init__(self, vectorArray):
        n_rows, n_cols = vectorArray.shape
        if not (n_cols == 4):
            raise ValueError(
                "Input vector must be of shape (n_vectors, 4), given shape was ({0}, {1})".format(n_rows, n_cols)
            )
        self.n_vectors = n_rows
        self.vectorArray = vectorArray

    @classmethod
    def vector_only(cls, vectorArray):
        n_rows, n_cols = vectorArray.shape
        if not (n_cols == 3):
            raise ValueError(
                "Input vector must be of shape (n_vectors, 3), given shape was {0}, {1})".format(n_rows, n_cols)
            )
        t = np.zeros(n_rows)
        vectorArray = np.array(np.vstack[vectorArray.T, t].T)
        return cls(vectorArray)
    
    @property
    def fourVector(self):
        return self.vectorArray
    
    @fourVector.setter
    def fourVector(self, vectorArray):
        n_rows, n_cols = vectorArray.shape
        if not (n_cols == 4):
            raise ValueError(
                "Input vector must be of shape (n_vectors, 4), given shape was ({0}, {1})".format(n_rows, n_cols)
            )
        self.n_vectors = n_rows
        self.vectorArray = vectorArray

    @property
    def x(self):
        return self.vectorArray[:, 0]
    
    @x.setter
    def x(self, value):
        self.vectorArray[:, 0] = value

    @property
    def y(self):
        return self.vectorArray[:, 1]
    
    @y.setter
    def y(self, value):
        self.vectorArray[:, 1] = value

    @property
    def z(self):
        return self.vectorArray[:, 2]
    
    @z.setter
    def z(self, value):
        self.vectorArray[:, 2] = value

    @property
    def t(self):
        return self.vectorArray[:, 3]
    
    @t.setter
    def t(self, value):
        self.vectorArray[:, 3] = value

    def __len__(self):
        return self.n_vectors
    
    @property
    def mag(self):
        return np.linalg.norm(self.vectorArray[:,:-1], axis=1)

    @property
    def mag2(self):
        return np.sum(np.power(self.vectorArray[:,:-1], 2), axis=1)
    
    @property
    def p(self):
        return self.mag
    
    @property
    def vector(self):
        return self.vectorArray[:,:-1]
    
    @property
    def pt(self):
        return np.linalg.norm(self.vectorArray[:,:-2], axis=1)
    
    @property
    def perp2(self):
        return np.sum(np.power(self.vectorArray[:, :-2], 2), axis=1)

    def set(self, vectorArray):
        n_rows, n_cols = vectorArray.shape
        if not (n_cols == 4):
            raise ValueError(
                "Input vector must be of shape (n_vectors, 4), given shape was ({0}, {1})".format(n_rows, n_cols)
            )
        self.n_vectors = n_rows
        self.vectorArray = vectorArray

    @property
    def boostvector(self):
        return (self.vector.T/self.t).T

    def boost(self, boostvector):
        b2 = np.sum(np.power(boostvector,2), axis=1) #[n_particles]
        gamma = 1./np.sqrt(1.0 - b2) #[n_particles]
        bp = np.sum(boostvector * self.vector, axis=1) #[n_particles]
        b2[b2 < 0] = 0
        gamma2 = np.divide(gamma - 1., b2, out=np.zeros(gamma.shape), where=b2 != 0)

        net_p = self.vector + (gamma2*bp)[:, np.newaxis]*boostvector - gamma[:, np.newaxis]*boostvector*self.t[:, np.newaxis]
        net_t = gamma*(self.t - bp)

        return LorentzVector(np.vstack([net_p.T, net_t]).T)
    
    def angle(self, vector):
        return np.arccos(np.dot(vector, self.vector.T/self.mag))
    
    def copy(self):
        return LorentzVector(self.vectorArray.copy())
    
    def __mul__(self, other):
        v = self.copy()
        v *= other
        return  v
    
    def __imul__(self, other):
        self.vectorArray *= other
        return self
    
    def __rmul__(self, other):
        v = self.copy()
        v.vectorArray = other * v.vectorArray
        return v

#    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
#        print("how did I get here?")
#        lhs, rhs = inputs
#        return lhs * rhs

    def __add__(self, other):
        v = self.copy()
        v += other
        return v

    def __iadd__(self, other):
        self.vectorArray += other
        return self

    def __sub__(self, other):
        v = self.copy()
        v -= other
        return v
    
    def __isub__(self, other):
        self.vectorArray -= other
        return self
    
    def __truediv__(self, other):
        v = self.copy()
        v /= other
        return v
    
    def __itruediv__(self, other):
        self.vectorArray /= other
        return self

    def dot(self, v):
        v = v.reshape((-1, 4))
        n_rows, n_cols = v.shape
        if not (n_rows == 1) or (n_rows == self.n_vectors):
            raise ValueError(
                "Input error, v must be broadcastable to (1, 4) or ({0}, 4) but shape was ({1}, {2})".format(self.n_vectors, n_rows, n_cols)
            )
        v[:,:-1] = -1*v[:,:-1]
        return np.dot(self.vectorArray, v)
    
    def combine(self, others):
        other_vectors = [oth.fourVector for oth in others]
        total_vector = np.vstack([self.fourVector] + other_vectors)
        return LorentzVector(total_vector)
    
    def save(self, path):
        pickle.dump(self, open(path, 'wr'))
    
    def save_all(self, dirpath):
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        filename = "/LorentzVector"
        nparraydir = dirpath + '/raw_arrays'
        if not os.path.exist(dirpath):
            os.makedirs(nparraydir)
        np.save(nparraydir + filename + "_array.npy", self.vectorArray)
        pickle.save(self, dirpath + filename)