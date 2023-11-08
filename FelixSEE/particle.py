from .utilities import Utility
import os
import pickle
import numpy as np

class Particle():
    """A class that implements a specific instance of a particle in the detector.
    It contains the particle ID and parameters (mass, charge, lifetime)
    and the associated momentum and weight
    """
    def __init__(self, pid, momentum, weight=1):
        """
        Particle constructor, constructs a particle instance from a given ID and
        momentum.
        """
        self._pid = pid
        self._load_particle(pid)
        self._momentum = momentum
        self._weight = weight
        self._n_particles = len(momentum)

    def __str__(self):
        mass, charge, ctau = self.get_particle_properties()
        return f"Particle instance: {self._pid} (mass: {mass}, charge: {charge}, ctau: {ctau})\n 4-momenta: {self._momentum}, \n weight: {self._weight}"

    
    def _load_particle(self, pid):
        self._mass, self._charge, self._ctau = Utility.get_particle_params(pid)
    
    def get_particle_properties(self):
        return self._mass, self._charge, self._ctau
    
    def get_statistical_properties(self):
        return self._momentum, self._weight
    
    def set_weight(self, weight):
        self._weight = weight

    def get_weight(self):
        return self._weight
    
    def get_momentum(self):
        return self._momentum
    
    def get_ID(self):
        return self._pid
    
    def get_n_particles(self):
        return self._n_particles
    
    def save(self, filename):
        pickle.dump(self, open(filename, 'wb'))

    def save_all(self, dirpath):
        if not os.path.exists():
            os.makedirs(dirpath)
        filename = "/" + str(self._pid) + "_" + str(self._n_particles)
        self.save(dirpath + filename + ".pkl")
        self._momentum.save_all(dirpath)
        outputArray = np.array([self._momentum.fourVector, self._weight], dtype='object')
        np.save(dirpath + filename + ".npy",outputArray)