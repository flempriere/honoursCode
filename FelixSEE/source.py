from .utilities import DataHandler, Utility
from .particle import Particle
import pickle
import os

class Source():
    """
    Stores source particle files and data meaning we don't need to access the file everytime we restart a simulation run
    unless we want to.
    """
    def __init__(self,label,sourcefile, filetype, pid, n_samples = 1, preselectioncut=None, nocuts=False, extend_to_low_pt_scale=False):
        self._label = label
        self._sourcefile = sourcefile
        self._filetype = filetype
        self._pid = pid
        self._n_samples = n_samples
        self._preselectioncut = preselectioncut
        self._nocuts=False
        self._extend_to_low_pt_scale=False
        self._source_particles = None
    
    def load_file(self):
        self._particles, self._weights = DataHandler.convert_list_to_momenta(self._sourcefile, Utility.get_mass(self._pid), self._filetype, self._n_samples,
                                                                  self._preselectioncut, self._nocuts, self._extend_to_low_pt_scale)
        self._source_particles = Particle(self._pid, self._particles, self._weights)
    
    def get_particles(self):
        return self._source_particles
    
    def display_spectrum(self, prange=[[-6,0, 100], [0,5, 80]]):
        plt,_,_,_ = DataHandler.convert_to_hist_list(self._particles, self._weights, do_plot=True, prange=prange)
        return plt
    
    def __str__(self):
        return self._label + "_" + str(self._n_samples)
    
    def save(self, path):
        pickle.dump(self, open(path, 'wr'))

    @classmethod
    def load(path):
        return pickle.load(open(path, 'rb'))
    
    def save_all(self, dirpath):
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        filename = self._label + "_" + self._n_samples + ".pkl"
        self.save(dirpath + filename)
        if self._source_particles is not None:
            particle_dir = dirpath + "/particles"
            self._source_particles.save_all(particle_dir)

    def get_label(self):
        return self._label
    
    def get_pid(self):
        return self._pid