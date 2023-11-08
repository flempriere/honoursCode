import pickle
from abc import ABC, abstractmethod
from .particle import Particle
from .lorentzVector import LorentzVector
import numpy as np
from .utilities import Utility
import os
import pickle
import scipy as sp

class Decay(ABC):
    """Parent class for a particle Decay, generally should be """
    def __init__(self, label, parent_ID,  product_IDs, branching_function, branching_kwargs=None, visible=True):
        self._label = label
        self._parent_ID = parent_ID
        self._product_IDs = product_IDs
        self._branching_function = branching_function
        self._branching_kwargs = branching_kwargs
        self._visible= visible
    
    @abstractmethod
    def decay_particle(self, particle):
        pass
        
    def set_branching_kwargs(self, branching_kwargs):
        self._branching_kwargs = branching_kwargs

    def update_branching_kwargs(self, updated_kwargs):
        self._branching_kwargs.update(updated_kwargs)
    
    def get_label(self):
        return self.__label
    
    def _make_particle(self, pid, momenta, weight):
        return Particle(pid, momenta, weight)
    
    def __str__(self):
        return f"decay {self.__label}: {self.__parent_ID} -> {self.__product_IDs}"

    def is_visible(self):
        return self._visible

    def save(self, filename):
        pickle.dump(self, open(filename, 'wr'))

class Two_Body_Decay(Decay):

    def __init__(self, label, parent_ID, product_IDs, branching_function, branching_kwargs=None, visible=True):
        super().__init__(label, parent_ID, product_IDs, branching_function, branching_kwargs, visible)

    def _get_rotation(self, momentum):
        #momentum is [n_particles][4] LorentzVector object

        zaxis = np.array([0, 0, 1])
        cross_product = np.cross(zaxis, momentum.vector)
        rotaxis = (cross_product.T / np.linalg.norm(cross_product, axis=1)).T #np array of dimension [n_particles]
        rotangle = momentum.angle(zaxis)
        #rotangle = np.arccos(np.dot(zaxis, momentum.vector.T/momentum.mag)) #np array of dimension [n_particles]


        #rotaxis = [zaxis.cross(momenta.vector).unit() for momenta in momentum]
        #rotangle = [zaxis.angle(momenta.vector) for momenta in momentum]

        rotation_vec = (rotangle[:,np.newaxis]*rotaxis)
        Rotations = sp.spatial.transform.Rotation.from_rotvec(rotation_vec)
        return Rotations
    
    def _determine_energy_momenta(self, parent_mass, product_masses, n_particles):

        mass_squared = np.power(product_masses, 2)
        energy1   = (parent_mass**2 + mass_squared[0] - mass_squared[1])/(2.*parent_mass)
        energy2   = (parent_mass - energy1)
        energies = np.array([energy1, energy2])
        momenta = np.sqrt(np.maximum(np.power(energies,  2) - mass_squared, 0))
        
        fourvectors = self._assign_direction(energies, momenta, n_particles)
        return fourvectors
    
    def _assign_direction(self, energies, momenta, n_particles):

        #momenta is [2]
        #angles is [n_particles]
        #outer product dimension is [2][n_particles]
        
        phi = Utility.get_uniform_random_numbers(np.pi, -np.pi, n_particles)
        costheta = Utility.get_uniform_random_numbers(1., -1, n_particles)

        pz1, pz2 = np.outer(momenta,costheta) 
        py1, py2 = np.outer(momenta, np.sqrt(1.-costheta**2) * np.sin(phi))
        px1, px2 = np.outer(momenta, np.sqrt(1.-costheta**2) * np.cos(phi))
        e1, e2 = np.repeat(energies, n_particles).reshape(2,-1)

        P1 = np.array([px1, py1, pz1, e1]).T
        P2 = np.array([px2, py2, pz2, e2]).T

        #P1 = [LorentzVector(px[0][i], py[0][i], pz[0][i], energies[0]) for i in range(n_particles)]
        #P2 = [LorentzVector(-px[1][i], -py[1][i], -pz[1][i], energies[1]) for i in range(n_particles)]

        return [P1, P2]



    def decay_particle(self, mother_particle, diag_coupling, off_diag_coupling):

        parent_mass,_,_ = mother_particle.get_particle_properties()
        parent_momentum = mother_particle.get_momentum()
        product_masses = np.vectorize(Utility.get_mass)(self._product_IDs)
        n_particles = mother_particle.get_n_particles()

        Rotations = self._get_rotation(parent_momentum)
        fourvectors = self._determine_energy_momenta(parent_mass, product_masses, n_particles)

        product_particles = []
        weight = self._branching_function(diag_coupling, off_diag_coupling,**self._branching_kwargs) #dimension is [coupling]

        product_particles = []
        for i in range(2):
            rotated_vector = Rotations.apply(fourvectors[i][:,:3])
            fourvectors[i][:,:3] = rotated_vector
            product_particles.append(self._make_particle(self._product_IDs[i], LorentzVector(fourvectors[i]).boost(-1*parent_momentum.boostvector), 
                                                         np.outer(np.ones(n_particles), weight)))
        return product_particles

    def branching_ratio(self, diag_coupling, off_diag_coupling):
        return self._branching_function(diag_coupling, off_diag_coupling,**self._branching_kwargs) 

    def __str__(self):
        return f"Decay {self._label}: {self._parent_ID} to {self._product_IDs}"

    def save(self, path):
        pickle.dump(self, open(path, 'wr'))

    def save_all(self, dirpath):
        if not os.path.exists():
            os.makedirs(dirpath)
        filename = "/" + self._label + ".pkl"
        pickle.dump(self, open(filename, 'wb'))    
