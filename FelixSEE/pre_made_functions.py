import numpy as np

def Kallen_function(x, y, z):
    return np.maximum(x**2 + y**2 + z**2 - 2*x*y - 2*x*z - 2*y*z, 0)

def meson_decay(diagonal_coupling, off_diagonal_coupling, **kwargs):
    mass = kwargs["mass"]
    B_meson_mass = kwargs["B_meson_mass"]
    vev = kwargs["vev"]
    if mass > B_meson_mass: return 0
    return 4*(diagonal_coupling**2)*((vev)**2)* (3.1 * (1 - np.power(mass/B_meson_mass, 2)) + 3.7 * np.power((1 - np.power(mass/B_meson_mass, 2)),3))

#setting up axion decays
def fermion_decay(mass, coupling, m1, m2):
    k_f = np.sqrt(Kallen_function(1, (m1/mass)**2, (m2/mass)**2))
    prefactor = (coupling**2)/(8*np.pi)
    mass_factor = mass*((m1 + m2)**2)*np.maximum([1 - (m1 - m2)**2/mass**2], 0)
    return (prefactor*mass_factor*k_f)

def diagonal_fermion_decay(diagonal_coupling, off_diagonal_coupling,**kwargs):
    mass = kwargs["mass"]
    coupling = diagonal_coupling
    m1 = kwargs["m1"]
    m2 = kwargs["m2"]
    return fermion_decay(mass, coupling, m1, m2)

def off_diagonal_fermion_decay(diagonal_coupling, off_diagonal_coupling, **kwargs):
    mass = kwargs["mass"]
    coupling = off_diagonal_coupling
    m1 = kwargs["m1"]
    m2 = kwargs["m2"]
    return fermion_decay(mass, coupling, m1, m2)

def preselection_cut(momenta, pow=1):
    return (momenta > 10**pow)

lepton_masses = {
    "e" : 0.510e-3,
    "mu" : 106e-3,
    "tau" : 1780e-3}

quark_masses = {
    #"d" : 28.81e-3,
    #"u" : 105.4e-3,
    #"s" : 385.9e-3,
    "c" : 1412e-3,
    "b" : 5169e-3
}


lepton_ids = {
    "e" : 11,
    "mu" : 13,
    "tau" : 15
    }
quark_ids = {
    #"d" : 1,
    #"u" : 2,
    #"s" : 3,
    "c" : 4,
    "b" : 5
}