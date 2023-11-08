from .utilities import Utility, DataHandler
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from .particle import Particle
from .lorentzVector import LorentzVector
import os
import pickle



class Experiment():
    def __init__(self, label="FASER2", distance=480, acceptance_function=None, constraint_function=None, length=5, luminosity=3000,number_density=3.754e+29, ermin=0.03, ermax=1,
                 sources = [], decays = {}):
        #detector profile
        self._label = label
        self._distance= distance
        self._length = length
        self._acceptance_function=acceptance_function

        #function to modulate weight by taking into account geometric structure of LHC
        self._constraint_function=constraint_function

        #LHC profile
        self._luminosity = luminosity
        self._number_density = number_density

        #unused for now - used for direct production I believe
        self._ermin = ermin
        self._ermax = ermax

        #decay procedures
        self._sources = sources
        self._decays = decays
        self._particles = []
        self._decay_products = []
    
    def sources(self):
        return [src.get_pid() for src in self._sources]
    
    def label(self):
        return self._label

    def check_selection(self, particle):
        """
        returns probability that a particle decays in the detector.
        """
        #return 1
    
        if self._acceptance_function is None: return 1.
        zaxis = np.array([0,0,1])
        momentum = particle.get_momentum()

        particle_mass,_,_ = particle.get_particle_properties()
        particle_ctau = Utility.get_ctau(particle.get_ID()) #[coupling]
        #n_particles = particle.get_n_particles()
        
        #check that the particle is aligned with the detector
        x_start = momentum*((self._distance/momentum.z)[:, np.newaxis])
        x_end = momentum*(((self._distance + self._length)/momentum.z)[:, np.newaxis])
        costheta = np.cos(momentum.angle(zaxis))

        x_start = self._acceptance_function(x_start)
        x_end = self._acceptance_function(x_end)
        probabilities = np.logical_and(x_start, x_end).astype(int)

        #calculate probability that it then decays in the volume
        dbar = (particle_ctau*momentum.p[:, np.newaxis]/particle_mass)
        effective_length = dbar*costheta[:, np.newaxis]
        #now need to mask out those outside of acceptance window
        decay_prob = probabilities[:, np.newaxis]*effective_length.copy()
        decay_prob[decay_prob > 0] = np.exp(-(self._distance)/effective_length[decay_prob > 0])*(1 - np.exp(-self._length/effective_length[decay_prob > 0]))
        return decay_prob

    def check_constraints(self, particle):
        """ Still needs to be implemented -> modify particle weight to factor in decay must proceed before any geometric constraints"""
        return 1
    
    def _update_decay_parameters(self, update_dict):
        for decay in self._decays:
            decay.update_branching_kwargs(update_dict)

    def generate_LLP_spectrum(self, mass, reload_sources=False, filenames='default', do_plot=False, do_save=True):
        self.initialise(mass, Utility.referenceCoupling, Utility.referenceCoupling, reload_sources=reload_sources)
        momenta, weights = self.decay_particles(Utility.referenceCoupling, Utility.referenceCoupling, do_plot)
        if do_save:
            dirname = Utility.dirpath+"/model/LLP_spectra/"
            if not os.path.exists(dirname): os.makedirs(dirname)
            if filenames == 'default':
                filenames = ["14TeV_m_" + str(float(mass)) + "_" + str(src.get_pid()) + ".npy" for src in self._sources] #annoying loop over sources twice...
            for momentum, weight, filename in zip(momenta, weights, filenames):
                if momentum is None: continue
                filenamesave = dirname + filename
                DataHandler.convert_to_hist_list(momentum, weight.squeeze(), do_plot=False, filename=filenamesave)

    
    def initialise(self, mass, diagonal_coupling, off_diagonal_coupling, reload_sources=False):

        Utility.set_LLP_mass(mass)
        update_dict = {'mass' : mass}
        diagonal_coupling, off_diagonal_coupling = np.asarray(diagonal_coupling), np.asarray(off_diagonal_coupling)


        decay_width = np.zeros(np.maximum(diagonal_coupling.size, off_diagonal_coupling.size))
        branching_ratio = np.zeros(np.maximum(diagonal_coupling.size, off_diagonal_coupling.size))
        for pid in self._decays.keys():
            for decay in self._decays.get(pid):
                decay.update_branching_kwargs(update_dict)
                if pid == 0:
                    decay_width += decay.branching_ratio(diagonal_coupling, off_diagonal_coupling)
                    if decay.is_visible():
                        branching_ratio += decay.branching_ratio(diagonal_coupling, off_diagonal_coupling)
        
        lifetime = (1./decay_width)*Utility.invGeVtoSeconds
        Utility.set_LLP_tau(lifetime)
        self._branching_ratio = (branching_ratio / decay_width) #branching ratio is thus a numpy array of dimension [coupling].

        if (reload_sources):   
            self.particles = []
            for source in self._sources:
                    source.load_file()
                    self.particles.append(source.get_particles())

    def decay_particles(self, diagonal_coupling, off_diagonal_coupling, do_plot=False):
        self._decay_products = []
        momenta, weights = [], []

        for particle in self.particles: #decay all particles
            src_momenta, src_weights = [], []
            geometric_weight = self.check_constraints(particle) #dimension is [n_particles] or scalar
            luminosity_factor = self._luminosity*Utility.picobarn_2_femptobarn 
            experiment_weight = geometric_weight*luminosity_factor
            decays = self._decays.get(particle.get_ID())
            if decays is None: continue
            for decay_channel in decays: #apply to all decays
                new_products = decay_channel.decay_particle(particle, diagonal_coupling, off_diagonal_coupling)
                for product in new_products:
                    if not product.get_ID() == 0: continue
                    product.set_weight((particle.get_weight()*product.get_weight().T).T*experiment_weight) #combine weights dimension now [n_particles][coupling  ]
                    m, w = product.get_statistical_properties()
                    src_momenta.append(m)
                    src_weights.append(w)
            if not len(src_momenta) == 0:
                m_tot = src_momenta[-1].combine(momenta[:-1])
                w_tot = np.vstack(src_weights)
                momenta.append(m_tot)
                weights.append(w_tot)
                self._decay_products.append(Particle(0, m_tot, w_tot))
            else: #if for a given source there are no relevant decays fill with None -> should never occur
                self._decay_products.append(None)
                momenta.append(None)
                weights.append(None)
            #end result is we have a list [ALPs] [momenta], [weights] where each is for a given source....
        if do_plot:
            #remember to remove plotting stuff!
            for src, m, w in zip(self._sources, momenta, weights):
                if m is None: continue
                index = 0
                plot = DataHandler.convert_to_hist_list(m, w.T[index], do_plot=True, prange=[[-5,0, 200], [0,4, 200]])[0]
                plot.text(-2.4,3.75, r"$a$ at 14TeV LHC [pb/bin]",fontsize=15,color="k",rotation=0)
                plot.text(-2,3.5, r"From {0} via Pythia8".format(src.get_pid()),fontsize=15,color="k",rotation=0)
                #plot.savefig("/Users/felixkling/Downloads/Figure.pdf")
                plot.subplots_adjust(left=0.10, right=1.05, bottom=0.10, top=0.97)

                plot.show()
        return momenta, weights
    
    def count_events(self, diagonal_coupling, off_diagonal_coupling, do_plot=True, scaling=1):

        diagonal_coupling, off_diagonal_coupling = np.asarray(diagonal_coupling), np.asarray(off_diagonal_coupling)
        n_signals = np.zeros(np.maximum(diagonal_coupling.size, off_diagonal_coupling.size))

        stat_p, stat_w, src_sig = [], [], []
        for particle in self._decay_products:
                momenta, weight = particle.get_statistical_properties() #momenta is lorentzVector of dim [n_particles][4], weight is 2darray of shape [n_particles][couplings] or 1d of shape [n_particles]
                decay_in_detector_probability = self.check_selection(particle) #numpy ndarray with dimensionality [nparticles][couplings]
                particle_signals = weight.reshape((len(momenta),-1))*(decay_in_detector_probability*self._branching_ratio*scaling)
                particle_signals[particle_signals < 1e-7] = 0
                sigs = np.sum(particle_signals, axis=0)
                src_sig.append(sigs)
                n_signals += sigs
                nonzero_sig = np.any(particle_signals, axis=1)
                if (np.any(nonzero_sig)):
                    stat_p.append(LorentzVector(momenta.fourVector[nonzero_sig]))
                    stat_w.append(particle_signals[nonzero_sig])
                else:
                    stat_p.append(None)
                    stat_w.append(None)

        # now we have #n_signals = #[coupling] <- total signals for a given coupling
        # stat_p = list of momenta for which there is a nonzero sig per source
        # stat_w = list of arrays each of which is [nonzero_momenta][coupling] dimension corresponding to signals for the given momenta per source
        # src_sig  #list of arrays of shape [coupling] <- total number of sigs for a given coupling for a specific source.
        if do_plot:
            for src, sig, m, w in zip(self._sources, src_sig, stat_p, stat_w):
                plot = DataHandler.convert_to_hist_list(m, w.T[3], do_plot=True, prange=[[-5,0, 200], [0,4,200]])[0]
                plot.title(r"$a$ signals in FASER\n [detections/bin]",fontsize=15,color="k",rotation=0)
                plot.text(3,3.75, r"From ${0}$ via Pythia8".format(src.get_pid()),fontsize=15,color="k",rotation=0)
                plot.text(3,3.5, "total signals = {0:3e}".format(sig[3]))
                plot.subplots_adjust(left=0.10, right=1.05, bottom=0.10, top=0.97)
                #plot.savefig("/Users/felixkling/Downloads/Figure.pdf")
                plot.show()

        return n_signals, src_sig, stat_p, stat_w
    
    def load_LLP_spectrum(self, filenames="default", n_samples=1, preselectioncut=None, nocuts=False,extend_to_low_pt_scale=False, do_plot=False):
        self._decay_products = []
        if filenames == "default":
            filenames = [Utility.dirpath + "/model/LLP_spectra/14TeV_m_" + str(float(Utility.get_mass(0))) + "_" + str(src.get_pid()) + ".npy" for src in self._sources]
        for filename in filenames:
            momenta, weights = DataHandler.convert_list_to_momenta(filename, Utility.get_mass(0), 'npy', n_samples,
            preselectioncut, nocuts, extend_to_low_pt_scale)
            if do_plot:
                plot = DataHandler.convert_to_hist_list(momenta, weights, do_plot=True, prange=[[-6,0, 120], [0,5, 100]])[0]
                plot.text(-3,3.75, r"$a$ signals in FASER\n [detections/bin]",fontsize=15,color="k",rotation=0)
                plot.text(-3,3.5, r"From $B^{0}$ via Pythia8",fontsize=15,color="k",rotation=0)
                plot.subplots_adjust(left=0.10, right=1.05, bottom=0.10, top=0.97)
            #plot.savefig("/Users/felixkling/Downloads/Figure.pdf")
                plot.show()
            self._decay_products.append(Particle(0, momenta, weights))

               
    def run_experiment(self, mass, diagonal_coupling, off_diagonal_coupling, reload_sources=False,scale_results=0,plot_LLP_spectrum=False, plot_detections=False):
        self.initialise(mass, diagonal_coupling, off_diagonal_coupling, reload_sources)
        if scale_results == 0:
            self.decay_particles(diagonal_coupling, off_diagonal_coupling, do_plot=plot_LLP_spectrum)
        return self.count_events(diagonal_coupling, off_diagonal_coupling, do_plot=plot_detections, scaling=((diagonal_coupling/Utility.referenceCoupling)**scale_results))
    
    def save(self, path):
        pickle.dump(self, open(path, 'wb'))
    
    def save_all(self, dirpath):
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        filename = "experiment_"+str(self._label) + ".pkl"
        self.save(dirpath + "/" + filename)
        if not os.path.exists(dirpath + "/sources"):
            os.makedirs(dirpath)
        if len(self._decay_products > 0):
            if not os.path.exists(dirpath + "/LLP_spectra"):
                os.makedirs()
            for src, product in zip(self._sources, self._decay_products):
                src.save_all(dirpath + "/sources" + src.get_pid())
                product.save_all(dirpath + "/LLP_spectra/" + src.get_pid())
        else:
            for src, in self._sources:
                src.save_all(dirpath + "/sources" + src.get_pid())