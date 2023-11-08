from .utilities import Utility, DataHandler
import os
import numpy as np
import pickle
from tqdm.auto import tqdm

class simulation():
    def __init__(self, experiment, dirpath):
        self._experiment = experiment
        Utility.dirpath = dirpath
        self._runs = 0
    
    def generate_LLP_spectra(self, mass_range, do_plot=False):
        self._experiment.initialise(1,Utility.referenceCoupling,Utility.referenceCoupling, reload_sources=True)
        for mass in mass_range:
            print(f"generating LLP spectra for mass: {mass}")
            self._experiment.generate_LLP_spectrum(mass, reload_sources=False, do_plot=do_plot)
    
    def vary_off_diagonal(self, mass_range, diagonal_coupling, off_diagonal_coupling,
                          n_samples=100, preselectioncut=None, nocuts=False,extend_to_low_pt_scale=False, do_save=True):
        results = []
        source_ids = self._experiment.sources()
        experiment_label = self._experiment.label()

        if do_save:
            results_dir = Utility.dirpath + "/model/results"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

        for mass in mass_range:
            mass_array = mass*np.ones(len(off_diagonal_coupling))
            Utility.set_LLP_mass(mass)
            dirname = Utility.dirpath+"/model/LLP_spectra"
            filename = "default"
            
            self._experiment.load_LLP_spectrum(filename, n_samples, preselectioncut, nocuts, extend_to_low_pt_scale)
            counts, _, stat_p, stat_w = self._experiment.run_experiment(mass, diagonal_coupling, off_diagonal_coupling, scale_results=2)

            results.append(np.array([mass_array, np.array(off_diagonal_coupling), counts]).T)
            #counts is [coupling] array, src_count is list of [src]([coupling]), stat_p is [src]([n_particles][coupling]), stat_w = [src]([n_particles][coupling])
            #for now we won't bother saving all the source results seperately...
        results = np.vstack(results)
        if do_save:
            np.save(results_dir + "/results_" + str(experiment_label) + "_diagonal_fixed_" + str(diagonal_coupling) + "_run_" + str(self._runs) + ".npy", results)

        self._runs += 1
        return results
    
    def vary_on_diagonal(self, mass_range, diagonal_coupling, off_diagonal_coupling,
                            n_samples=100, preselectioncut=None, nocuts=False,extend_to_low_pt_scale=False, do_save=True):
        results = []
        source_ids = self._experiment.sources()
        experiment_label = self._experiment.label()


        if do_save:
            results_dir = Utility.dirpath + "/model/results"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

        for mass in mass_range:
            mass_array = mass*np.ones(len(diagonal_coupling))
            Utility.set_LLP_mass(mass)
            dirname = Utility.dirpath+"/model/LLP_spectra"
            filename = "default"
            self._experiment.load_LLP_spectrum(filename, n_samples, preselectioncut, nocuts, extend_to_low_pt_scale)
            counts, _, stat_p, stat_w = self._experiment.run_experiment(mass, diagonal_coupling, off_diagonal_coupling, scale_results=2)

            results.append(np.array([mass_array, np.array(diagonal_coupling), counts]).T)
            #counts is [coupling] array, src_count is list of [src]([coupling]), stat_p is [src]([n_particles][coupling]), stat_w = [src]([n_particles][coupling])
            #for now we won't bother saving all the source results seperately...
        results = np.vstack(results)
        if do_save:
            np.save(str(results_dir) + "/results_" + str(experiment_label) + "_off_diagonal_fixed_" + str(off_diagonal_coupling) + "_run_" + str(self._runs) + ".npy", results)

        self._runs += 1
        return results
    
    def vary_both_coupling(self, mass_range, coupling,
                            n_samples=100, preselectioncut=None, nocuts=False,extend_to_low_pt_scale=False, do_save=True):
        results = []
        #source_ids = self._experiment.sources()
        experiment_label = self._experiment.label()

        if do_save:
            results_dir = Utility.dirpath + "/model/results"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)


        for mass in tqdm(mass_range):
            mass_array = mass*np.ones(len(coupling))
            Utility.set_LLP_mass(mass)
            dirname = Utility.dirpath+"/model/LLP_spectra"
            #filename = dirname+"/14TeV_m_"+str(mass)+".npy"
            filename = "default"
            self._experiment.load_LLP_spectrum(filename, n_samples, preselectioncut, nocuts, extend_to_low_pt_scale)
            counts, _, stat_p, stat_w = self._experiment.run_experiment(mass, coupling, coupling, scale_results=2)

            results.append(np.array([mass_array, np.array(coupling), counts]).T)
            #counts is [coupling] array, src_count is list of [src]([coupling]), stat_p is [src]([n_particles][coupling]), stat_w = [src]([n_particles][coupling])
            #for now we won't bother saving all the source results seperately...
        results = np.vstack(results)
        if do_save:
            np.save(results_dir + "/results_" + str(experiment_label) + "_run_" + str(self._runs) + ".npy", results)

        self._runs += 1
        return results
    
    def save(self, filename):
        pickle.dump(self, open(filename, 'wb'))

    def save_all(self):
        save_dir = Utility.dirpath + "/model"
        filename = "/simulation_" + self._experiment.label() + ".pkl"
        self.save(save_dir + filename)
        self._experiment.save_all(save_dir)