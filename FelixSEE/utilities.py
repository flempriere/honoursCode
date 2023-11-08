import matplotlib
import matplotlib.pyplot as plt
import numpy as np 
from .lorentzVector import LorentzVector


class Utility():
    """
    A Singleton class designed to hold global parameters for simplicity
    """
    charges = {11 : -1, 13 : -1, 15: -1, -11: 1, -13: 1, -15 : 1,
               2212: 1, -2212: 1, 211: 1, 321: 1, 411: 1, 431: 1,
               -211: -1, -321: -1, -411: -1, -431: -1}
    masses = {2112: 0.938, -2112: 0.938, 2212: 0.938, -2212: 0.938,
              211: 0.13957, -211: 0.13957, 321: 0.49368, -321: 0.49368,
              310: 0.49761, 130: 0.49761, 111: 0.135, 221: 0.547,
              331: 0.957, 3122: 1.11568, -3122: - 1.11568, 3222: 1.18937,
              -3222: 1.18937, 3112: 1.19745, -3112: 1.19745, 3322: 1.31486,
              -3322: 1.31486, 3312: 1.32171, -3112: 1.32171, 3334: 1.67245,
              -3334: 1.67245, 113: 0.77545, 223: 0.78266, 333: 1.019461,
              213: 0.77545, -213: 0.77545, 411: 1.86961, -411: 1.86961,
              421: 1.86484, -421: 1.86484, 431: 1.96830, -431: 1.96830,
              4122: 2.28646, -4122: 2.28646, 511: 5.27961, -511: 5.27961,
              521: 5.27929, -521: 5.27929, 531: 5.36679, -531: 5.36679,
              541: 6.2749, -541: 6.2749, 4: 1.5, -4: 1.5, 5: 4.5, -5: 4.5,
              11: 0.000511, -11: 0.000511, 13: 0.105658, -13: 0.105658,
              15: 1.777, -15: 1.777, 22: 0, 23: 91., 24: 80.4, -24: 80.4,
              443: 3.096, 100443: 3.686, 553: 9.460, 100553: 10.023,
              200553: 10.355, 12: 0, -12: 0, 14: 0, -14:0, 16:0, -16:0}
    tau = {2112 : 1e+8, -2112 : 1e+8, 2212: 1e+8, -2212: 1e+8,
            211: 2.603e-8, -211: 2.603e-8, 321:1.238e-8, -321: 1.238e-8,
            310: 8.954e-11, 130: 5.116e-8, 3122: 2.60e-10, -3122: 2.60e-11,
            3222: 8.018e-11, -3222: 8.018e-11, 3112: 1.479e-10, -3112: 1.479e-10,
            3322: 2.90e-10, -3322: 1.639e-10, 3312: 1.639e-10, -3312: 1.639e-10,
            3334: 8.21e-11, -3334: 8.21e-11}
    
    dirpath = "../../"

    rng = np.random.default_rng(seed=12345)

    picobarn_2_femptobarn = 1000
    invGeVtoSeconds = 6.582e-25
    referenceCoupling = 1e-6

    def seed_rng(seed):
        """
        Set the seed for the rng stored in Utility
        """
        rng = np.random.default_rng(seed)

    def set_LLP_mass(mass):
        """Sets the mass of the LLP
        Input: mass (float)"""
        Utility.masses[0] = mass
    
    def set_LLP_tau(tau):
        """
        Sets the ctau of the LLP
        Input: ctau (float)
        """
        Utility.tau[0] = tau
    
    def get_charge(pid):

        """Returns the charge of a particle
        Input: pid (Int)
        Output: charge (Int)
        """
        charge = Utility.charges.get(pid)
        if charge is None:
            charge = 0
        return charge
        
    def get_mass(pid):
        """
        Returns the mass of a given particle

        Input: pid (int) - id of the particle
        Output: mass (float) - mass of the particle in GeV
        """
        particle_mass = Utility.masses.get(pid)
        if particle_mass is None:
            print("Invalid particle ID")
        return particle_mass
    
    def get_ctau(pid):
        """
        Returns c x the lifetime of a particle
        
        Input: pid (int)
        Output: ctau (float)"""
        tau = Utility.tau.get(pid)
        if tau is None:
        #    print("Invalid particle ID")
            return tau
        return 3.0e+8 * tau

    def get_particle_params(pid):
        """
        Given a particle id, returns the mass, charge and ctau
        Input: pid (int)
        Output: mass, charge, ctau (float, int, float)
        """
        mass = Utility.get_mass(pid)
        charge = Utility.get_charge(pid)
        ctau = Utility.get_ctau(pid)
        return mass, charge, ctau
    
    def get_uniform_random_numbers(hi, lo, nsamples=1):
        return (hi - lo)*Utility.rng.random(nsamples) + lo
    
#this should realistically all be put into the Simulation class and passed down as needed but yay...

class DataHandler():
    """Static class for data handling such as file reading"""
    def convert_list_to_momenta(filename,mass,filetype="txt",nsample=1,preselectioncut=None, nocuts=False, extend_to_low_pt_scale=False):
        if filetype=="txt":
            logth, logp, xs = DataHandler.readfile(filename).T
        elif filetype=="npy":
            logth, logp, xs = np.load(filename)
        else:
            print ("ERROR: cannot rtead file type")
        if extend_to_low_pt_scale:
            list_xs = DataHandler.extend_to_low_pt(logth, logp, xs, ptmatch=extend_to_low_pt_scale)

        


        p = np.power(10., logp)
        p = np.repeat(p, nsample)

        total_samples = p.size

        th = np.power(10., logth)
        th = np.repeat(th, nsample)

        pt = p*np.sin(th)

        xs = np.repeat(xs, nsample)
        if nocuts==False:
            xs[xs < 10e-6] = 0.
            if preselectioncut is not None:
                xs[~preselectioncut(p)] = 0
        xs /= nsample

        phi = Utility.get_uniform_random_numbers(-np.pi, np.pi, total_samples)
        fth = np.power(10, Utility.get_uniform_random_numbers(-0.025, 0.025, total_samples))
        fp = np.power(10, Utility.get_uniform_random_numbers(-0.025, 0.025, total_samples))

        th_sm = th*fth
        p_sm = p*fp
        en = np.sqrt(np.power(p_sm, 2) + np.power(mass, 2))
        pz = p_sm*np.cos(th_sm)
        pt = p_sm*np.sin(th_sm)
        px = pt*np.cos(phi)
        py = pt*np.sin(phi)

        particles = LorentzVector(np.vstack([px, py, pz, en]).T) #dimension is [nparticles] x [4]
        
        return particles, xs
    
    def readfile(filename):
        array = []
        with open(filename) as f:
            for line in f:
                if line[0]=="#":continue
                words = [float(elt.strip()) for elt in line.split( )]
                array.append(words)
        return np.array(array)
    
    def extend_to_low_pt(list_t, list_p, list_w, ptmatch=0.5, navg=2):

        # round lists and ptmatch(so that we can easily search them)
        list_t = [round(t,3) for t in list_t]
        list_p = [round(p,3) for p in list_p]
        l10ptmatch = round(round(np.log10(ptmatch)/0.05)*0.05,3)

        # for each energy, get 1/theta^2 * dsigma/dlog10theta, which should be constant
        logps = np.linspace(1+0.025,5-0.025,80)
        values = {}
        for logp in logps:
            rlogp = round(logp,3)
            rlogts = [round(l10ptmatch - rlogp + i*0.05,3) for i in range(-navg,navg+1)]
            vals = [list_w[(list_p==rlogp)*(list_t==rlogt)][0]/(10**rlogt)**2 for rlogt in rlogts]
            values[rlogp] = np.mean(vals)

        # using that, let's extrapolate to lower pT
        list_wx = []
        for logt, logp, w in zip(list_t, list_p, list_w):
            rlogp, rlogt = round(logp,3), round(logt,3)
            if  logt>l10ptmatch-logp-2.5*0.05 or logp<1:list_wx.append(w)
            else:list_wx.append(values[rlogp]*(10**rlogt)**2)

        #return results
        return list_wx
    
    def convert_to_hist_list(momenta,weights, do_plot=False, filename=None, do_return=False, prange=[[-6, 0, 80],[ 0, 5, 80]], vmin=None, vmax=None):

        #momenta should be a LorentzVector Object [nparticles][4] 
        #weights should be 
        tmin, tmax, tnum = prange[0]
        pmin, pmax, pnum = prange[1]
        t_edges = np.logspace(tmin, tmax, num=tnum+1)
        p_edges = np.logspace(pmin, pmax, num=pnum+1)

        tx = np.arctan(momenta.pt/momenta.z)
        px = momenta.p

        w, t_edges, p_edges = np.histogram2d(tx, px, weights=weights,  bins=(t_edges, p_edges))

        t_centers = np.logspace(tmin+0.5*(tmax-tmin)/float(tnum), tmax-0.5*(tmax-tmin)/float(tnum), num=tnum)
        p_centers = np.logspace(pmin+0.5*(pmax-pmin)/float(pnum), pmax-0.5*(pmax-pmin)/float(pnum), num=pnum) 

        list_t = np.repeat(np.log10(t_centers), pnum)
        list_p = np.tile(np.log10(p_centers), tnum)
        list_w = w.flatten() 

        #for it,t in enumerate(t_centers):
        #    for ip,p in enumerate(p_centers):
        #        list_t.append(np.log10 ( t_centers[it] ) )
        #        list_p.append(np.log10 ( p_centers[ip] ) )
        #        list_w.append(w[it][ip])

        if filename is not None:
            print ("save data to file:", filename)
            np.save(filename,[list_t,list_p,list_w])
        if do_plot==False:
            return list_t,list_p,list_w

        #get plot
        ticks = np.array([[np.linspace(10**(j),10**(j+1),9)] for j in range(-7,6)]).flatten()
        ticks = np.log10(ticks)
        ticklabels = np.array([[r"$10^{"+str(j)+"}$","","","","","","","",""] for j in range(-7,6)]).flatten()
        matplotlib.rcParams.update({'font.size': 15})
        #fig = plt.figure(figsize=(8,5.5))
        fig = plt.figure(figsize=(7,5.5))
        ax = plt.subplot(1,1,1)
        h=ax.hist2d(x=list_t,y=list_p,weights=list_w,
                    bins=[tnum,pnum],range=[[tmin,tmax],[pmin,pmax]],
                    norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax), cmap="rainbow",
        )
        fig.colorbar(h[3], ax=ax)
        ax.set_xlabel(r"angle wrt. beam axis $\theta$ [rad]")
        ax.set_ylabel(r"momentum $p$ [GeV]")
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels)
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticklabels)
        ax.set_xlim(tmin, tmax)
        ax.set_ylim(pmin, pmax)

        return plt, list_t,list_p,list_w
    
