'''
Ryan Raikman, Deep Learning for Mid-Band Gravitational Wave Data Analysis
Code to generate GW strain on a detector for given parameters
'''

# Importing all relevant libraries
import numpy as np
import copy
import scipy.signal as sig
import time
import os
import random

# Helper function definition

def init_constants():
    global G, c, Msun, Mpc_cm
    G = 6.67e-8; c = 2.9972e10
    Msun = 1.989e33; Mpc_cm = 3.086e24
    
def init_diffeqs():
    global fe1, fe2, P
    fe1 = lambda e: (1.+73./24.*e*e+37./96.*e**4.)/(1.-e*e)**3.5
    fe2 = lambda e: e * (1.+ 121./304.*e*e)/(1.-e*e)**2.5
    P = lambda a : np.sqrt(4.*np.pi**2./(G*M)*a**3.)
    
    global dadt, dedt, dnudt, domdt
    dadt = lambda a, e, nu, om : -64./5.*G**3./c**5.*m1*m2*M/a**3. * fe1(e)
    dedt = lambda a, e, nu, om : -304./15.*G**3./c**5. * m1*m2*M/a**4. * fe2(e)
    dnudt = lambda a, e, nu, om : np.sqrt(G*M*a*(1.-e*e))*(1.+e*np.cos(nu))**2./(a*a*(1.-e*e)**2.)
    domdt = lambda a, e, nu, om : 3.*G**(2./3.)/c/c*(2.*np.pi/P(a))**(5./3.)*M**(2./3.)/(1.-e*e)

def calc_a(m1, m2, e, f):
    top = np.sqrt(G * (m1 + m2)) * ((1+e) ** 1.1954)
    bot = np.pi * ((1-e*e)**1.5) * f
    return (top / bot) ** (2/3)

def map_diffeqs(conds):
    #conds = [a, e, nu, om]
    a, e, nu, om = conds
    ka = dadt(a, e, nu, om)
    ke = dedt(a, e, nu, om)
    knu = dnudt(a, e, nu, om)
    kom = domdt(a, e, nu, om)
    
    return [ka, ke, knu, kom]

def update_values(conds, derivs, dt, factor):
    #conds = [a, e, nu, om] - initial
    #derivs = [ka, ke, knu, kom]
    #dt - time step
    #factor - 1 or 2, for rk4  
    updated_vals = []
    for i in range(len(conds)):
        x = conds[i]
        dxdt = derivs[i]
        xp = x + dt * dxdt / factor
        updated_vals.append(xp)
        
    return updated_vals

def rk4_combine(derivs):
    weightmap = {
        0 : 1/6,
        1 : 1/3,
        2 : 1/3,
        3 : 1/6
    }
    #note now that derivs is array of arrays
    kf = [0, 0, 0, 0]
    
    for i, ki in enumerate(derivs):        
        for j, kij in enumerate(ki):
            weight = weightmap[i]
            kf[j] += weight * kij
            
    return kf  

def step_rk4(v0, dt): 
    #v0 = [a, e, nu, om]
    k0 = map_diffeqs(v0)

    v1 = update_values(v0, k0, dt, 2)
    k1 = map_diffeqs(v1)
    
    v2 = update_values(v1, k1, dt, 2)
    k2 = map_diffeqs(v2)
    
    v3 = update_values(v2, k2, dt, 1)
    k3 = map_diffeqs(v3)
    
    kf = rk4_combine([k0, k1, k2, k3])
    vf = update_values(v0, kf, dt, 1)
    
    return vf

def update(params, hists):
    for i, hist in enumerate(hists):
        hist.append(params[i])

def mc(m1, m2):
    return ((m1 * m2) ** 0.6) / ( (m1 + m2)**0.2 )

def waveform_main(m1_, m2_, e, D, f0, fend, dt, dt_interp = 0.01, timer = False): 
    '''
    INPUTS:
    m1_: mass of first object
    m2_: mass of second object
    e: starting eccentricity
    D: luminosity distance to source
    f0: starting gravitational wave frequency
    fend: ending gravitationl wave frequency
    dt: time step for performing integration
    dt_interp: time step for interpolation of variables and computing the resulting waveform
    timer: boolean, show how long each step takes

    OUTPUTS:
    hxy: final waveform strain
    es: eccentricity evolution of the signal
    '''
    #initialize global parameters
    init_constants()
   
    global m1, m2, M, mu0, D0
    m1 = m2 = M = mu0 = D0 = 0
    m1, m2 = m1_ * Msun, m2_ * Msun
    M = m1 + m2; mu0 = m1 * m2 / M
    D0 = D * Mpc_cm
    
    #calculate corresponding separations
    a0 = calc_a(m1, m2, e, f0)
    a_end = calc_a(m1, m2, 0, fend) 
    
    #initialize differential equations
    init_diffeqs()
    
    #initial parameters
    a, e, om, nu = a0, e, 0, 0
    t = 0

    #perform approximation
    ts_intergration_0 = time.time()
    t_1_day = 3600*24

    hist_np = np.zeros((int(t_1_day/dt)+1, 5))
    hist_np[0] = [a, e, om, nu, t]
    i=1

    while (t <= t_1_day and a > a_end):
        a, e, om, nu = step_rk4([a, e, om, nu], dt)
        t += dt

        hist_np[i] = [a, e, om, nu, t]
        i += 1

    hist_np = hist_np[:i]
    ts_integration_1 = time.time()
    ts_interpolation_0 = time.time()

    ts = np.arange(0, t, dt_interp)

    #formatting all relevant quantities
    aas, es, nus, oms, ts_old = np.hsplit(hist_np, 5)
    aas = np.squeeze(aas, axis = 1)
    es = np.squeeze(es, axis = 1)
    nus = np.squeeze(nus, axis = 1)
    oms = np.squeeze(oms, axis = 1)
    ts_old = np.squeeze(ts_old, axis = 1) #don't really need this for anything

    aas = np.interp(ts, ts_old, aas)
    es = np.interp(ts, ts_old, es)
    nus = np.interp(ts, ts_old, nus)
    oms = np.interp(ts, ts_old, oms)
    
    
    ts_interpolation_1 = time.time()

    ts_waveform_0 = time.time()
    ths = oms + nus
    rs = aas*(1.-es*es)/(1.+es*np.cos(nus))
    Ps = np.sqrt(4.*np.pi**2./(G*M)*aas**3.) 
    
    #initialization for kernel used to smooth waveform over multiple differentiations
    kernel_mult = int(0.01/dt_interp)
    if kernel_mult < 1:
        kernel_mult = 1
    km = int(kernel_mult)
    k = np.ones(20)
    gauss = lambda x: (1.05)** (-(x-10)*(x-10))   
    
    for i in range(20):
        k[i] = gauss(i)
    
    kernel_g = np.array(k)
    kernel = kernel_g / np.sum(kernel_g)

    grad1 = np.gradient(mu0*rs*rs*np.cos(ths)*np.sin(ths), ts)

    smooth = 1
    if smooth:
        grad1 = np.convolve(grad1, kernel, mode = 'same')

    hxy = 2.*G/c**4./D0*np.gradient(grad1, ts)
    smooth2 = 1
    if smooth2:
        hxy = np.convolve(hxy, kernel, mode = "same")
    hxy = hxy[np.isfinite(hxy)]
    ts_waveform_1 = time.time()
    
    
    if timer:
        t_int = round(ts_integration_1 - ts_intergration_0, 5)
        t_interp = round(ts_interpolation_1 - ts_interpolation_0, 5)
        t_wave = round(ts_waveform_1 - ts_waveform_0, 5)
        
        print("Timetable:")
        print(f"Integration:   {t_int}s \nInterpolation: {t_interp}s \nWaveform:      {t_wave}s")
    
    return hxy, es

def sample_parameters(n_trials, m_min, m_max, e_min, e_max):
    '''
    INPUT:
    n_trials: number of waveforms you wish to construct
    m_min: minimum mass of individual black hole
    m_max: maximum mass of individual black hole
    e_min: minimum starting eccentricity
    e_max: maximum starting eccentricity

    OUTPUT:
    masses: sampled binary masses
    eccens: sampled starting eccentricities
    '''
    masses = np.random.uniform(m_min, m_max, shape=(n_trials, 2))
    eccens = np.random.uniform(e_min, e_max, shape=n_trials)
    
    return masses, eccens

# Parameters used in our work
n_trials = 100 #note that this code was run multiple times. If you wish to generate a single waveform,
               #simply set n_trials to 1, and use the waveform mode. You can hand-pick parameters for 
               #masses and eccens, bypassing the sample_parameters code, generally used for creating 
               #many signals for NN training.
m_min = 20
m_max = 50
mc_max = mc(m_max, m_max)
e_min = 0
e_max = 0.3
f0 = 1
fend = 3
dt_approx = 0.1
dt_interp = 0.01
D = 200 # units of Mpc
masses, eccens = sample_parameters(n_trials, m_min, m_max, e_min, e_max)

mode = ["psd", "waveform"][0] #both are valid options
noise_amp = 1e-22
noise = True #valid options are True, False

save_data = [] #holds all generated data - either each segmented PSD or the entire waveforms
params = [] #holds corresponding parameters, either Mc and current eccentricity for segmented PSD, 
            #or entire eccentricity history and chirp mass for the waveforms
for i in range(n_trials):
    m1_i, m2_i = masses[i]
    e_i = eccens[i]
    
    hxy, e_hist = waveform_main(m1_i, m2_i, e_i, D, f0, fend, dt_approx, dt_interp, timer = False)

    if noise:
        hxy += np.random.normal(0, noise_amp, len(hxy))
    
    if mode == "psd":
        fs = int(1/dt_interp)
        split_dur = 30 * fs #30 seconds
        max_splits = int((len(hxy) / split_dur)) - 1

        for s in range(max_splits):
            start = int(split_dur * i)
            end = int(split_dur * (i+1))

            e_cur = e_hist[(start+end)//2] #sample eccentricity in middle of segment

            indiv_psd = sig.welch(hxy[start:end], fs = fs)[1]

            #normalize, for NN training, and only take first 20 out of 129 datapoints
            indiv_psd = np.log(indiv_psd[:20])

            save_data.append(indiv_psd)
            
            #normalize parameters for NN training
            params.append([mc(m1_i, m2_i)/mc_max, e_cur/e_max])

    elif mode == "waveform":
        #generally used for plotting/visualization of signal, so no need to normalize parameters
        save_data.append(hxy)
        params.append([e_hist, mc(m1_i, m2_i)])



