import numpy as np
import dill
import os
from scipy.integrate import odeint
from scipy import signal
from scipy.stats import wasserstein_distance
import torch
import glob
from functools import partial
from dask_jobqueue import SLURMCluster
import dask
from distributed import Client
from sbi import utils as sbi_utils
from sbi import analysis as sbi_analysis
from sbi import inference as sbi_inference
from sklearn.decomposition import PCA
import scipy
from scipy.signal import periodogram, welch
from sklearn.linear_model import LinearRegression
from fooof import FOOOF
from torch import optim

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

from hnn_core import jones_2009_model, simulate_dipole, pick_connection
from hnn_core.params import _short_name
from hnn_core.network import _connection_probability

rng_seed = 123
rng = np.random.default_rng(rng_seed)
torch.manual_seed(rng_seed)
np.random.seed(rng_seed)

device = 'cpu'
num_cores = 256

def run_hnn_sim(net, param_function, prior_dict, theta_samples, tstop, save_path, save_suffix, theta_extra=dict()):
    """Run parallel HNN simulations using Dask distributed interface
    
    Parameters
    ----------
    net: Network object
    
    param_function: function definition
        Function which accepts theta_dict and updates simulation parameters 
    prior_dict: dict 
        Dictionary storing information to map uniform sampled parameters to prior distribution.
        Form of {'param_name': {'bounds': (lower_bound, upper_bound), 'scale_func': callable}}.
    theta_samples: array-like
        Unscaled paramter values in range of (0,1) sampled from prior distribution
    tstop: int
        Simulation stop time (ms)
    save_path: str
        Location to store simulations. Must have subdirectories 'sbi_sims/' and 'temp/'
    save_suffix: str
        Name appended to end of output files
    theta_extra: dict
        Extra information needed for param_function passed through this variable
    """
    
    # create simulator object, rescale function transforms (0,1) to range specified in prior_dict    
    simulator = partial(simulator_hnn, prior_dict=prior_dict, param_function=param_function,
                        network_model=net, tstop=tstop, theta_extra=theta_extra, return_objects=True)
    # Generate simulations
    seq_list = list()
    num_sims = theta_samples.shape[0]
    step_size = num_cores
    
    for i in range(0, num_sims, step_size):
        seq = list(range(i, i + step_size))
        if i + step_size < theta_samples.shape[0]:
            batch(simulator, seq, theta_samples[i:i + step_size, :], save_path)
        else:
            seq = list(range(i, theta_samples.shape[0]))
            batch(simulator, seq, theta_samples[i:, :], save_path)
        seq_list.append(seq)
        
    # Load simulations into single array, save output, and remove small small files
    dpl_files = [f'{save_path}/temp/dpl_temp{seq[0]}-{seq[-1]}.npy' for seq in seq_list]
    spike_times_files = [f'{save_path}/temp/spike_times_temp{seq[0]}-{seq[-1]}.npy' for seq in seq_list]
    spike_gids_files = [f'{save_path}/temp/spike_gids_temp{seq[0]}-{seq[-1]}.npy' for seq in seq_list]
    theta_files = [f'{save_path}/temp/theta_temp{seq[0]}-{seq[-1]}.npy' for seq in seq_list]

    dpl_orig, spike_times_orig, spike_gids_orig, theta_orig = load_prerun_simulations(
        dpl_files, spike_times_files, spike_gids_files, theta_files)
    
    dpl_name = f'{save_path}/sbi_sims/dpl_{save_suffix}.npy'
    spike_times_name = f'{save_path}/sbi_sims/spike_times_{save_suffix}.npy'
    spike_gids_name = f'{save_path}/sbi_sims/spike_gids_{save_suffix}.npy'
    theta_name = f'{save_path}/sbi_sims/theta_{save_suffix}.npy'
    
    np.save(dpl_name, dpl_orig)
    np.save(spike_times_name, spike_times_orig)
    np.save(spike_gids_name, spike_gids_orig)
    np.save(theta_name, theta_orig)

    files = glob.glob(str(save_path) + '/temp/*')
    for f in files:
        os.remove(f) 

def start_cluster():
    """Reserve SLURM resources using Dask Distributed interface"""
     # Set up cluster and reserve resources
    cluster = SLURMCluster(
        cores=32, processes=32, queue='compute', memory="256GB", walltime="10:00:00",
        job_extra_directives=['-A csd403', '--nodes=1'], log_directory=os.getcwd() + '/slurm_out')

    client = Client(cluster)
    client.upload_file('../utils.py')
    print(client.dashboard_link)
    
    client.cluster.scale(num_cores)
        
def train_posterior(data_path, ntrain_sims, x_noise_amp, theta_noise_amp, extra_dict=None):
    """Train sbi posterior distribution"""
    posterior_dict = dict()
    posterior_dict_training_data = dict()


    prior_dict = dill.load(open(f'{data_path}/sbi_sims/prior_dict.pkl', 'rb'))
    sim_metadata = dill.load(open(f'{data_path}/sbi_sims/sim_metadata.pkl', 'rb'))

    prior = UniformPrior(parameters=list(prior_dict.keys()))
    n_params = len(prior_dict)
    limits = list(prior_dict.values())

    # x_orig stores full waveform to be used for embedding
    window_samples = extra_dict['window_samples']
    x_orig, theta_orig = np.load(f'{data_path}/sbi_sims/dpl_sbi.npy'), np.load(f'{data_path}/sbi_sims/theta_sbi.npy')
    x_orig, theta_orig = x_orig[:ntrain_sims, window_samples[0]:window_samples[1]], theta_orig[:ntrain_sims, :]

    #spike_gids_orig = np.load(f'{data_path}/sbi_sims/spike_gids_sbi.npy', allow_pickle=True)
    #spike_gids_orig = spike_gids_orig[:ntrain_sims]

    # Add noise for regularization
    x_noise = rng.normal(loc=0.0, scale=x_noise_amp, size=x_orig.shape)
    x_orig_noise = x_orig + x_noise
    
    theta_noise = rng.normal(loc=0.0, scale=theta_noise_amp, size=theta_orig.shape)
    theta_orig_noise = theta_orig + theta_noise

    dt = sim_metadata['dt'] # Sampling interval used for simulation
    fs = (1/dt) * 1e3
    
    slope_func = partial(get_scalefree_slope, fs=fs)

    posterior_metadata = {'rng_seed': rng_seed, 'x_noise_amp': x_noise_amp, 'theta_noise_amp': theta_noise_amp,
                          'ntrain_sims': ntrain_sims, 'fs': fs, 'window_samples': window_samples,
                          'extra_dict': extra_dict}
    posterior_metadata_save_label = f'{data_path}/posteriors/posterior_metadata.pkl'
    with open(posterior_metadata_save_label, 'wb') as output_file:
            dill.dump(posterior_metadata, output_file)
            
    raw_data_type = {'dpl': x_orig_noise,
                     'aperiodic_fname': extra_dict['aperiodic_fname']
                     #'spike_gids': spike_gids_orig,
                     #'dpl_spike_gids': {'dpl': x_orig_noise, 'spike_gids': spike_gids_orig}
                    }
    input_type_list = {'aperiodic': {
                           'embedding_func': torch.nn.Identity,
                           'embedding_dict': dict(), 'feature_func': np.load,
                           'data_type': 'aperiodic_fname'},
                        'bandpower': {
                           'embedding_func': torch.nn.Identity,
                           'embedding_dict': dict(), 'feature_func': partial(get_dataset_bandpower, fs=fs),
                           'data_type': 'dpl'}
                      }
    

    # Train a posterior for each input type and save state_dict
    for input_type, input_dict in input_type_list.items():
        print(input_type)

        neural_posterior = sbi_utils.posterior_nn(model='maf', embedding_net=input_dict['embedding_func'](**input_dict['embedding_dict']))
        inference = sbi_inference.SNPE(prior=prior, density_estimator=neural_posterior, show_progress_bars=True, device=device)
        x_train = torch.tensor(input_dict['feature_func'](raw_data_type[input_dict['data_type']])).float()
        theta_train = torch.tensor(theta_orig_noise).float()
        if x_train.dim() == 1:
            x_train= x_train.reshape(-1, 1)

        inference.append_simulations(theta_train, x_train, proposal=prior)

        print(theta_train.shape, x_train.shape)
        nn_posterior = inference.train(num_atoms=10, training_batch_size=5000, use_combined_loss=True, discard_prior_samples=True, show_train_summary=True)

        posterior_dict[input_type] = {'posterior': nn_posterior.state_dict(),
                                      'n_params': n_params,
                                      'n_sims': ntrain_sims,
                                      'input_dict': input_dict}

        # Save intermediate progress
        posterior_save_label = f'{data_path}/posteriors/posterior_dicts.pkl'
        with open(posterior_save_label, 'wb') as output_file:
            dill.dump(posterior_dict, output_file)
            
            
def validate_posterior(net, nval_sims, param_function, data_path):
        
    # Open relevant files
    with open(f'{data_path}/posteriors/posterior_dicts.pkl', 'rb') as output_file:
        posterior_state_dicts = dill.load(output_file)
    with open(f'{data_path}/sbi_sims/prior_dict.pkl', 'rb') as output_file:
        prior_dict = dill.load(output_file)
    with open(f'{data_path}/sbi_sims/sim_metadata.pkl', 'rb') as output_file:
        sim_metadata = dill.load(output_file)
    with open(f'{data_path}/posteriors/posterior_metadata.pkl', 'rb') as output_file:
        posterior_metadata = dill.load(output_file)

    dt = sim_metadata['dt'] # Sampling interval used for simulation
    tstop = sim_metadata['tstop'] # Sampling interval used for simulation
    window_samples = posterior_metadata['window_samples']


    prior = UniformPrior(parameters=list(prior_dict.keys()))

    # x_orig stores full waveform to be used for embedding
    x_orig, theta_orig = np.load(f'{data_path}/sbi_sims/x_sbi.npy'), np.load(f'{data_path}/sbi_sims/theta_sbi.npy')
    x_cond, theta_cond = np.load(f'{data_path}/sbi_sims/x_grid.npy'), np.load(f'{data_path}/sbi_sims/theta_grid.npy')

    x_orig = x_orig[:, window_samples[0]:window_samples[1]]
    x_cond = x_cond[:, window_samples[0]:window_samples[1]]

    load_info = {name: {'x_train': posterior_dict['input_dict']['feature_func'](x_orig), 
                        'x_cond': posterior_dict['input_dict']['feature_func'](x_cond)}
                 for name, posterior_dict in posterior_state_dicts.items()}


    for input_type, posterior_dict in posterior_state_dicts.items():
        state_dict = posterior_dict['posterior']
        input_dict = posterior_dict['input_dict']
        embedding_net =  input_dict['embedding_func'](**input_dict['embedding_dict'])
        
        posterior = load_posterior(state_dict=state_dict,
                                   x_infer=torch.tensor(load_info[input_type]['x_train'][:10,:]).float(),
                                   theta_infer=torch.tensor(theta_orig[:10,:]), prior=prior, embedding_net=embedding_net)


        samples_list = list()
        for cond_idx in range(x_cond.shape[0]):
            if cond_idx % 100 == 0:    
                print(cond_idx, end=' ')
            samples = posterior.sample((nval_sims,), x=load_info[input_type]['x_cond'][cond_idx,:])
            samples_list.append(samples)

        theta_samples = torch.tensor(np.vstack(samples_list))

        save_suffix = f'{input_type}_validation'
        run_hnn_sim(net=net, param_function=param_function, prior_dict=prior_dict,
                theta_samples=theta_samples, tstop=tstop, save_path=data_path, save_suffix=save_suffix)

# Create batch simulation function
def batch(simulator, seq, theta_samples, save_path, save_spikes=True):
    print(f'Sim Idx: {(seq[0], seq[-1])}')
    res_list = list()
    # Create lazy list of tasks    
    for sim_idx in range(len(seq)):
        res = dask.delayed(simulator)(theta_samples[sim_idx,:])
        res_list.append(res)

    # Run tasks
    final_res = dask.compute(*res_list)
    
    # Unpack dipole and spiking data
    dpl_list = list()
    spike_times_list = list()
    spike_gids_list = list()
    for res in final_res:
        net_res = res[0][0]
        dpl_res = res[0][1]
        
        # dpl_list.append(dpl_res[0].copy().smooth(20).data['agg'])
        dpl_list.append(dpl_res[0].copy().data['agg'])
        spike_times_list.append(net_res.cell_response.spike_times[0])
        spike_gids_list.append(net_res.cell_response.spike_gids[0])

        
    spike_times_list = np.array(spike_times_list, dtype=object)
    spike_gids_list = np.array(spike_gids_list, dtype=object)
    
    dpl_name = f'{save_path}/temp/dpl_temp{seq[0]}-{seq[-1]}.npy'
    spike_times_name = f'{save_path}/temp/spike_times_temp{seq[0]}-{seq[-1]}.npy'
    spike_gids_name = f'{save_path}/temp/spike_gids_temp{seq[0]}-{seq[-1]}.npy'
    

    theta_name = f'{save_path}/temp/theta_temp{seq[0]}-{seq[-1]}.npy'

    np.save(dpl_name, dpl_list)
    np.save(theta_name, theta_samples.detach().cpu().numpy())
    
    if save_spikes:
        np.save(spike_times_name, spike_times_list)
        np.save(spike_gids_name, spike_gids_list)
    else:
        np.save(spike_times_name, list())
        np.save(spike_gids_name, list())

def linear_scale_forward(value, bounds, constrain_value=True):
    """Scale value in range (0,1) to range bounds"""
    if constrain_value:
        assert np.all(value >= 0.0) and np.all(value <= 1.0)
        
    assert isinstance(bounds, tuple)
    assert bounds[0] < bounds[1]
    
    return (bounds[0] + (value * (bounds[1] - bounds[0]))).astype(float)

def linear_scale_array(value, bounds, constrain_value=True):
    """Scale columns of array according to bounds"""
    assert value.shape[1] == len(bounds)
    return np.vstack(
        [linear_scale_forward(value[:, idx], bounds[idx], constrain_value) for 
         idx in range(len(bounds))]).T

def log_scale_forward(value, bounds, constrain_value=True):
    """log scale value in range (0,1) to range bounds in base 10"""
    rescaled_value = linear_scale_forward(value, bounds, constrain_value)
    
    return 10**rescaled_value

def log_scale_array(value, bounds, constrain_value=True):
    """log scale columns of array according to bounds in base 10"""
    assert value.shape[1] == len(bounds)
    return np.vstack(
        [log_scale_forward(value[:, idx], bounds[idx], constrain_value) for 
         idx in range(len(bounds))]).T

# 1/f slope citation https://www.sciencedirect.com/science/article/pii/S1053811917305621?via=ihub
def get_scalefree_slope(x, fs, min_freq=30, max_freq=100):    
    slope_list = list()
    for idx in range(x.shape[0]):
        freqs, Pxx = welch(x[idx,:], fs, nperseg=1024, average='median')

        lm = LinearRegression()
        mask = np.logical_and(freqs > min_freq, freqs < max_freq)

        lm.fit(np.log(freqs[mask].reshape(-1,1)), np.log(Pxx[mask]).reshape(-1,1))
        slope_list.append(lm.coef_[0][0])
        
    return np.array(slope_list).reshape(-1,1)

def get_aperiodic(dpl, fs, min_freq=3, max_freq=80):
    fm = FOOOF()
    freqs, Pxx = welch(dpl, fs, nperseg=10_000, average='median', noverlap=5000)
    
    freq_range = [min_freq, max_freq]
    # Define frequency range across which to model the spectrum
    fm.report(freqs, Pxx, freq_range)

    aperiodic_params = fm.get_results().aperiodic_params
    offset, exponent = aperiodic_params[0], aperiodic_params[1]
    
    return offset, exponent

def bandpower(x, fs, fmin, fmax):
    f, Pxx = scipy.signal.periodogram(x, fs=fs)
    ind_min = np.argmax(f > fmin) - 1
    ind_max = np.argmax(f > fmax) - 1
    return np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])

# Bands freq citation: https://www.frontiersin.org/articles/10.3389/fnhum.2020.00089/full
def get_dataset_bandpower(x, fs):
    freq_band_list = [(0,13), (13,30), (30,50), (50,80)]
    
    x_bandpower_list = list()
    for idx in range(x.shape[0]):
        x_bandpower = np.array([bandpower(x[idx,:], fs, freq_band[0], freq_band[1]) for freq_band in freq_band_list])
        x_bandpower_list.append(x_bandpower)
        
    return np.vstack(np.log(x_bandpower_list))

def get_dataset_psd(x_raw, fs, return_freq=True, max_freq=200):
    """Calculate PSD on observed time series (rows of array)"""
    x_psd = list()
    for idx in range(x_raw.shape[0]):
        f, Pxx_den = signal.periodogram(x_raw[idx, :], fs)
        x_psd.append(Pxx_den[(f<max_freq)&(f>0)])
    if return_freq:
        return np.vstack(np.log(x_psd)), f[(f<max_freq)&(f>0)]
    else:
        return np.vstack(np.log(x_psd))
    
# Source: https://stackoverflow.com/questions/44547669/python-numpy-equivalent-of-bandpower-from-matlab
def bandpower(x, fs, fmin, fmax):
    f, Pxx = scipy.signal.periodogram(x, fs=fs)
    ind_min = scipy.argmax(f > fmin) - 1
    ind_max = scipy.argmax(f > fmax) - 1
    return scipy.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])


def load_posterior(state_dict, x_infer, theta_infer, prior, embedding_net):
    """Load a pretrained SBI posterior distribution
    Parameters
    ----------
    """
    neural_posterior = sbi_utils.posterior_nn(model='maf', embedding_net=embedding_net)
    inference = sbi_inference.SNPE(prior=prior, density_estimator=neural_posterior, show_progress_bars=True, device=device)
    inference.append_simulations(theta_infer, x_infer, proposal=prior)

    nn_posterior = inference.train(num_atoms=10, training_batch_size=5000, use_combined_loss=True, discard_prior_samples=True, max_num_epochs=2, show_train_summary=False)
    nn_posterior.zero_grad()
    nn_posterior.load_state_dict(state_dict)

    posterior = inference.build_posterior(nn_posterior)
    return posterior

class UniformPrior(sbi_utils.BoxUniform):
    """Prior distribution object that generates uniform sample on range (0,1)"""
    def __init__(self, parameters):
        """
        Parameters
        ----------
        parameters: list of str
            List of parameter names for prior distribution
        """
        self.parameters = parameters
        low = len(parameters)*[0]
        high = len(parameters)*[1]
        super().__init__(low=torch.tensor(low, dtype=torch.float32),
                         high=torch.tensor(high, dtype=torch.float32))
        
        
# __Simulation__
class HNNSimulator:
    """Simulator class to run HNN simulations"""
    
    def __init__(self, prior_dict, param_function, network_model, tstop,
                 return_objects):
        """
        Parameters
        ----------
        prior_dict: dict 
            Dictionary storing parameters to be updated as {name: (lower_bound, upper_bound)}
            where pameter values passed in the __call__() are scaled between the lower and upper
            bounds
        param_function: function definition
            Function which accepts theta_dict and updates simulation parameters
        network_model: function definiton
            Function defined in network_models.py of hnn_core which builds the desired Network to
            be simulated.
        tstop: int
            Simulation stop time (ms)
        return_objects: bool
            If true, returns tuple of (Network, Dipole) objects. If False, a preprocessed time series
            of the aggregate current dipole (Dipole.data['agg']) is returned.
        """
        self.dt = 0.05  # Used for faster simulations, default.json uses 0.025 ms
        self.tstop = tstop  # ms
        self.prior_dict = prior_dict
        self.param_function = param_function
        self.return_objects = return_objects
        self.network_model = network_model

    def __call__(self, theta_dict, theta_extra=dict()):
        """
        Parameters
        ----------
        theta_dict: dict
            Dictionary indexing parameter values to be updated. Keys must match those defined
            in prior_dict
            
        Returns: array-like
            Simulated Output
        """        
        assert len(theta_dict) == len(self.prior_dict)
        assert theta_dict.keys() == self.prior_dict.keys()
        theta_dict['theta_extra'] = theta_extra

        # instantiate the network object -- only connectivity params matter
        net = self.network_model.copy()
        
        # Update parameter values from prior dict
        self.param_function(net, theta_dict)

        # simulate dipole over one trial
        dpl = simulate_dipole(net, tstop=self.tstop, dt=self.dt, n_trials=1, postproc=False)

        # get the signal output
        x = torch.tensor(dpl[0].copy().smooth(20).data['agg'], dtype=torch.float32)
        
        if self.return_objects:
            return net, dpl
        else:
            del net, dpl
            return x      

def simulator_hnn(theta, prior_dict, param_function, network_model,
                  tstop, theta_extra=dict(), return_objects=False):
    """Helper function to run simulations with HNN class

    Parameters
    ----------
    theta: array-like
        Unscaled paramter values in range of (0,1) sampled from prior distribution
    prior_dict: dict 
        Dictionary storing parameters to be updated as {name: (lower_bound, upper_bound)}
        where pameter values passed in the __call__() are scaled between the lower and upper
        bounds
    param_function: function definition
        Function which accepts theta_dict and updates simulation parameters
    network_model: function definiton
        Function defined in network_models.py of hnn_core which builds the desired Network to
        be simulated.
    tstop: int
        Simulation stop time (ms)
    theta_extra: dict
        Extra information needed for param_function passed through this variable
    return_objects: bool
        If true, returns tuple of (Network, Dipole) objects. If False, a preprocessed time series
        of the aggregate current dipole (Dipole.data['agg']) is returned.
        
    Returns
    -------
    x: array-like
        Simulated output
    """

    # create simulator
    hnn = HNNSimulator(prior_dict, param_function, network_model, tstop, return_objects)

    # handle when just one theta
    if theta.ndim == 1:
        return simulator_hnn(theta.view(1, -1), prior_dict, param_function,
                             return_objects=return_objects, network_model=network_model, tstop=tstop,
                             theta_extra=theta_extra)

    # loop through different values of theta
    x = list()
    for sample_idx, thetai in enumerate(theta):
        theta_dict = {param_name: param_dict['rescale_function'](thetai[param_idx].numpy(), param_dict['bounds']) for 
                      param_idx, (param_name, param_dict) in enumerate(prior_dict.items())}
        theta_extra['sample_idx'] =  sample_idx
        
        print(theta_dict)
        xi = hnn(theta_dict, theta_extra)
        x.append(xi)

    # Option to return net and dipole objects or just the 
    if return_objects:
        return x
    else:
        x = torch.stack(x)
        return torch.tensor(x, dtype=torch.float32)
    
def beta_tuning_param_function(net, theta_dict):    
    seed_rng = np.random.default_rng(theta_dict['theta_extra']['sample_idx'])
    seed_array = seed_rng.integers(10e5, size=100)

    seed_count = 0
    
    valid_conn_list = theta_dict['theta_extra']['valid_conn_list']
    for conn_name in valid_conn_list:
    
        conn_prob_name = f'{conn_name}_prob'
        conn_gbar_name = f'{conn_name}_gbar'
        
        conn_indices = theta_dict['theta_extra'][f'{conn_name}_conn_indices']
        assert len(conn_indices) == 1
        conn_idx = conn_indices[0]
        
        probability = theta_dict[conn_prob_name]
        gbar = theta_dict[conn_gbar_name]
        
        # Prune connections using internal connection_probability function
        _connection_probability(
            net.connectivity[conn_idx], probability=probability, conn_seed=seed_array[seed_count])
        net.connectivity[conn_idx]['probability'] = probability
        net.connectivity[conn_idx]['nc_dict']['A_weight'] = gbar
        seed_count = seed_count + 1

    for conn_idx in range(len(net.connectivity)):
        net.connectivity[conn_idx]['nc_dict']['lamtha'] = theta_dict['theta_extra']['lamtha']          
        
    rate = 10
    # Add Poisson drives
    weights_ampa_d1 = {'L2_pyramidal': theta_dict['L2e_distal_gbar'], 'L5_pyramidal': theta_dict['L5e_distal_gbar'],
                       'L2_basket': theta_dict['L2i_distal_gbar']}
    rates_d1 = {'L2_pyramidal': rate, 'L5_pyramidal': rate, 'L2_basket': rate}

    net.add_poisson_drive(
        name='distal', tstart=0, tstop=None, rate_constant=rates_d1, location='distal', n_drive_cells='n_cells',
        cell_specific=True, weights_ampa=weights_ampa_d1, weights_nmda=None, space_constant=1e50,
        synaptic_delays=0.0, probability=1.0, event_seed=seed_array[-1], conn_seed=seed_array[-2])

    weights_ampa_p1 = {'L2_pyramidal': theta_dict['L2e_proximal_gbar'], 'L5_pyramidal': theta_dict['L5e_proximal_gbar'],
                       'L2_basket': theta_dict['L2i_proximal_gbar'], 'L5_basket': theta_dict['L5i_proximal_gbar']}
    rates_p1 = {'L2_pyramidal': rate, 'L5_pyramidal': rate, 'L2_basket': rate, 'L5_basket': rate}

    net.add_poisson_drive(
        name='proximal', tstart=0, tstop=None, rate_constant=rates_p1, location='proximal', n_drive_cells='n_cells',
        cell_specific=True, weights_ampa=weights_ampa_p1, weights_nmda=None, space_constant=1e50,
        synaptic_delays=0.0, probability=1.0, event_seed=seed_array[-3], conn_seed=seed_array[-4])
    
def load_prerun_simulations(
    dpl_files, spike_times_files, spike_gids_files,
    theta_files, downsample=1, save_name=None, save_data=False):
    "Aggregate simulation batches into single array"
    
    print(dpl_files)
    print(spike_times_files)
    print(spike_gids_files)
    print(theta_files)
        
    dpl_all, spike_times_all, spike_gids_all, theta_all = list(), list(), list(), list()
    
    for file_idx in range(len(dpl_files)):
        dpl_all.append(np.load(dpl_files[file_idx])[:,::downsample])
        theta_all.append(np.load(theta_files[file_idx]))
        
        spike_times_list = np.load(spike_times_files[file_idx], allow_pickle=True)
        spike_gids_list = np.load(spike_gids_files[file_idx], allow_pickle=True)
        
        for sim_idx in range(len(spike_times_list)):
            spike_times_all.append(spike_times_list[sim_idx])
            spike_gids_all.append(spike_gids_list[sim_idx])
    
    dpl_all = np.vstack(dpl_all)
    theta_all = np.vstack(theta_all)
    spike_times_all = np.array(spike_times_all, dtype=object)
    spike_gids_all = np.array(spike_gids_all, dtype=object)
    
    if save_data and isinstance(save_name, str):
        np.save(save_name + '_dpl_all.npy', dpl_all)
        np.save(save_name + '_spike_times_all.npy', spike_times_all)
        np.save(save_name + '_spike_gids_all.npy', spike_gids_all)

        np.save(save_name + '_theta_all.npy', theta_all)
    else:
        return dpl_all, spike_times_all, spike_gids_all, theta_all
    
def get_parameter_recovery(theta_val, theta_cond, n_samples=10):
    """Calculate the PPC using root mean squared error
    Parameters
    ----------
    x_val: array-like
    
    x_cond: array-like
    
    n_samples: int
    
    Returns
    -------
    dist_array: array-like
    """
    
    dist_list = list()
    for cond_idx in range(theta_cond.shape[0]):
        start_idx, stop_idx = cond_idx*n_samples, (cond_idx+1)*n_samples
        dist = [wasserstein_distance(theta_val[start_idx:stop_idx, param_idx], [theta_cond[cond_idx,param_idx]]) for
                param_idx in range(theta_cond.shape[1])]
        dist_list.append(dist)
    dist_array = np.array(dist_list)
    return dist_array

def get_posterior_predictive_check(x_val, x_cond, n_samples=10):
    """Calculate the PPC using root mean squared error
    Parameters
    ----------
    x_val: array-like
    
    x_cond: array-like
    
    n_samples: int
    
    Returns
    -------
    dist_array: array-like
        
    """
    dist_list = list()
    for cond_idx in range(x_cond.shape[0]):
        start_idx, stop_idx = cond_idx*n_samples, (cond_idx+1)*n_samples
        dist = np.sqrt(np.mean(np.square(x_val[start_idx:stop_idx,:] - np.tile(x_cond[cond_idx,:], n_samples).reshape(n_samples,-1))))
        dist_list.append(dist)
    dist_array = np.array(dist_list)
    return dist_array

class PriorBetaFiltered():
    """Class for creating a prior distribution from
       heuristically filtered simulations"""
    
    def __init__(self, parameters):
        self.parameters = parameters
        nparams = len(parameters)
        self.flow = self.__flow_init(nparams)
        self.acc_rate = 0.0

    def __flow_init(self, nparams):
        num_layers = 5
        base_dist = StandardNormal(shape=[nparams])
        transforms = []
        for _ in range(num_layers):
            transforms.append(ReversePermutation(features=nparams))
            transforms.append(MaskedAffineAutoregressiveTransform(features=nparams,
                                hidden_features=50,
                                context_features=None,
                                num_blocks=2,
                                use_residual_blocks=False,
                                random_mask=False,
                                activation=torch.tanh,
                                dropout_probability=0.0,
                                use_batch_norm=True))
        transform = CompositeTransform(transforms)
        return Flow(transform, base_dist)

    def sample(self, sample_shape, return_acc_rate=False):
        base_prior = UniformPrior(self.parameters)        
        nsamples = sample_shape[0]
        nsamples_keep = 0
        samples_keep = torch.tensor([])
        total_samples = 0
        while nsamples_keep < nsamples:
            samples = self.flow.sample(nsamples).detach()
            total_samples = total_samples + len(samples)
            indices = list()
            for idx in range(len(samples)):
                try:
                    if base_prior.log_prob(samples[idx]) == 0:
                        indices.append(idx)
                except ValueError:
                    pass
                
            samples_keep = torch.cat([samples_keep, samples[indices]])
            nsamples_keep = len(samples_keep)
        acc_rate = torch.tensor(nsamples_keep / total_samples)
        samples = samples_keep[:nsamples]   
        if return_acc_rate:
            return samples, acc_rate
        else:
            return samples     

    def log_prob(self, theta):
        if self.acc_rate == 0.0:
            # get the acceptance rate right away
            _, acc_rate = self.sample((10_000,), return_acc_rate=True)            
        return self.flow.log_prob(theta).detach() - torch.log(acc_rate)
    
class Flow_base(Flow):
    def __init__(self, batch_theta, batch_x, embedding_net, n_layers=5,
                 z_score_theta=True, z_score_x=True, device='cpu'):

        # instantiate the flow
        flow = build_nsf(batch_x=batch_theta,
                         batch_y=batch_x,
                         z_score_x=z_score_theta,
                         z_score_y=z_score_x,
                         embedding_net=embedding_net).to(device)

        super().__init__(flow._transform, 
                         flow._distribution, 
                         flow._embedding_net)

    def save_state(self, filename):
        state_dict = {}
        state_dict['flow'] = self.state_dict()
        torch.save(state_dict, filename)

    def load_state(self, filename):
        state_dict = torch.load(filename, map_location=device)
        self.load_state_dict(state_dict['flow'])

def build_flow(batch_theta,
               batch_x,
               embedding_net,
               **kwargs):

    flow = Flow_base(batch_theta, 
                     batch_x, 
                     embedding_net,
                     device,
                     **kwargs).to(device)

    return flow
