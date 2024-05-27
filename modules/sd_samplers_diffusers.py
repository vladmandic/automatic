import os
import re
import inspect
from modules import shared
from modules import sd_samplers_common
from modules.tcd import TCDScheduler


debug = shared.log.trace if os.environ.get('SD_SAMPLER_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: SAMPLER')


try:
    from diffusers import (
        CMStochasticIterativeScheduler,
        DDIMScheduler,
        DDPMScheduler,
        UniPCMultistepScheduler,
        DEISMultistepScheduler,
        DPMSolverMultistepScheduler,
        DPMSolverSinglestepScheduler,
        DPMSolverSDEScheduler,
        EDMDPMSolverMultistepScheduler,
        EDMEulerScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        HeunDiscreteScheduler,
        IPNDMScheduler,
        KDPM2DiscreteScheduler,
        KDPM2AncestralDiscreteScheduler,
        LCMScheduler,
        LMSDiscreteScheduler,
        PNDMScheduler,
        SASolverScheduler,
    )
except Exception as e:
    import diffusers
    shared.log.error(f'Diffusers import error: version={diffusers.__version__} error: {e}')


config = {
    # beta_start, beta_end are typically per-scheduler, but we don't want them as they should be taken from the model itself as those are values model was trained on
    # prediction_type is ideally set in model as well, but it maybe needed that we do auto-detect of model type in the future
    'All': { 'num_train_timesteps': 1000, 'beta_start': 0.0001, 'beta_end': 0.02, 'beta_schedule': 'linear', 'prediction_type': 'epsilon' },
    'DDIM': { 'clip_sample': False, 'set_alpha_to_one': True, 'steps_offset': 0, 'clip_sample_range': 1.0, 'sample_max_value': 1.0, 'timestep_spacing': 'linspace', 'rescale_betas_zero_snr': False },
    'UniPC': { 'solver_order': 2, 'thresholding': False, 'sample_max_value': 1.0, 'predict_x0': 'bh2', 'lower_order_final': True, 'timestep_spacing': 'linspace' },
    'DEIS': { 'solver_order': 2, 'thresholding': False, 'sample_max_value': 1.0, 'algorithm_type': "deis", 'solver_type': "logrho", 'lower_order_final': True, 'timestep_spacing': 'linspace' },
    'DPM++': { 'solver_order': 2, 'thresholding': False, 'sample_max_value': 1.0, 'algorithm_type': "dpmsolver++", 'solver_type': "midpoint", 'lower_order_final': True, 'use_karras_sigmas': False, 'final_sigmas_type': 'sigma_min' },
    'DPM++ 2M': { 'thresholding': False, 'sample_max_value': 1.0, 'algorithm_type': "dpmsolver++", 'solver_type': "midpoint", 'lower_order_final': True, 'use_karras_sigmas': False, 'final_sigmas_type': 'zero', 'timestep_spacing': 'linspace' },
    'DPM SDE': { 'use_karras_sigmas': False, 'noise_sampler_seed': None, 'timestep_spacing': 'linspace', 'steps_offset': 0 },
    'Euler a': { 'rescale_betas_zero_snr': False, 'timestep_spacing': 'linspace' },
    'Euler': { 'interpolation_type': "linear", 'use_karras_sigmas': False, 'rescale_betas_zero_snr': False, 'timestep_spacing': 'linspace' },
    'Heun': { 'use_karras_sigmas': False, 'timestep_spacing': 'linspace' },
    'DDPM': { 'variance_type': "fixed_small", 'clip_sample': False, 'thresholding': False, 'clip_sample_range': 1.0, 'sample_max_value': 1.0, 'timestep_spacing': 'linspace', 'rescale_betas_zero_snr': False },
    'KDPM2': { 'steps_offset': 0, 'timestep_spacing': 'linspace' },
    'KDPM2 a': { 'steps_offset': 0, 'timestep_spacing': 'linspace' },
    'LMSD': { 'use_karras_sigmas': False, 'timestep_spacing': 'linspace', 'steps_offset': 0 },
    'PNDM': { 'skip_prk_steps': False, 'set_alpha_to_one': False, 'steps_offset': 0, 'timestep_spacing': 'linspace' },
    'SA Solver': {'predictor_order': 2, 'corrector_order': 2, 'thresholding': False, 'lower_order_final': True, 'use_karras_sigmas': False, 'timestep_spacing': 'linspace'},
    'LCM': { 'beta_start': 0.00085, 'beta_end': 0.012, 'beta_schedule': "scaled_linear", 'set_alpha_to_one': True, 'rescale_betas_zero_snr': False, 'thresholding': False, 'timestep_spacing': 'linspace' },
    'TCD': { 'set_alpha_to_one': True, 'rescale_betas_zero_snr': False, 'beta_schedule': 'scaled_linear' },
    'Euler SGM': { 'timestep_spacing': "trailing", 'prediction_type': "sample" },
    'Euler EDM': { },
    'DPM++ 2M EDM': { 'solver_order': 2, 'solver_type': 'midpoint', 'final_sigmas_type': 'zero', 'algorithm_type': 'dpmsolver++' },
    'CMSI': { }, #{ 'sigma_min':  0.002, 'sigma_max': 80.0, 'sigma_data': 0.5, 's_noise': 1.0, 'rho': 7.0, 'clip_denoised': True },
    'IPNDM': { },
}

samplers_data_diffusers = [
    sd_samplers_common.SamplerData('Default', None, [], {}),

    sd_samplers_common.SamplerData('UniPC', lambda model: DiffusionSampler('UniPC', UniPCMultistepScheduler, model), [], {}),
    sd_samplers_common.SamplerData('DEIS', lambda model: DiffusionSampler('DEIS', DEISMultistepScheduler, model), [], {}),
    sd_samplers_common.SamplerData('SA Solver', lambda model: DiffusionSampler('SA Solver', SASolverScheduler, model), [], {}),
    sd_samplers_common.SamplerData('DDIM', lambda model: DiffusionSampler('DDIM', DDIMScheduler, model), [], {}),
    sd_samplers_common.SamplerData('Heun', lambda model: DiffusionSampler('Heun', HeunDiscreteScheduler, model), [], {}),
    sd_samplers_common.SamplerData('Euler', lambda model: DiffusionSampler('Euler', EulerDiscreteScheduler, model), [], {}),
    sd_samplers_common.SamplerData('Euler a', lambda model: DiffusionSampler('Euler a', EulerAncestralDiscreteScheduler, model), [], {}),
    sd_samplers_common.SamplerData('DPM++', lambda model: DiffusionSampler('DPM++', DPMSolverSinglestepScheduler, model), [], {}),
    sd_samplers_common.SamplerData('DPM++ 2M', lambda model: DiffusionSampler('DPM++ 2M', DPMSolverMultistepScheduler, model), [], {}),
    sd_samplers_common.SamplerData('DPM SDE', lambda model: DiffusionSampler('DPM SDE', DPMSolverSDEScheduler, model), [], {}),

    sd_samplers_common.SamplerData('PNDM', lambda model: DiffusionSampler('PNDM', PNDMScheduler, model), [], {}),
    sd_samplers_common.SamplerData('IPNDM', lambda model: DiffusionSampler('IPNDM', IPNDMScheduler, model), [], {}),
    sd_samplers_common.SamplerData('DDPM', lambda model: DiffusionSampler('DDPM', DDPMScheduler, model), [], {}),
    sd_samplers_common.SamplerData('LMSD', lambda model: DiffusionSampler('LMSD', LMSDiscreteScheduler, model), [], {}),
    sd_samplers_common.SamplerData('KDPM2', lambda model: DiffusionSampler('KDPM2', KDPM2DiscreteScheduler, model), [], {}),
    sd_samplers_common.SamplerData('KDPM2 a', lambda model: DiffusionSampler('KDPM2 a', KDPM2AncestralDiscreteScheduler, model), [], {}),
    sd_samplers_common.SamplerData('DPM++ 2M EDM', lambda model: DiffusionSampler('DPM++ 2M EDM', EDMDPMSolverMultistepScheduler, model), [], {}),
    sd_samplers_common.SamplerData('Euler EDM', lambda model: DiffusionSampler('Euler EDM', EDMEulerScheduler, model), [], {}),
    sd_samplers_common.SamplerData('Euler SGM', lambda model: DiffusionSampler('Euler SGM', EulerDiscreteScheduler, model), [], {}),
    sd_samplers_common.SamplerData('LCM', lambda model: DiffusionSampler('LCM', LCMScheduler, model), [], {}),
    sd_samplers_common.SamplerData('TCD', lambda model: DiffusionSampler('TCD', TCDScheduler, model), [], {}),
    sd_samplers_common.SamplerData('CMSI', lambda model: DiffusionSampler('CMSI', CMStochasticIterativeScheduler, model), [], {}),

    sd_samplers_common.SamplerData('Same as primary', None, [], {}),
]


class DiffusionSampler:
    def __init__(self, name, constructor, model, **kwargs):
        if name == 'Default':
            return
        self.name = name
        self.config = {}
        if not hasattr(model, 'scheduler'):
            return
        for key, value in config.get('All', {}).items(): # apply global defaults
            self.config[key] = value
        # shared.log.debug(f'Sampler: name={name} type=all config={self.config}')
        for key, value in config.get(name, {}).items(): # apply diffusers per-scheduler defaults
            self.config[key] = value
        # shared.log.debug(f'Sampler: name={name} type=scheduler config={self.config}')
        if hasattr(model.scheduler, 'scheduler_config'): # find model defaults
            orig_config = model.scheduler.scheduler_config
        else:
            orig_config = model.scheduler.config
        for key, value in orig_config.items(): # apply model defaults
            if key in self.config:
                self.config[key] = value
        # shared.log.debug(f'Sampler: name={name} type=model config={self.config}')
        for key, value in kwargs.items(): # apply user args, if any
            if key in self.config:
                self.config[key] = value
        # shared.log.debug(f'Sampler: name={name} type=user config={self.config}')
        # finally apply user preferences
        if shared.opts.schedulers_prediction_type != 'default':
            self.config['prediction_type'] = shared.opts.schedulers_prediction_type
        if shared.opts.schedulers_beta_schedule != 'default':
            self.config['beta_schedule'] = shared.opts.schedulers_beta_schedule
        if 'use_karras_sigmas' in self.config:
            timesteps = re.split(',| ', shared.opts.schedulers_timesteps)
            timesteps = [int(x) for x in timesteps if x.isdigit()]
            self.config['use_karras_sigmas'] = shared.opts.schedulers_use_karras if len(timesteps) == 0 else False
        if 'thresholding' in self.config:
            self.config['thresholding'] = shared.opts.schedulers_use_thresholding
        if 'lower_order_final' in self.config:
            self.config['lower_order_final'] = shared.opts.schedulers_use_loworder
        if 'solver_order' in self.config:
            self.config['solver_order'] = shared.opts.schedulers_solver_order
        if 'predict_x0' in self.config:
            self.config['predict_x0'] = shared.opts.uni_pc_variant
        if 'beta_start' in self.config and shared.opts.schedulers_beta_start > 0:
            self.config['beta_start'] = shared.opts.schedulers_beta_start
        if 'beta_end' in self.config and shared.opts.schedulers_beta_end > 0:
            self.config['beta_end'] = shared.opts.schedulers_beta_end
        if 'rescale_betas_zero_snr' in self.config:
            self.config['rescale_betas_zero_snr'] = shared.opts.schedulers_rescale_betas
        if 'timestep_spacing' in self.config and shared.opts.schedulers_timestep_spacing != 'default' and shared.opts.schedulers_timestep_spacing is not None:
            self.config['timestep_spacing'] = shared.opts.schedulers_timestep_spacing
        if 'num_train_timesteps' in self.config:
            self.config['num_train_timesteps'] = shared.opts.schedulers_timesteps_range
        if name in {'DPM++ 2M', 'DPM++ 2M EDM'}:
            self.config['algorithm_type'] = shared.opts.schedulers_dpm_solver
        if name == 'DEIS':
            self.config['algorithm_type'] = 'deis'
        if 'EDM' in name:
            del self.config['beta_start']
            del self.config['beta_end']
            del self.config['beta_schedule']
        if name in {'IPNDM', 'CMSI'}:
            del self.config['beta_start']
            del self.config['beta_end']
            del self.config['beta_schedule']
            del self.config['prediction_type']
        if 'SGM' in name:
            self.config['timestep_spacing'] = 'trailing'
        # validate all config params
        signature = inspect.signature(constructor, follow_wrapped=True)
        possible = signature.parameters.keys()
        debug(f'Sampler: sampler="{name}" config={self.config} signature={possible}')
        for key in self.config.copy().keys():
            if key not in possible:
                shared.log.warning(f'Sampler: sampler="{name}" config={self.config} invalid={key}')
                del self.config[key]
        # shared.log.debug(f'Sampler: sampler="{name}" config={self.config}')
        self.sampler = constructor(**self.config)
        # shared.log.debug(f'Sampler: class="{self.sampler.__class__.__name__}" config={self.sampler.config}')
        self.sampler.name = name
