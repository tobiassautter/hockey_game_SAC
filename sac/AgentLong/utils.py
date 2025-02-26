import sys
sys.path.insert(0, '.')
import omegaconf
import torch

def get_config(source_config=None, cli_override=True):
    config = omegaconf.OmegaConf.load("config.yaml")
    if cli_override:
        config = omegaconf.OmegaConf.merge(config, omegaconf.OmegaConf.from_dotlist(sys.argv[1:]))  # Override config by command line
    if config.env.obs_augmentation:
        config.env.obs_dim = config.env.obs_augmentation_dim
    if source_config is not None:
        config =  omegaconf.OmegaConf.merge(config, source_config)
    return config

def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)

def get_scheduler(config, optimizer):
    if config.optimizer.scheduler_option == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config.optimizer.schedulers.cosine.t_max, eta_min=config.optimizer.schedulers.cosine.eta_min)
    elif config.optimizer.scheduler_option == "cosine_constant_200k":
        assert config.optimizer.rl.training_steps == 200_000
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config.optimizer.schedulers.cosine_constant_200k.t_max, eta_min=config.optimizer.schedulers.cosine_constant_200k.eta_min)
    raise NotImplementedError(f"Scheduler {config.optimizer.scheduler} not implemented")
