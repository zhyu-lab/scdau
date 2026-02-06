import argparse
import inspect
import sys
sys.path.append("..")

from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from models.unet_without_cond import Linear_UNet_nocn
from models.dae import DAE




def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    return dict(
        image_size=16,
        patch_size=2,
        embed_dim=64,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        window_size=4,
        qkv_bias=False,
        mlp_time_embed=False,
        learn_sigma=False,
        sigma_small=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="250",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        use_checkpoint=True,
    )



def create_model_and_diffusion(
    image_size,
    learn_sigma,
    sigma_small,
    patch_size,
    embed_dim,
    depth,
    num_heads,
    mlp_ratio,
    qkv_bias,
    mlp_time_embed,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    window_size,
):
    model = create_model(
        # image_size=image_size,
        # patch_size=patch_size,
        # embed_dim=embed_dim,
        # num_heads=num_heads,
        # depth=depth,
        # mlp_ratio=mlp_ratio,
        # qkv_bias=qkv_bias,
        # mlp_time_embed=mlp_time_embed,
        # use_checkpoint=use_checkpoint,
        # window_size=window_size,
        input_dim=128,
        cond_dim=64,
        hidden_dims=[512, 256, 128, 64],
        dropout=0.1
    )

    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion



def create_model(
        input_dim=128,
        cond_dim=64,
        hidden_dims=[512, 256, 128, 64],
        dropout=0.1
        
):
    
    return Linear_UNet_nocn(
        input_dim=64,
        hidden_dims=[512, 256, 128, 64],
        dropout=0.1
    )


def diffusion_defaults():
    """
    Defaults for diffusion.
    """
    return dict(
        learn_sigma=False,
        sigma_small=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="250",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True
    )


def create_dae(
    input_dim,
    emb_size,
    hidden_dims
):
   
    dae = DAE(
        input_dim=input_dim,
        emb_size=emb_size,
        hidden_dims=hidden_dims
    )

    return dae



def create_diffusion(
    learn_sigma,
    sigma_small,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas
):

    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing
    )

    return diffusion


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    
    betas = gd.get_named_beta_schedule(noise_schedule, steps)


    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps
    )



def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
