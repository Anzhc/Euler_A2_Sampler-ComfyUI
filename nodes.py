import comfy.samplers
import torch
from comfy.k_diffusion import sampling as k_diffusion_sampling
from comfy.k_diffusion.sampling import default_noise_sampler
from tqdm.auto import trange


SAMPLER_NAME = "euler_a2"


@torch.no_grad()
def sample_euler_a2(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    noise_sampler=None,
    eta=1.0,
    s_noise=1.0,
    extrapolation=0.425,
):
    """Euler ancestral sampler that averages two noise paths and extrapolates along their mean direction."""
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})

        if sigmas[i + 1] == 0:
            x = denoised
            continue

        downstep_ratio = 1 + (sigmas[i + 1] / sigmas[i] - 1) * eta
        sigma_down = sigmas[i + 1] * downstep_ratio
        alpha_ip1 = 1 - sigmas[i + 1]
        alpha_down = 1 - sigma_down

        sigma_down_i_ratio = sigma_down / sigmas[i]
        deterministic_path = sigma_down_i_ratio * x + (1 - sigma_down_i_ratio) * denoised

        if eta > 0 and s_noise != 0:
            base = (alpha_ip1 / alpha_down) * deterministic_path
            renoise_coeff = (
                sigmas[i + 1] ** 2 - sigma_down ** 2 * alpha_ip1 ** 2 / alpha_down ** 2
            ).clamp_min(0).sqrt()
            noise_scale = s_noise * renoise_coeff

            noise_1 = noise_sampler(sigmas[i], sigmas[i + 1])
            noise_2 = noise_sampler(sigmas[i], sigmas[i + 1])

            path_1 = base + noise_1 * noise_scale
            path_2 = base + noise_2 * noise_scale
            merged = 0.5 * (path_1 + path_2)
            direction = merged - base
            x = merged + extrapolation * direction
        else:
            x = deterministic_path

    return x


def _append_unique(target, value):
    if value not in target:
        target.append(value)


def _register_sampler():
    setattr(k_diffusion_sampling, f"sample_{SAMPLER_NAME}", sample_euler_a2)

    _append_unique(comfy.samplers.KSAMPLER_NAMES, SAMPLER_NAME)
    _append_unique(comfy.samplers.SAMPLER_NAMES, SAMPLER_NAME)
    _append_unique(comfy.samplers.KSampler.SAMPLERS, SAMPLER_NAME)


_register_sampler()


class EulerA2Sampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": False}),
                "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": False}),
                "extrapolation": ("FLOAT", {"default": 0.425, "min": -10.0, "max": 10.0, "step": 0.001, "round": False}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "sampling/custom_sampling/samplers"

    def get_sampler(self, eta, s_noise, extrapolation):
        sampler = comfy.samplers.ksampler(
            SAMPLER_NAME,
            {
                "eta": eta,
                "s_noise": s_noise,
                "extrapolation": extrapolation,
            },
        )
        return (sampler,)


NODE_CLASS_MAPPINGS = {
    "Euler_A2_Sampler": EulerA2Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Euler_A2_Sampler": "Euler_A2_Sampler",
}
