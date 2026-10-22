"""Versioned Contact inference semantics, independent of posterior training."""

CONTACT_LATENT_MODES = ("zero", "prior_mean")


def validate_contact_latent_mode(kind, mode):
    if mode not in CONTACT_LATENT_MODES:
        raise ValueError("contact latent mode must be zero or prior_mean")
    if kind != "contact" and mode != "zero":
        raise ValueError("prior_mean is only supported by the contact model")
    return mode


def contact_latent_contract(mode):
    validate_contact_latent_mode("contact", mode)
    return {"version": "real_robot_contact_latent_v1",
            "training": "posterior_sample", "evaluation": mode,
            "deterministic_prior": True, "zero_reference": mode == "prior_mean"}


def saved_contact_latent_mode(run_config):
    """Missing metadata denotes the historical zero-latent evaluation path."""
    contract = (run_config or {}).get("contact_latent_config")
    if contract is None:
        return "zero"
    if not isinstance(contract, dict) or contract != contact_latent_contract(contract.get("evaluation")):
        raise ValueError("invalid contact_latent_config")
    return contract["evaluation"]


def resolve_contact_latent_mode(kind, requested=None, run_config=None, *, resume=False):
    saved = saved_contact_latent_mode(run_config)
    validate_contact_latent_mode(kind, saved)
    mode = validate_contact_latent_mode(kind, saved if requested is None else requested)
    if resume and mode != saved:
        raise ValueError("cannot change contact latent mode while resuming; start a separate run")
    return mode


def validate_contact_latent_identity(previous, current):
    if saved_contact_latent_mode(previous) != saved_contact_latent_mode(current):
        raise ValueError("contact_latent_config differs from this run")
