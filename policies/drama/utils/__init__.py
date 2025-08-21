"""
Utility functions for the DreamerV3 ([1]) algorithm.

[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf
"""

_ALLOWED_MODEL_DIMS = [
    "XS",
    "D",
]


def get_cnn_multiplier(model_size, override=None):
    if override is not None:
        return override

    assert model_size in _ALLOWED_MODEL_DIMS
    cnn_multipliers = {
        "XS": 24,
        "D": 48,
    }
    return cnn_multipliers[model_size]


def get_dense_hidden_units(model_size, override=None):
    if override is not None:
        return override

    assert model_size in _ALLOWED_MODEL_DIMS
    dense_units = {
        "XS": 64,
        "D": 256,
    }
    return dense_units[model_size]

def get_dense_hidden_units_reward(model_size, override=None):
    if override is not None:
        return override

    assert model_size in _ALLOWED_MODEL_DIMS
    dense_units = {
        "XS": 64,
        "D": 256,
    }
    return dense_units[model_size]

def get_dense_hidden_units_continue(model_size, override=None):
    if override is not None:
        return override

    assert model_size in _ALLOWED_MODEL_DIMS
    dense_units = {
        "XS": 64,
        "D": 256,
    }
    return dense_units[model_size]

def get_dense_hidden_units_encoder(model_size, override=None):
    if override is not None:
        return override

    assert model_size in _ALLOWED_MODEL_DIMS
    dense_units = {
        "XS": 64,
        "D": 256,
    }
    return dense_units[model_size]

def get_dense_hidden_units_decoder(model_size, override=None):
    if override is not None:
        return override

    assert model_size in _ALLOWED_MODEL_DIMS
    dense_units = {
        "XS": 64,
        "D": 256,
    }
    return dense_units[model_size]

def get_dense_hidden_units_actor(model_size, override=None):
    if override is not None:
        return override

    assert model_size in _ALLOWED_MODEL_DIMS
    dense_units = {
        "XS": 64,
        "D": 256,
    }
    return dense_units[model_size]

def get_dense_hidden_units_critic(model_size, override=None):
    if override is not None:
        return override

    assert model_size in _ALLOWED_MODEL_DIMS
    dense_units = {
        "XS": 128,
        "D": 512,
    }
    return dense_units[model_size]


def get_gru_units(model_size, override=None):
    if override is not None:
        return override

    assert model_size in _ALLOWED_MODEL_DIMS
    gru_units = {
        "XS": 256,
        "D": 1025,
    }
    return gru_units[model_size]


def get_num_z_categoricals(model_size, override=None):
    if override is not None:
        return override

    assert model_size in _ALLOWED_MODEL_DIMS
    gru_units = {
        "XS": 32,
        "D": 32,
    }
    return gru_units[model_size]


def get_num_z_classes(model_size, override=None):
    if override is not None:
        return override

    assert model_size in _ALLOWED_MODEL_DIMS
    gru_units = {
        "XS": 32,
        "D": 32,
    }
    return gru_units[model_size]


def get_num_curiosity_nets(model_size, override=None):
    if override is not None:
        return override

    assert model_size in _ALLOWED_MODEL_DIMS
    num_curiosity_nets = {
        "XS": 8,
        "D": 8,
    }
    return num_curiosity_nets[model_size]


def get_num_dense_layers(model_size, override=None):
    if override is not None:
        return override

    assert model_size in _ALLOWED_MODEL_DIMS
    num_dense_layers = {
        "XS": 1,
        "D": 2,
    }
    return num_dense_layers[model_size]

def get_num_dense_layers_reward(model_size, override=None):
    if override is not None:
        return override

    assert model_size in _ALLOWED_MODEL_DIMS
    num_dense_layers = {
        "XS": 1,
        "D": 1,
    }
    return num_dense_layers[model_size]

def get_num_dense_layers_continue(model_size, override=None):
    if override is not None:
        return override

    assert model_size in _ALLOWED_MODEL_DIMS
    num_dense_layers = {
        "XS": 1,
        "D": 1,
    }
    return num_dense_layers[model_size]

def get_num_dense_layers_encoder(model_size, override=None):
    if override is not None:
        return override

    assert model_size in _ALLOWED_MODEL_DIMS
    num_dense_layers = {
        "XS": 1,
        "D": 3,
    }
    return num_dense_layers[model_size]

def get_num_dense_layers_decoder(model_size, override=None):
    if override is not None:
        return override

    assert model_size in _ALLOWED_MODEL_DIMS
    num_dense_layers = {
        "XS": 1,
        "D": 3,
    }
    return num_dense_layers[model_size]

def get_num_dense_layers_actor(model_size, override=None):
    if override is not None:
        return override

    assert model_size in _ALLOWED_MODEL_DIMS
    num_dense_layers = {
        "XS": 1,
        "D": 3,
    }
    return num_dense_layers[model_size]

def get_num_dense_layers_critic(model_size, override=None):
    if override is not None:
        return override

    assert model_size in _ALLOWED_MODEL_DIMS
    num_dense_layers = {
        "XS": 1,
        "D": 3,
    }
    return num_dense_layers[model_size]


def do_symlog_obs(observation_space, symlog_obs_user_setting):
    # If our symlog_obs setting is NOT set specifically (it's set to "auto"), return
    # True if we don't have an image observation space, otherwise return False.

    # TODO (sven): Support mixed observation spaces.

    is_image_space = len(observation_space.shape) in [2, 3]
    return (
        not is_image_space
        if symlog_obs_user_setting == "auto"
        else symlog_obs_user_setting
    )
