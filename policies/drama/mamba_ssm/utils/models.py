_ALLOWED_MODEL_DIMS = [
    "XS",
    "D",
]

def get_model_units(model_size, override=None):
    if override is not None:
        return override

    assert model_size in _ALLOWED_MODEL_DIMS
    units = {
        "XS": 32,
        "D": 2560,
    }
    return units[model_size]

def get_n_layer(model_size, override=None):
    if override is not None:
        return override

    assert model_size in _ALLOWED_MODEL_DIMS
    units = {
        "XS": 2, # EMRAN used to be 32 
        "D": 4, # EMRAN used to be 64
    }
    return units[model_size]

def get_intermediate_units(model_size, override=None):
    if override is not None:
        return override

    assert model_size in _ALLOWED_MODEL_DIMS
    units = {
        "XS": 0,
        "D": 0,
    }
    return units[model_size]

def get_stoch_dim(model_size, override=None):
    if override is not None:
        return override

    assert model_size in _ALLOWED_MODEL_DIMS
    units = {
        "XS": 32,
        "D": 1024,
    }
    return units[model_size]