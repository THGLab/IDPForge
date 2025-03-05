# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
import urllib
import warnings
from argparse import Namespace
from pathlib import Path

import torch

import esm
from esm.esm2 import ESM2


def _has_regression_weights(model_name):
    """Return whether we expect / require regression weights;
    Right now that is all models except ESM-1v, ESM-IF, and partially trained ESM2 models"""
    return not ("esm1v" in model_name or "esm_if" in model_name or "270K" in model_name or "500K" in model_name)


def load_model_and_alphabet(model_name):
    if model_name.endswith(".pt"):  # treat as filepath
        return load_model_and_alphabet_local(model_name)
    else:
        return load_model_and_alphabet_hub(model_name)


def load_hub_workaround(url):
    try:
        data = torch.hub.load_state_dict_from_url(url, progress=False, map_location="cpu")
    except RuntimeError:
        # Pytorch version issue - see https://github.com/pytorch/pytorch/issues/43106
        fn = Path(url).name
        data = torch.load(
            f"{torch.hub.get_dir()}/checkpoints/{fn}",
            map_location="cpu",
        )
    except urllib.error.HTTPError as e:
        raise Exception(f"Could not load {url}, check if you specified a correct model name?")
    return data


def load_regression_hub(model_name):
    url = f"https://dl.fbaipublicfiles.com/fair-esm/regression/{model_name}-contact-regression.pt"
    regression_data = load_hub_workaround(url)
    return regression_data


def _download_model_and_regression_data(model_name):
    url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
    model_data = load_hub_workaround(url)
    if _has_regression_weights(model_name):
        regression_data = load_regression_hub(model_name)
    else:
        regression_data = None
    return model_data, regression_data


def load_model_and_alphabet_hub(model_name):
    model_data, regression_data = _download_model_and_regression_data(model_name)
    return load_model_and_alphabet_core(model_name, model_data, regression_data)


def load_model_and_alphabet_local(model_location):
    """Load from local path. The regression weights need to be co-located"""
    model_location = Path(model_location)
    model_data = torch.load(str(model_location), map_location="cpu")
    model_name = model_location.stem
    if _has_regression_weights(model_name):
        regression_location = str(model_location.with_suffix("")) + "-contact-regression.pt"
        regression_data = torch.load(regression_location, map_location="cpu")
    else:
        regression_data = None
    return load_model_and_alphabet_core(model_name, model_data, regression_data)


def has_emb_layer_norm_before(model_state):
    """Determine whether layer norm needs to be applied before the encoder"""
    return any(k.startswith("emb_layer_norm_before") for k, param in model_state.items())


def _load_model_and_alphabet_core_v2(model_data):
    def upgrade_state_dict(state_dict):
        """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
        prefixes = ["encoder.sentence_encoder.", "encoder."]
        pattern = re.compile("^" + "|".join(prefixes))
        state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
        return state_dict

    cfg = model_data["cfg"]["model"]
    state_dict = model_data["model"]
    state_dict = upgrade_state_dict(state_dict)
    alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
    model = ESM2(
        num_layers=cfg.encoder_layers,
        embed_dim=cfg.encoder_embed_dim,
        attention_heads=cfg.encoder_attention_heads,
        alphabet=alphabet,
        token_dropout=cfg.token_dropout,
    )
    return model, alphabet, state_dict


def load_model_and_alphabet_core(model_name, model_data, regression_data=None):
    if regression_data is not None:
        model_data["model"].update(regression_data["model"])

    if model_name.startswith("esm2"):
        model, alphabet, model_state = _load_model_and_alphabet_core_v2(model_data)
    else:
        raise RuntimeError("model not supported")

    expected_keys = set(model.state_dict().keys())
    found_keys = set(model_state.keys())

    if regression_data is None:
        expected_missing = {"contact_head.regression.weight", "contact_head.regression.bias"}
        error_msgs = []
        missing = (expected_keys - found_keys) - expected_missing
        if missing:
            error_msgs.append(f"Missing key(s) in state_dict: {missing}.")
        unexpected = found_keys - expected_keys
        if unexpected:
            error_msgs.append(f"Unexpected key(s) in state_dict: {unexpected}.")

        if error_msgs:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    model.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        if expected_missing - found_keys:
            warnings.warn(
                "Regression weights not found, predicting contacts will not produce correct results."
            )

    model.load_state_dict(model_state, strict=regression_data is not None)

    return model, alphabet


def esm2_t6_8M_UR50D():
    """6 layer ESM-2 model with 8M params, trained on UniRef50.

    Returns a tuple of (Model, Alphabet).
    """
    return load_model_and_alphabet_hub("esm2_t6_8M_UR50D")


def esm2_t12_35M_UR50D():
    """12 layer ESM-2 model with 35M params, trained on UniRef50.

    Returns a tuple of (Model, Alphabet).
    """
    return load_model_and_alphabet_hub("esm2_t12_35M_UR50D")


def esm2_t30_150M_UR50D():
    """30 layer ESM-2 model with 150M params, trained on UniRef50.

    Returns a tuple of (Model, Alphabet).
    """
    return load_model_and_alphabet_hub("esm2_t30_150M_UR50D")


def esm2_t33_650M_UR50D():
    """33 layer ESM-2 model with 650M params, trained on UniRef50.

    Returns a tuple of (Model, Alphabet).
    """
    return load_model_and_alphabet_hub("esm2_t33_650M_UR50D")


def esm2_t36_3B_UR50D():
    """36 layer ESM-2 model with 3B params, trained on UniRef50.

    Returns a tuple of (Model, Alphabet).
    """
    return load_model_and_alphabet_hub("esm2_t36_3B_UR50D")


def esm2_t48_15B_UR50D():
    """48 layer ESM-2 model with 15B params, trained on UniRef50.
    If you have OOM while loading this model, please refer to README
    on how to employ FSDP and ZeRO CPU offloading

    Returns a tuple of (Model, Alphabet).
    """
    return load_model_and_alphabet_hub("esm2_t48_15B_UR50D")
