"""Centralized configuration for frontier_generator.

Usage
-----
    from config import load_config
    cfg = load_config()               # defaults + defaults.yaml (if present)
    cfg = load_config('my_cfg.yaml')  # explicit YAML overrides

The TIAMAT_DATA_DIR environment variable, if set, overrides
``cfg['paths']['tiamat_data_dir']``.
"""

import os
from copy import deepcopy
from pathlib import Path

# ── Hardcoded defaults (works with zero config files) ───────────────────────

_DEFAULTS = {
    # Paths — edit config/defaults.yaml for your machine
    'paths': {
        'tiamat_data_dir': None,
        'detr_checkpoint': None,
        'gain_checkpoint': None,
        'frontier_id':     7000,
    },

    # Refinement constants (shared by v1/v2/v3)
    'refinement': {
        'conf_thresh':        0.3,
        'atlas_flood_radius': 15.0,
        'remove_gain_thresh': 1.0,
        'overlap_thresh':     0.50,
        'cover_radius':       3.0,
        'boundary_radius':    5.0,
        'sample_stride':      0.6,
        'add_gain_thresh':    0.0,
        'kappa':              5.0,
        'q_min':              0.2,
    },

    # v2 marginal-coverage refinement
    'v2': {
        'delta_drop':     1.0,
        'delta_keep':     1.0,
        'delta_add':      5.0,
        'max_add_per_wp': 2,
        'eta':            0.0,
    },

    # v3 set-cover refinement
    'v3': {
        'coverage_frac':      0.65,
        'shift_radius_cells': 5,
        'wall_margin_cells':  3,
        'proximity_disq_m':   2.0,
    },

    # Self-supervised training
    'training': {
        'few_epochs':  10,
        'finetune_lr': 1e-5,
        'n_rounds':    3,
    },
}


def _deep_merge(base, override):
    """Recursively merge *override* into *base* (returns new dict)."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(yaml_path=None):
    """Load configuration with optional YAML overrides.

    Resolution order (later wins):
      1. Hardcoded ``_DEFAULTS``
      2. ``config/defaults.yaml`` (if it exists next to this file)
      3. Explicit *yaml_path* (if provided)
      4. ``TIAMAT_DATA_DIR`` environment variable
    """
    cfg = deepcopy(_DEFAULTS)

    # Auto-load defaults.yaml if it sits next to this __init__.py
    auto_yaml = Path(__file__).parent / 'defaults.yaml'
    if auto_yaml.exists():
        try:
            import yaml
            with open(auto_yaml) as f:
                overrides = yaml.safe_load(f) or {}
            cfg = _deep_merge(cfg, overrides)
        except ImportError:
            pass  # pyyaml not installed — skip YAML, use hardcoded defaults

    # Explicit YAML (highest-priority file overrides)
    if yaml_path is not None:
        import yaml
        with open(yaml_path) as f:
            overrides = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, overrides)

    # Environment variable override
    env_data = os.environ.get('TIAMAT_DATA_DIR')
    if env_data:
        cfg['paths']['tiamat_data_dir'] = env_data

    return cfg
