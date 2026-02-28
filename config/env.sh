#!/bin/bash
# frontier_generator shell environment
# Reads defaults from config/defaults.yaml; env vars take precedence.

_CONFIG_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
_YAML="${_CONFIG_DIR}/defaults.yaml"

# Helper: read a value from defaults.yaml (simple key: value on its own line)
_yaml_val() {
    grep -E "^\s+${1}:" "$_YAML" 2>/dev/null | head -1 | sed 's/^[^:]*:\s*//' | sed 's/\s*#.*//'
}

export TIAMAT_DATA_DIR="${TIAMAT_DATA_DIR:-$(_yaml_val tiamat_data_dir)}"
export DETR_CHECKPOINT="${DETR_CHECKPOINT:-$(_yaml_val detr_checkpoint)}"
export GAIN_CHECKPOINT="${GAIN_CHECKPOINT:-$(_yaml_val gain_checkpoint)}"
export FRONTIER_ID="${FRONTIER_ID:-$(_yaml_val frontier_id)}"
