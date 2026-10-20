"""
rl/visualization/__init__.py
============================
RL Audio Processing Visualization package for DAPN.
"""

from rl.visualization.audio_utils import (
    compute_audio_metrics,
    compute_waveform_peaks,
    load_audio_from_path,
    save_audio_to_path,
    tensor_to_base64_wav,
    tensor_to_wav_bytes,
)
from rl.visualization.config import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_PRESETS,
    WORKFLOW_STAGES,
    get_effects_spec_map,
)
from rl.visualization.server import VisualizationRequestHandler, run_server
from rl.visualization.state_extractor import (
    compute_reward_breakdown,
    extract_action_diagnostics,
    extract_observation_diagnostics,
    map_parameters_with_physical,
)
from rl.visualization.workflow_engine import WorkflowEngine

__all__ = [
    "WorkflowEngine",
    "run_server",
    "VisualizationRequestHandler",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "DEFAULT_PRESETS",
    "WORKFLOW_STAGES",
    "get_effects_spec_map",
    "compute_audio_metrics",
    "compute_waveform_peaks",
    "load_audio_from_path",
    "save_audio_to_path",
    "tensor_to_wav_bytes",
    "tensor_to_base64_wav",
    "map_parameters_with_physical",
    "extract_observation_diagnostics",
    "extract_action_diagnostics",
    "compute_reward_breakdown",
]
