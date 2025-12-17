"""
PLACEHOLDER FILE - Phase 4/5 Implementation
Deployment utilities for flightrl_v2.

TODO: Implement model export when sim-to-real preparation begins.

Model export utilities for deployment.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import torch

def export_to_onnx(
        model_path: Union[str, Path],
        output_path: Union[str, Path],
        input_shape: Optional[Tuple[int, ...]] = None,
        opset_version: int = 11,
    ) -> Path:

    """
    Export a trained model to ONNX format.

    NOT YET IMPLEMENTED - Placeholder for Phase 4.

    Args:
        model_path: Path to trained SB3 model (.zip)
        output_path: Path for ONNX output
        input_shape: Input tensor shape (inferred if None)
        opset_version: ONNX opset version

    Returns:
        Path to exported ONNX model
    
    
    """

    raise NotImplementedError (
        "Phase 4: ONNX export not yet implemented. "
        "This will be implemented for sim-to-real transfer."
        )

def export_to_torchscript(
        model_path: Union[str, Path],
        output_path: Union[str, Path],
        use_tracing: bool = True,
) -> Path:
    """
    Export a trained model to TorchScript format.

    NOT YET IMPLEMENTED - Placeholder for Phase 4.

    Args:
        model_path: Path to trained SB3 model
        output_path: Path for TorchScript output
        use_tracing: Use tracing (vs scripting)

    Returns:
        Path to exported TorchScript model
    """
    raise NotImplementedError(
        "Phase 4: TorchScript export not yet implemented."
    )