"""Hailo 8L AI Accelerator driver for LawnBerry Pi (BEAD-100).

Provides edge inference capabilities using the 13 TOPS Hailo-8L accelerator.
Supports both real hardware and simulation modes.

Hardware Requirements:
- Raspberry Pi 5 with Hailo-8L M.2 AI HAT+
- Matching versions: driver 4.24.0, firmware 4.24.0, hailort 4.24.0

Note: Version mismatches between kernel driver, firmware, and library will
cause initialization to fail. Check versions with:
  - Driver: modinfo hailo_pci | grep version
  - Firmware: hailortcli fw-control identify
  - Library: python -c "import hailo_platform; print(hailo_platform.__version__)"
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ...core.simulation import is_simulation_mode
from ..base import HardwareDriver


@dataclass
class HailoInferenceResult:
    """Result from a Hailo inference run."""

    outputs: dict[str, np.ndarray]
    inference_time_ms: float
    model_name: str
    input_shape: tuple
    timestamp: float = field(default_factory=time.time)


@dataclass
class HailoModelInfo:
    """Information about a loaded Hailo model."""

    name: str
    hef_path: str
    input_names: list[str]
    input_shapes: dict[str, tuple]
    output_names: list[str]
    output_shapes: dict[str, tuple]


class HailoDriver(HardwareDriver):
    """Hailo 8L AI accelerator driver.

    Provides async lifecycle management and inference capabilities.
    In simulation mode, generates mock inference results for testing.

    Supported models:
    - detector: YOLO26 object detection (see ``yolo26.py``)
    - scdepthv3: Monocular depth estimation

    Example usage:
        driver = HailoDriver({"hef_path": "/path/to/model.hef"})
        await driver.initialize()
        await driver.start()
        result = await driver.infer(input_data)
        await driver.stop()
    """

    # Default HEF model paths, relative to LAWNBERRY_DATA_DIR. Resolved at call
    # time (not import time) so the env var can be set after this module loads.
    DEFAULT_MODEL_RELPATHS = {
        "detector": "hailo/detector.hef",
        "scdepthv3": "hailo/scdepthv3.hef",
    }
    INPUT_FORMATS = ("float32", "uint8")

    @staticmethod
    def default_model_path(model_name: str) -> Path:
        """Default HEF location for `model_name`, under `LAWNBERRY_DATA_DIR`."""
        base = os.getenv("LAWNBERRY_DATA_DIR", "./data")
        relpath = HailoDriver.DEFAULT_MODEL_RELPATHS.get(
            model_name, HailoDriver.DEFAULT_MODEL_RELPATHS["detector"]
        )
        return Path(base) / relpath

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config=config)
        self._device = None
        self._network_group = None
        self._infer_pipeline = None
        self._activation = None
        self._model_info: HailoModelInfo | None = None
        self._inference_count = 0
        self._total_inference_time_ms = 0.0
        self._last_inference_time_ms = 0.0
        self._hw_available = False
        self._hw_error: str | None = None
        self._version_info: dict[str, str] = {}

        # Get HEF path from config or use default
        self._hef_path = self.config.get("hef_path")
        if self._hef_path is None:
            model_name = self.config.get("model", "detector")
            self._hef_path = str(self.default_model_path(model_name))
        self._input_format = self.config.get("input_format", "float32")
        if self._input_format not in self.INPUT_FORMATS:
            raise ValueError(
                f"input_format must be one of {self.INPUT_FORMATS}, got {self._input_format!r}"
            )

    async def initialize(self) -> None:
        """Initialize Hailo device and load model."""
        if is_simulation_mode() or os.environ.get("SIM_MODE") == "1":
            await self._init_simulation()
        else:
            await self._init_hardware()
        self.initialized = True

    async def _init_simulation(self) -> None:
        """Initialize in simulation mode with mock model info."""
        self._hw_available = False

        # Determine model type from HEF path
        hef_name = Path(self._hef_path).stem.lower()

        if "depth" in hef_name:
            self._model_info = HailoModelInfo(
                name="scdepthv3_sim",
                hef_path=self._hef_path,
                input_names=["input"],
                input_shapes={"input": (1, 256, 320, 3)},
                output_names=["depth"],
                output_shapes={"depth": (1, 256, 320, 1)},
            )
        else:
            # Generic model info
            self._model_info = HailoModelInfo(
                name=f"{hef_name}_sim",
                hef_path=self._hef_path,
                input_names=["input"],
                input_shapes={"input": (1, 224, 224, 3)},
                output_names=["output"],
                output_shapes={"output": (1, 1000)},
            )

        self._version_info = {
            "mode": "simulation",
            "model": self._model_info.name,
        }

    async def _init_hardware(self) -> None:
        """Initialize real Hailo hardware."""
        try:
            # Skip version check env var for 4.21 library compatibility
            os.environ["HAILO_SKIP_FW_VERSION_CHECK"] = "1"

            import hailo_platform
            from hailo_platform import (
                HEF,
                ConfigureParams,
                FormatType,
                HailoStreamInterface,
                InferVStreams,
                InputVStreamParams,
                OutputVStreamParams,
                VDevice,
            )

            lib_version = getattr(hailo_platform, "__version__", "unknown")

            # Check the model before claiming the device, so a missing HEF never
            # holds /dev/hailo0.
            if not Path(self._hef_path).exists():
                raise FileNotFoundError(f"HEF file not found: {self._hef_path}")

            # Create virtual device
            self._device = VDevice()

            hef = HEF(self._hef_path)
            network_groups = hef.get_network_group_names()

            # Configure network
            cfg_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
            self._network_group = self._device.configure(hef, cfg_params)[0]

            # Get stream info
            input_infos = self._network_group.get_input_vstream_infos()
            output_infos = self._network_group.get_output_vstream_infos()

            # Build model info
            self._model_info = HailoModelInfo(
                name=network_groups[0] if network_groups else Path(self._hef_path).stem,
                hef_path=self._hef_path,
                input_names=[info.name for info in input_infos],
                input_shapes={info.name: tuple(info.shape) for info in input_infos},
                output_names=[info.name for info in output_infos],
                output_shapes={info.name: tuple(info.shape) for info in output_infos},
            )

            # Create inference pipeline: uint8 input feeds the HEF's own
            # normalization; outputs are always dequantized to float32.
            in_format = FormatType.UINT8 if self._input_format == "uint8" else FormatType.FLOAT32
            input_params = InputVStreamParams.make(self._network_group, format_type=in_format)
            output_params = OutputVStreamParams.make(
                self._network_group, format_type=FormatType.FLOAT32
            )
            self._infer_pipeline = InferVStreams(self._network_group, input_params, output_params)

            self._hw_available = True
            self._version_info = {
                "mode": "hardware",
                "library_version": lib_version,
                "model": self._model_info.name,
                "device": "0001:01:00.0",  # PCIe address
            }

        except Exception as e:
            self._release_device()
            self._hw_available = False
            self._hw_error = str(e)
            self._version_info = {
                "mode": "hardware_failed",
                "error": self._hw_error,
            }
            # Fall back to simulation
            await self._init_simulation()

    def _release_device(self) -> None:
        """Release the VDevice so another process can claim /dev/hailo0."""
        if self._device is not None:
            try:
                self._device.release()
            except Exception:
                pass
        self._device = None
        self._network_group = None

    async def start(self) -> None:
        """Start the driver (enter inference mode).

        The default VDevice has no scheduler, so the network group must be
        activated explicitly while the vstreams are open.
        """
        if self._hw_available and self._infer_pipeline:
            self._infer_pipeline.__enter__()
            self._activation = self._network_group.activate(self._network_group.create_params())
            self._activation.__enter__()
        self.running = True

    async def stop(self) -> None:
        """Stop the driver and release resources."""
        self.running = False
        if self._activation is not None:
            try:
                self._activation.__exit__(None, None, None)
            except Exception:
                pass
            self._activation = None
        if self._hw_available and self._infer_pipeline:
            try:
                self._infer_pipeline.__exit__(None, None, None)
            except Exception:
                pass
        self._infer_pipeline = None
        self._release_device()

    async def health_check(self) -> dict[str, Any]:
        """Return health snapshot for /health endpoint."""
        avg_inference_ms = (
            self._total_inference_time_ms / self._inference_count
            if self._inference_count > 0
            else 0.0
        )

        return {
            "driver": "hailo_8l",
            "initialized": self.initialized,
            "running": self.running,
            "hardware_available": self._hw_available,
            "hardware_error": self._hw_error,
            "model": self._model_info.name if self._model_info else None,
            "hef_path": self._hef_path,
            "inference_count": self._inference_count,
            "avg_inference_ms": round(avg_inference_ms, 2),
            "last_inference_ms": round(self._last_inference_time_ms, 2),
            "version_info": self._version_info,
            "simulation": not self._hw_available,
        }

    async def infer(self, input_data: np.ndarray | dict[str, np.ndarray]) -> HailoInferenceResult:
        """Run inference on input data.

        Args:
            input_data: Either a numpy array (for single-input models) or
                       a dict mapping input names to numpy arrays.

        Returns:
            HailoInferenceResult with output tensors and timing info.
        """
        if not self.initialized or not self.running:
            raise RuntimeError("Driver not initialized or not running")

        if not self._model_info:
            raise RuntimeError("No model loaded")

        # Normalize input to dict format
        if isinstance(input_data, np.ndarray):
            input_name = self._model_info.input_names[0]
            input_dict = {input_name: input_data}
        else:
            input_dict = input_data

        start_time = time.perf_counter()

        if self._hw_available and self._infer_pipeline:
            # Real hardware inference
            results = self._infer_pipeline.infer(input_dict)
            outputs = {name: results[name] for name in self._model_info.output_names}
        else:
            # Simulation inference
            outputs = await self._simulate_inference(input_dict)

        inference_time_ms = (time.perf_counter() - start_time) * 1000

        # Update statistics
        self._inference_count += 1
        self._total_inference_time_ms += inference_time_ms
        self._last_inference_time_ms = inference_time_ms

        return HailoInferenceResult(
            outputs=outputs,
            inference_time_ms=inference_time_ms,
            model_name=self._model_info.name,
            input_shape=tuple(next(iter(input_dict.values())).shape),
        )

    async def _simulate_inference(self, input_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Generate simulated inference outputs."""
        # Simulate inference latency (Hailo 8L typically 5-20ms)
        await asyncio.sleep(0.01)  # 10ms simulated latency

        outputs = {}
        for name, shape in self._model_info.output_shapes.items():
            if "depth" in name.lower() or "depth" in self._model_info.name.lower():
                # Simulate depth map with gradient
                h, w = shape[1], shape[2]
                y_coords = np.linspace(0, 1, h)[:, np.newaxis]
                depth = np.tile(y_coords, (1, w))
                depth = depth + np.random.uniform(-0.05, 0.05, (h, w))
                outputs[name] = depth.reshape(shape).astype(np.float32)
            else:
                # Generic output
                outputs[name] = np.random.randn(*shape).astype(np.float32)

        return outputs

    def get_model_info(self) -> HailoModelInfo | None:
        """Get information about the loaded model."""
        return self._model_info

    @property
    def hardware_available(self) -> bool:
        """Check if real hardware is available."""
        return self._hw_available

    @property
    def hardware_error(self) -> str | None:
        """Why hardware init failed (None when it succeeded or was not attempted)."""
        return self._hw_error


__all__ = ["HailoDriver", "HailoInferenceResult", "HailoModelInfo"]
