"""Hailo 8L AI Accelerator driver for LawnBerry Pi (BEAD-100).

Provides edge inference capabilities using the 13 TOPS Hailo-8L accelerator.
Supports both real hardware and simulation modes.

Hardware Requirements:
- Raspberry Pi 5 with Hailo-8L M.2 AI HAT+
- Matching versions: driver 4.20.0, firmware 4.20.0, hailort 4.20.0
  OR driver 4.21.0, firmware 4.21.0, hailort 4.21.0

Note: Version mismatches between kernel driver, firmware, and library will
cause initialization to fail. Check versions with:
  - Driver: modinfo hailo_pci | grep version
  - Firmware: hailortcli fw-control identify
  - Library: python -c "import hailo_platform; print(hailo_platform.__version__)"
"""
from __future__ import annotations

import os
import time
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import numpy as np

from ...core.simulation import is_simulation_mode
from ..base import HardwareDriver


@dataclass
class HailoInferenceResult:
    """Result from a Hailo inference run."""
    outputs: Dict[str, np.ndarray]
    inference_time_ms: float
    model_name: str
    input_shape: tuple
    timestamp: float = field(default_factory=time.time)


@dataclass
class HailoModelInfo:
    """Information about a loaded Hailo model."""
    name: str
    hef_path: str
    input_names: List[str]
    input_shapes: Dict[str, tuple]
    output_names: List[str]
    output_shapes: Dict[str, tuple]


class HailoDriver(HardwareDriver):
    """Hailo 8L AI accelerator driver.

    Provides async lifecycle management and inference capabilities.
    In simulation mode, generates mock inference results for testing.

    Supported models:
    - yolov8m: Object detection (640x640 input)
    - scdepthv3: Monocular depth estimation

    Example usage:
        driver = HailoDriver({"hef_path": "/path/to/model.hef"})
        await driver.initialize()
        await driver.start()
        result = await driver.infer(input_data)
        await driver.stop()
    """

    # Default HEF model paths
    DEFAULT_MODELS = {
        "yolov8m": Path("/home/kp/repos/lawnberry_pi/hailo/yolov8m.hef"),
        "scdepthv3": Path("/home/kp/repos/lawnberry_pi/hailo/scdepthv3.hef"),
    }

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config=config)
        self._device = None
        self._network_group = None
        self._infer_pipeline = None
        self._model_info: Optional[HailoModelInfo] = None
        self._inference_count = 0
        self._total_inference_time_ms = 0.0
        self._last_inference_time_ms = 0.0
        self._hw_available = False
        self._hw_error: Optional[str] = None
        self._version_info: Dict[str, str] = {}

        # Get HEF path from config or use default
        self._hef_path = self.config.get("hef_path")
        if self._hef_path is None:
            model_name = self.config.get("model", "yolov8m")
            self._hef_path = str(self.DEFAULT_MODELS.get(model_name, self.DEFAULT_MODELS["yolov8m"]))

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

        if "yolo" in hef_name:
            self._model_info = HailoModelInfo(
                name="yolov8m_sim",
                hef_path=self._hef_path,
                input_names=["images"],
                input_shapes={"images": (1, 640, 640, 3)},
                output_names=["output0"],
                output_shapes={"output0": (1, 84, 8400)},  # YOLOv8m output format
            )
        elif "depth" in hef_name:
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
            os.environ['HAILO_SKIP_FW_VERSION_CHECK'] = '1'

            from hailo_platform import (
                VDevice, HEF, ConfigureParams, HailoStreamInterface,
                InferVStreams, InputVStreamParams, OutputVStreamParams
            )
            import hailo_platform

            lib_version = getattr(hailo_platform, "__version__", "unknown")

            # Create virtual device
            self._device = VDevice()

            # Load HEF model
            if not Path(self._hef_path).exists():
                raise FileNotFoundError(f"HEF file not found: {self._hef_path}")

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

            # Create inference pipeline
            input_params = InputVStreamParams.make_from_network_group(
                self._network_group, quantized=False
            )
            output_params = OutputVStreamParams.make_from_network_group(
                self._network_group, quantized=False
            )
            self._infer_pipeline = InferVStreams(
                self._network_group, input_params, output_params
            )

            self._hw_available = True
            self._version_info = {
                "mode": "hardware",
                "library_version": lib_version,
                "model": self._model_info.name,
                "device": "0001:01:00.0",  # PCIe address
            }

        except Exception as e:
            self._hw_available = False
            self._hw_error = str(e)
            self._version_info = {
                "mode": "hardware_failed",
                "error": self._hw_error,
            }
            # Fall back to simulation
            await self._init_simulation()

    async def start(self) -> None:
        """Start the driver (enter inference mode)."""
        if self._hw_available and self._infer_pipeline:
            self._infer_pipeline.__enter__()
        self.running = True

    async def stop(self) -> None:
        """Stop the driver and release resources."""
        self.running = False
        if self._hw_available and self._infer_pipeline:
            try:
                self._infer_pipeline.__exit__(None, None, None)
            except Exception:
                pass
        self._infer_pipeline = None

    async def health_check(self) -> Dict[str, Any]:
        """Return health snapshot for /health endpoint."""
        avg_inference_ms = (
            self._total_inference_time_ms / self._inference_count
            if self._inference_count > 0 else 0.0
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

    async def infer(
        self,
        input_data: np.ndarray | Dict[str, np.ndarray]
    ) -> HailoInferenceResult:
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

    async def _simulate_inference(
        self, input_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
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
            elif "yolo" in self._model_info.name.lower():
                # Simulate YOLO detection output
                outputs[name] = np.random.randn(*shape).astype(np.float32) * 0.1
            else:
                # Generic output
                outputs[name] = np.random.randn(*shape).astype(np.float32)

        return outputs

    def get_model_info(self) -> Optional[HailoModelInfo]:
        """Get information about the loaded model."""
        return self._model_info

    @property
    def hardware_available(self) -> bool:
        """Check if real hardware is available."""
        return self._hw_available


__all__ = ["HailoDriver", "HailoInferenceResult", "HailoModelInfo"]
