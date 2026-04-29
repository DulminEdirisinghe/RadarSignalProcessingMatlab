"""
Wavelet Parameter Tuner for FMCW Radar Signal Processing

This tool allows interactive tuning of wavelet parameters on a single frame.
Use this to compare different wavelet configurations and find the best ones
before running the full pipeline.

Features:
- Load a single frame from radar data
- Configure wavelet type, frequencies, and preprocessing parameters
- Switch between different parameter sets for quick comparison
- Save configurations for later use
"""
import torch
import os
import re
import sys
import json
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
from PIL import Image
import matplotlib.cm as cm
from dataclasses import dataclass
from typing import Dict, Optional, List

from fmcw_parameters import *
from helpers import get_valid_num_frames, read_master_bin
from ssqueezepy import cwt, Wavelet
from ssqueezepy.experimental import scale_to_freq


# ============================================================
# Configuration Classes
# ============================================================

@dataclass
class WaveletConfig:
    """Configuration for a wavelet parameter set"""
    name: str
    wavelet_type: str = "morlet"  # bump, gmw, morlet, hhhat, cmhat
    fmin_hz: float = 1e3
    fmax_hz: float = 3e6
    n_freq_bins: int = 1024
    dc_remove_mode: str = "hp"
    hp_win: int = 4096
    time_range_sec: tuple = (0.0, 0.001)  # Extended from 0.2ms to 1ms for better temporal resolution
    time_stride: int = 1
    
    def to_dict(self):
        return {
            'name': self.name,
            'wavelet_type': self.wavelet_type,
            'wavelet_param': self.wavelet_param,
            'fmin_hz': self.fmin_hz,
            'fmax_hz': self.fmax_hz,
            'n_freq_bins': self.n_freq_bins,
            'dc_remove_mode': self.dc_remove_mode,
            'hp_win': self.hp_win,
            'time_range_sec': self.time_range_sec,
            'time_stride': self.time_stride,
        }
    
    @staticmethod
    def from_dict(d: dict):
        return WaveletConfig(**d)


class WaveletProcessor:
    """Processes raw radar data with configurable wavelet parameters"""
    
    def __init__(self, config: WaveletConfig):
        self.config = config
        # Create wavelet - ssqueezepy supports: bump, gmw, morlet, hhhat, cmhat
        try:
            self.wavelet = Wavelet(config.wavelet_type)
            print(f"Created wavelet: {config.wavelet_type}")
        except Exception as e:
            print(f"Warning: Could not create wavelet '{config.wavelet_type}': {e}")
            print("Using default 'morlet' wavelet")
            self.wavelet = Wavelet('morlet')
    
    def _remove_dc(self, x: np.ndarray) -> np.ndarray:
        """Remove DC component using specified mode"""
        mode = self.config.dc_remove_mode.lower()
        
        if mode == "hp":
            # High-pass filter using hann window
            win = np.hanning(self.config.hp_win)
            win /= win.sum()
            mean = np.convolve(x, win, mode='same')
            return x - mean
        else:  # mean
            return x - np.mean(x)
    
    def _apply_time_range(self, x: np.ndarray, Fs: float):
        """Apply time range restriction"""
        t_min, t_max = self.config.time_range_sec
        
        if t_max is None:
            t_max = len(x) / Fs
        
        idx_min = max(0, int(t_min * Fs))
        idx_max = min(len(x), int(t_max * Fs))
        t_offset = t_min
        
        return x[idx_min:idx_max], t_offset
    
    def process(self, iq_data: np.ndarray, Fs: float = FS_FAST) -> Dict:
        """
        Process IQ data and return CWT results
        
        Args:
            iq_data: Complex IQ data vector
            Fs: Sampling frequency
            
        Returns:
            Dictionary containing CWT results and metadata
        """
        # DC removal
        x = self._remove_dc(iq_data)
        x, t_offset = self._apply_time_range(x, Fs)
        
        if x.size < 4:
            raise ValueError("Signal too short after time range restriction")
        
        # Time stride
        if self.config.time_stride > 1:
            x = x[::self.config.time_stride]
            Fs_eff = Fs / self.config.time_stride
        else:
            Fs_eff = Fs
        
        if x.size < 4:
            raise ValueError("Signal too short after time stride")
        
        # Time vector
        t = (np.arange(x.size, dtype=np.float64) / Fs_eff) + t_offset
        
        # CWT computation (split I/Q)
        I = np.real(x).astype(np.float32)
        Q = np.imag(x).astype(np.float32)
        
        WI, scales = cwt(I, wavelet=self.wavelet)
        WQ, _ = cwt(Q, wavelet=self.wavelet)
        
        # Combine energy
        Wx = (np.abs(WI) ** 2) + (np.abs(WQ) ** 2)
        
        # Scale to frequency
        f = scale_to_freq(scales, self.wavelet, N=len(x), fs=Fs_eff).astype(np.float64)
        
        # Band masking
        mask = np.isfinite(f) & (f >= self.config.fmin_hz) & (f <= self.config.fmax_hz)
        if np.any(mask):
            f = f[mask]
            Wx = Wx[mask, :]
        else:
            mask = np.isfinite(f)
            f = f[mask]
            Wx = Wx[mask, :]
        
        if f.size == 0:
            raise ValueError("No valid frequencies in specified range")
        
        # Sort by frequency
        sort_idx = np.argsort(f)
        f_plot = f[sort_idx]
        Wx_plot = Wx[sort_idx, :]
        
        # Resample to uniform frequency grid
        P_db = Wx_plot
        f_uniform = np.linspace(f_plot[0], f_plot[-1], self.config.n_freq_bins)
        
        # Use linear interpolation for smoother results instead of nearest-neighbor
        from scipy.interpolate import interp1d
        P_db_uniform = np.zeros((self.config.n_freq_bins, Wx_plot.shape[1]))
        for t_idx in range(Wx_plot.shape[1]):
            # Create interpolator for this time slice (frequency response at time t_idx)
            interp_func = interp1d(f_plot, P_db[:, t_idx], kind='linear', fill_value='extrapolate')
            P_db_uniform[:, t_idx] = interp_func(f_uniform)
        
        # Calculate power limits
        vmin = float(np.nanpercentile(P_db_uniform, 5))
        vmax = float(np.nanpercentile(P_db_uniform, 99.5))
        
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin = float(np.nanmin(P_db_uniform))
            vmax = float(np.nanmax(P_db_uniform))
            if not np.isfinite(vmin) or not np.isfinite(vmax):
                vmin = 0.0
                vmax = 1.0
            if vmax <= vmin:
                vmax = vmin + 1.0
        
        return {
            'P_db': P_db_uniform,
            'f': f_uniform,
            't': t,
            'vmin': vmin,
            'vmax': vmax,
            'config': self.config,
        }


class WaveletTunerApp:
    """Interactive wavelet tuner GUI"""
    
    def __init__(self, data_folder: str, frame_idx: int = 1, capture_idx: str = "0001"):
        self.data_folder = data_folder
        self.frame_idx = frame_idx
        self.capture_idx = capture_idx
        
        # Load data
        self._load_data()
        
        # Default configurations to compare
        self.configs: Dict[str, WaveletConfig] = {
            'morlet_1k': WaveletConfig(
                name='morlet_1k',
                wavelet_type='morlet',
                fmin_hz=1e3,
                fmax_hz=3e6,
                n_freq_bins=1024,
                time_range_sec=(0.0, 0.001),
            ),
            'gmw_1k': WaveletConfig(
                name='gmw_1k',
                wavelet_type='gmw',
                fmin_hz=1e3,
                fmax_hz=3e6,
                n_freq_bins=1024,
                time_range_sec=(0.0, 0.001),
            ),
            'morlet_2k': WaveletConfig(
                name='morlet_2k',
                wavelet_type='morlet',
                fmin_hz=1e3,
                fmax_hz=3e6,
                n_freq_bins=2048,
                time_range_sec=(0.0, 0.001),
            ),
        }
        
        self.current_config_name = 'morlet_1k'
        self.setup_ui()
    
    def _heatmap_to_rgb(self, heatmap_data: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        """Convert heatmap data to RGB using jet colormap"""
        # Normalize to [0, 1]
        if vmax <= vmin:
            normalized = np.zeros_like(heatmap_data)
        else:
            normalized = np.clip((heatmap_data - vmin) / (vmax - vmin), 0, 1).astype(np.float32)
        
        # Apply jet colormap
        cmap = cm.get_cmap('jet')
        rgba = cmap(normalized)  # shape: (H, W, 4)
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)  # Convert to uint8, drop alpha
        
        return rgb
    
    def _add_contours_to_rgb(self, rgb_image: np.ndarray, data: np.ndarray, levels: int = 10) -> np.ndarray:
        """Add white contour lines to RGB image"""
        from scipy import ndimage
        
        # Normalize data for contour levels
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)
        normalized = np.clip((data - vmin) / (vmax - vmin + 1e-10), 0, 1)
        
        # Generate contour levels
        contour_levels = np.linspace(0, 1, levels)
        
        # Create output image (copy of input)
        result = rgb_image.copy()
        
        # Draw contours
        for level in contour_levels[1:-1]:  # Skip first and last to avoid edges
            contour_mask = np.abs(normalized - level) < 0.02  # Tolerance band
            # Make contour white
            result[contour_mask] = [255, 255, 255]
        
        return result
    
    def _load_data(self):
        """Load a single frame of radar data"""
        print(f"Loading frame {self.frame_idx} from capture {self.capture_idx}...")
        
        try:
            self.iq_data = read_master_bin(
                self.data_folder,
                self.capture_idx,
                self.frame_idx,
                ADC_SAMPLES,
                NC_CHIRPS_PER_LOOP,
                NCHIRP_LOOPS
            )
            print(f"✓ Data loaded. Shape: {self.iq_data.shape}")
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            raise
    
    def setup_ui(self):
        """Setup PyQtGraph UI"""
        self.app = pg.mkQApp("Wavelet Tuner")
        
        # Main window
        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle("Wavelet Parameter Tuner")
        self.win.resize(1400, 800)
        
        # Central widget with main layout (plot + controls)
        central_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout()
        central_widget.setLayout(main_layout)
        self.win.setCentralWidget(central_widget)
        
        # Left side: Plot
        self.plot_widget = pg.PlotWidget(title="Wavelet Transform")
        self.img = pg.ImageItem()
        # Enable smooth interpolation to avoid pixelation
        self.img.setOpts(axisOrder='row-major')
        self.img.setLevels((0, 255), update=False)
        # Apply bicubic interpolation for smoother rendering
        self.img.setLookupTable(np.arange(256)[:, None] * np.ones(3))
        self.plot_widget.addItem(self.img)
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.setLabel('left', 'Frequency', units='Hz')
        main_layout.addWidget(self.plot_widget, stretch=3)
        
        # Right side: Control panel
        self.ctrl_layout = QtWidgets.QVBoxLayout()
        ctrl_widget = QtWidgets.QWidget()
        ctrl_widget.setLayout(self.ctrl_layout)
        ctrl_widget.setMinimumWidth(250)
        ctrl_widget.setMaximumWidth(350)
        main_layout.addWidget(ctrl_widget, stretch=1)
        
        # Config selector
        self.config_combo = QtWidgets.QComboBox()
        self.config_combo.addItems(list(self.configs.keys()))
        self.config_combo.currentTextChanged.connect(self.on_config_changed)
        self.ctrl_layout.addWidget(QtWidgets.QLabel("<b>Configuration:</b>"))
        self.ctrl_layout.addWidget(self.config_combo)
        self.ctrl_layout.addSpacing(10)
        
        # Parameter controls
        self.param_inputs = {}
        
        # Wavelet type selector
        self.param_inputs['wavelet_type'] = QtWidgets.QComboBox()
        self.param_inputs['wavelet_type'].addItems(['bump', 'gmw', 'morlet', 'hhhat', 'cmhat'])
        self.param_inputs['wavelet_type'].setCurrentText('morlet')
        self.ctrl_layout.addWidget(QtWidgets.QLabel("<b>Wavelet Type:</b>"))
        self.ctrl_layout.addWidget(self.param_inputs['wavelet_type'])
        
        self.param_inputs['fmin_hz'] = QtWidgets.QSpinBox()
        self.param_inputs['fmin_hz'].setRange(0, int(1e8))
        self.param_inputs['fmin_hz'].setValue(int(1e3))
        self.ctrl_layout.addWidget(QtWidgets.QLabel("<b>Min Freq (Hz):</b>"))
        self.ctrl_layout.addWidget(self.param_inputs['fmin_hz'])
        
        self.param_inputs['fmax_hz'] = QtWidgets.QSpinBox()
        self.param_inputs['fmax_hz'].setRange(0, int(1e8))
        self.param_inputs['fmax_hz'].setValue(int(3e6))
        self.ctrl_layout.addWidget(QtWidgets.QLabel("<b>Max Freq (Hz):</b>"))
        self.ctrl_layout.addWidget(self.param_inputs['fmax_hz'])
        
        self.param_inputs['n_freq_bins'] = QtWidgets.QSpinBox()
        self.param_inputs['n_freq_bins'].setRange(64, 4096)
        self.param_inputs['n_freq_bins'].setValue(1024)
        self.ctrl_layout.addWidget(QtWidgets.QLabel("<b>Frequency Bins:</b>"))
        self.ctrl_layout.addWidget(self.param_inputs['n_freq_bins'])
        
        # Contour checkbox
        self.contour_checkbox = QtWidgets.QCheckBox("Draw Contours")
        self.contour_checkbox.setChecked(False)
        self.ctrl_layout.addWidget(self.contour_checkbox)
        
        self.param_inputs['contour_levels'] = QtWidgets.QSpinBox()
        self.param_inputs['contour_levels'].setRange(3, 50)
        self.param_inputs['contour_levels'].setValue(10)
        self.ctrl_layout.addWidget(QtWidgets.QLabel("<b>Contour Levels:</b>"))
        self.ctrl_layout.addWidget(self.param_inputs['contour_levels'])
        
        self.ctrl_layout.addSpacing(15)
        
        # Update button
        update_btn = QtWidgets.QPushButton("Update Plot")
        update_btn.setStyleSheet("background-color: #0078d4; color: white; font-weight: bold; padding: 8px;")
        update_btn.clicked.connect(self.on_update_plot)
        self.ctrl_layout.addWidget(update_btn)
        
        # Save config button
        save_btn = QtWidgets.QPushButton("Save Config")
        save_btn.setStyleSheet("background-color: #107c10; color: white; font-weight: bold; padding: 8px;")
        save_btn.clicked.connect(self.on_save_config)
        self.ctrl_layout.addWidget(save_btn)
        
        self.ctrl_layout.addSpacing(15)
        
        # Info label
        self.info_label = QtWidgets.QLabel()
        self.info_label.setStyleSheet("border: 1px solid #ccc; padding: 10px; background-color: #f5f5f5;")
        self.ctrl_layout.addWidget(QtWidgets.QLabel("<b>Info:</b>"))
        self.ctrl_layout.addWidget(self.info_label)
        
        self.ctrl_layout.addStretch()
        
        # Show window
        self.win.show()
        
        # Plot initial configuration
        self.on_update_plot()
    
    def on_config_changed(self, config_name: str):
        """Handle configuration selection change"""
        self.current_config_name = config_name
        config = self.configs[config_name]
        
        # Update input fields
        self.param_inputs['wavelet_type'].setCurrentText(config.wavelet_type)
        self.param_inputs['fmin_hz'].setValue(int(config.fmin_hz))
        self.param_inputs['fmax_hz'].setValue(int(config.fmax_hz))
        self.param_inputs['n_freq_bins'].setValue(config.n_freq_bins)
    
    def on_update_plot(self):
        """Process and display with current parameters"""
        try:
            # Get current parameters
            config = WaveletConfig(
                name=self.current_config_name,
                wavelet_type=self.param_inputs['wavelet_type'].currentText(),
                fmin_hz=float(self.param_inputs['fmin_hz'].value()),
                fmax_hz=float(self.param_inputs['fmax_hz'].value()),
                n_freq_bins=self.param_inputs['n_freq_bins'].value(),
            )
            
            # Process data
            processor = WaveletProcessor(config)
            # Use first antenna, one group
            single_rx_data = self.iq_data[:, :, 0, 0]  # Select antenna 0
            result = processor.process(single_rx_data.flatten(order='F'))
            
            # Update image
            P_db = result['P_db']  # Shape: (n_freq, n_time)
            # Transpose so dimensions are (n_time, n_freq) -> X=time, Y=freq
            # Then flip vertically so highest frequency appears at top
            P_db_transposed = P_db.T  # Now (n_time, n_freq)
            P_db_display = np.flipud(P_db_transposed)
            
            # Convert to RGB using jet colormap
            rgb_image = self._heatmap_to_rgb(P_db_display, result['vmin'], result['vmax'])
            
            # Add contours if enabled
            if self.contour_checkbox.isChecked():
                num_levels = self.param_inputs['contour_levels'].value()
                rgb_image = self._add_contours_to_rgb(rgb_image, P_db_display, levels=num_levels)
            
            self.img.setImage(rgb_image, autoLevels=False)
            
            # Update rect with correct axis mapping
            # After transpose and flipud: pixel columns = time, pixel rows = frequency (high at top)
            t = result['t']
            f = result['f']
            if len(t) > 1 and len(f) > 1:
                # X-axis (columns): time from t[0] to t[-1]
                # Y-axis (rows): frequency from f[-1] (top) to f[0] (bottom)
                t_span = t[-1] - t[0]
                f_span = f[-1] - f[0]
                rect = QtCore.QRectF(t[0], f[-1], t_span, -f_span)
                self.img.setRect(rect)
            
            # Update title and info
            self.plot_widget.setTitle(
                f"Wavelet: {config.wavelet_type} | "
                f"Freq: {config.fmin_hz/1e3:.1f}kHz-{config.fmax_hz/1e6:.1f}MHz | "
                f"Bins: {config.n_freq_bins}"
            )
            
            info = (
                f"Config: {self.current_config_name}\n"
                f"Power Range: [{result['vmin']:.2e}, {result['vmax']:.2e}]\n"
                f"Shape: {P_db.shape}\n"
                f"Frame: {self.frame_idx} | Capture: {self.capture_idx}"
            )
            self.info_label.setText(info)
            
            print(f"✓ Plot updated: {config.name}")
            
        except Exception as e:
            print(f"✗ Error updating plot: {e}")
            import traceback
            traceback.print_exc()
    
    def on_save_config(self):
        """Save current configuration to JSON"""
        try:
            config = self.configs[self.current_config_name]
            filename = f"wavelet_config_{self.current_config_name}.json"
            
            with open(filename, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
            
            print(f"✓ Configuration saved to {filename}")
        except Exception as e:
            print(f"✗ Error saving configuration: {e}")
    
    def run(self):
        """Start the application"""
        self.app.exec()


# ============================================================
# Main Script
# ============================================================

if __name__ == "__main__":
    # Configuration
    DATA_FOLDER = r"D:\MLDataset-RAW\phantom\phantom_1m_128_03242024"
    FRAME_IDX = 10
    CAPTURE_IDX = "0000"
    
    print("=" * 70)
    print("Wavelet Parameter Tuner")
    print("=" * 70)
    print(f"\nData Folder: {DATA_FOLDER}")
    print(f"Frame: {FRAME_IDX}")
    print(f"Capture: {CAPTURE_IDX}")
    print("\nTip: Modify the configuration presets in the code to compare different")
    print("     wavelet types and parameter combinations.\n")
    
    try:
        app = WaveletTunerApp(DATA_FOLDER, FRAME_IDX, CAPTURE_IDX)
        app.run()
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
