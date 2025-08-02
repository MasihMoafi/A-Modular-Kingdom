#!/usr/bin/env python3
"""       
Transformer Model (GTN variant) Training Script for Forex Data - Version 10.4 (Enhanced Architecture with Learnable Time Embeddings)

This script implements an advanced Gated Transformer Network (GTN) for forex market prediction,
incorporating several architectural improvements and optimizations:

1. **Enhanced Model Architecture**:
   - Learnable Time Embeddings: Dedicated embedding layers for hour, minute, and day features
   - Pre-normalization in Transformers: Using norm_first=True for improved training stability
   - Residual Connections: Added residual paths to help with gradient flow
   - Gradient Clipping: Implemented to prevent exploding gradients

2. **Feature Engineering Optimizations**:
   - Stationary Price Features: Added to capture market dynamics more effectively
   - Grouped Feature Processing: Separate projections for numerical and binary features
   - Temporal Feature Integration: Enhanced handling of time-based patterns

3. **Training Improvements**:
   - Adaptive Learning Rate: Dynamic adjustment based on validation performance
   - Cost-Sensitive Learning: Custom loss function considering market impact
   - Enhanced Regularization: Strategic dropout and weight decay

The model maintains the dual-tower architecture (step-wise and channel-wise) while incorporating
these enhancements for improved performance and stability. All evaluation metrics, SHAP analysis,
and backtesting capabilities remain intact.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report, precision_score, recall_score
import time
import logging
import os
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Union, Iterator
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import math
from packaging import version
import joblib
import gc
import torch.nn.functional as F
from scipy.stats import entropy
from collections import deque

try:
    import shap
except ImportError:
    shap = None # Optional dependency, checked for at runtime

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    GaussianHMM = None # Will check for this at runtime

# --- Device Configuration ---
def setup_device_and_env():
    """Configure device and environment settings for CUDA or MPS."""
    if torch.cuda.is_available():
        # Get GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # Convert to GB
        logging.info(f"Using GPU: {gpu_name} with {gpu_memory:.1f}GB memory")
        
        torch.cuda.empty_cache()
        
        # Enable TF32 for Ampere GPUs (T4, A100)
        if any(arch in gpu_name.lower() for arch in ['a100', 't4']):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logging.info("Enabled TF32 for Ampere GPU")
        
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        # Set default device to CUDA
        if version.parse(torch.__version__) >= version.parse("2.1"):
            torch.set_default_dtype(torch.float32)
            torch.set_default_device('cuda')
            logging.info("Set default dtype to float32 and device to cuda using new API.")
        else:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            logging.info("Set default tensor type to torch.cuda.FloatTensor using old API.")
        return True
    
    elif torch.backends.mps.is_available():
        logging.info("Apple MPS device detected.")
        # We don't set a global default device for MPS, it's handled via .to(device)
        return True

    else:
        logging.warning("No accelerated GPU available. Running on CPU.")
        return False

# Call the setup function
is_gpu_available = setup_device_and_env()

# Global dictionary to store attention weights
attention_weights_store = {}

# Configure deterministic behavior for reproducibility
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
try:
    # PyTorch 1.8+ API
    torch.use_deterministic_algorithms(True, warn_only=True)
except AttributeError:
    # PyTorch 1.7 and older API
    torch.backends.cudnn.deterministic = True
    logging.warning("Using older PyTorch version. Enabling deterministic mode via torch.backends.cudnn.deterministic.")

# Always set these regardless of PyTorch version
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.getLogger().setLevel(logging.INFO)

def get_device() -> torch.device:
    """Get the appropriate device for training."""
    if torch.cuda.is_available():
        # For Colab, we want to use CUDA
        device = torch.device("cuda:0")
        # Log GPU memory info
        gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
        logging.info(f"Using CUDA device with {gpu_memory:.1f}GB memory")
        return device
    elif torch.backends.mps.is_available():
        # Configure MPS for optimal performance
        # Fallback to CPU for ops not supported on MPS is enabled by default in recent PyTorch versions.
        # torch.backends.mps.enable_fallback_to_cpu = True
        return torch.device("mps")
    else:
        logging.warning("No GPU available. Using CPU.")
        return torch.device("cpu")

def get_amp_device(device: torch.device) -> str:
    """Get the appropriate device string for automatic mixed precision."""
    if device.type == 'cuda':
        return 'cuda'
    elif device.type == 'mps':
        return 'cpu'  # MPS doesn't support AMP, fallback to CPU
    else:
        return 'cpu'

# --- GPU Monitoring ---
def log_gpu_usage(device: torch.device):
    """Logs current GPU memory usage."""
    if device.type == 'cuda':
        try:
            gpu_properties = torch.cuda.get_device_properties(device)
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
            total_memory = gpu_properties.total_memory / 1024**3
            utilization = (memory_allocated / total_memory) * 100 if total_memory > 0 else 0

            logging.info(f"GPU: {gpu_properties.name}")
            logging.info(f"Total Memory: {total_memory:.2f} GB")
            logging.info(f"Allocated Memory: {memory_allocated:.2f} GB")
            logging.info(f"Reserved Memory: {memory_reserved:.2f} GB")
            logging.info(f"Utilization (Allocated/Total): {utilization:.2f}%")
        except Exception as e:
            logging.error(f"Could not get GPU details: {e}", exc_info=True)
    elif device.type == 'mps':
        logging.info("Using Apple MPS (Metal Performance Shaders)")
    else:
        logging.warning("Using CPU. No GPU usage to log.")

# --- Precision & Performance Tweaks for Ampere/Hopper GPUs ---
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0).lower()
    gpu_supports_tf32 = any(arch in gpu_name for arch in ['a100', 'a10', 'a30', 'a40', 'a6000', 'h100', 'h800', 'l40'])
    if gpu_supports_tf32:
        logging.info(f"Detected TensorFloat32-capable GPU: {gpu_name}. Enabling TF32.")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        logging.info(f"GPU {gpu_name} does not explicitly support TF32 acceleration.")
    
    try: # Set float32 matmul precision hint (PyTorch >=1.12)
        torch.set_float32_matmul_precision('high')
    except AttributeError:
        pass

# Improve PyTorch CUDA fragmentation handling
torch_version_str = torch.__version__ 
try:
    current_torch_version = version.parse(torch_version_str) 
    if current_torch_version >= version.parse('2.1.0'):
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True,max_split_size_mb:128')
    else:
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')
except Exception as e: 
    logging.warning(f"Could not set PYTORCH_CUDA_ALLOC_CONF due to version parsing or other issue: {e}")
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128') # Fallback

# --- Data Handling ---
def create_test_set(file_path: str, months_to_cut: int = 6, test_file_path_str: Optional[str] = None) -> Tuple[str, str]:
    logging.info(f"Creating test set by splitting last {months_to_cut} months of data from {file_path}...")
    original_file_path = Path(file_path)
    if test_file_path_str is None:
        test_file_p = original_file_path.parent / f"{original_file_path.stem}_test.csv"
    else:
        test_file_p = Path(test_file_path_str)
    train_split_file_p = original_file_path.parent / f"{original_file_path.stem}_train_split.csv"

    try:
        df = pd.read_csv(original_file_path)
        if df.empty: raise ValueError(f"Data file {original_file_path} is empty")
        date_col = next((col for col in ['date','Date','datetime','Datetime','time','Time','timestamp','Timestamp'] if col in df.columns), df.columns[0])
        # Use args.date_col if provided and exists, otherwise stick to the auto-detected or first column
        config_date_col = getattr(args, 'date_col', date_col) # Get from args if main() is calling, else use auto-detected
        if config_date_col in df.columns:
            date_col = config_date_col
        else:
            logging.warning(f"Specified date_col '{config_date_col}' not found, falling back to auto-detected '{date_col}'.")

        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col)
        latest_date = df[date_col].max()
        cutoff_date = latest_date - pd.DateOffset(months=months_to_cut)
        train_data = df[df[date_col] < cutoff_date].copy()
        test_data = df[df[date_col] >= cutoff_date].copy()
        if train_data.empty or test_data.empty:
            logging.warning(f"Temporal split resulted in empty train/test. Using 80/20 fallback.")
            df_original_for_fallback = pd.read_csv(original_file_path) 
            split_idx_fallback = int(len(df_original_for_fallback) * 0.8)
            train_data = df_original_for_fallback.iloc[:split_idx_fallback].copy()
            test_data = df_original_for_fallback.iloc[split_idx_fallback:].copy()
        train_data.to_csv(train_split_file_p, index=False) 
        test_data.to_csv(test_file_p, index=False)
        logging.info(f"Train set ({train_data.shape[0]}) saved to {train_split_file_p}, Test set ({test_data.shape[0]}) saved to {test_file_p}")
        return str(train_split_file_p), str(test_file_p)
    except Exception as e:
        logging.error(f"Error creating test set: {e}", exc_info=True); raise

def load_data(file_path: str) -> pd.DataFrame:
    logging.info(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        if df.empty: raise ValueError(f"Loaded dataframe from {file_path} is empty.")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}", exc_info=True); raise

def preprocess_and_split_data_temporal(df: pd.DataFrame, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], RobustScaler, LabelEncoder]:
    logging.info("Preprocessing and splitting data temporally...")
    label_encoder = LabelEncoder()
    try:
        unique_targets_in_df = df[args.target_col].unique()
        label_encoder.fit(unique_targets_in_df)
        if set(args.target_classes) != set(label_encoder.classes_):
             logging.warning(f"LabelEncoder classes {list(label_encoder.classes_)} differ from args.target_classes {args.target_classes}. Using actual classes from data.")
        y = label_encoder.transform(df[args.target_col])
    except Exception as e:
        logging.error(f"Error encoding target '{args.target_col}': {e}", exc_info=True)
        raise
    logging.info(f"Target classes encoded. Mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

    if not args.date_col or args.date_col not in df.columns:
        raise ValueError(f"Date column '{args.date_col}' not found in DataFrame or not specified.")
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[args.date_col]):
        try:
            df[args.date_col] = pd.to_datetime(df[args.date_col])
        except Exception as e:
            logging.error(f"Failed to convert date column '{args.date_col}' to datetime: {e}", exc_info=True)
            raise

    # Extract dates before selecting features to ensure it's available for splitting
    dates = df[args.date_col].values

    if not all(f in df.columns for f in args.all_features):
        missing_features = [f for f in args.all_features if f not in df.columns]
        raise ValueError(f"Missing feature columns in DataFrame: {missing_features}")
    
    X_df = df[args.all_features].copy()
    
    # Verify binary features are actually binary
    for binary_feat in args.binary_features:
        if binary_feat in X_df.columns:
            unique_vals = X_df[binary_feat].unique()
            if not all(val in [0, 1] for val in unique_vals):
                logging.warning(f"Binary feature '{binary_feat}' contains non-binary values: {unique_vals}")
    
    split_idx = int(len(X_df) * (1.0 - args.val_size))
    if not (0 < split_idx < len(X_df) - 1):
         raise ValueError(f"Temporal split resulted in an empty/too small train or val set. Adjust val_size. Split idx: {split_idx}, len(X_df): {len(X_df)}")

    X_train_df, X_val_df = X_df.iloc[:split_idx].copy(), X_df.iloc[split_idx:].copy()
    y_train, y_val = y[:split_idx], y[split_idx:]
    dates_train, dates_val = dates[:split_idx], dates[split_idx:]
    
    logging.info(f"Temporal data split: Train={len(X_train_df)}, Val={len(X_val_df)}")
    
    scaler = RobustScaler()
    numerical_features_present = [f for f in args.numerical_features if f in X_train_df.columns]
    if numerical_features_present:
        X_train_df.loc[:, numerical_features_present] = scaler.fit_transform(X_train_df[numerical_features_present])
        if not X_val_df.empty:
            X_val_df.loc[:, numerical_features_present] = scaler.transform(X_val_df[numerical_features_present])
        logging.info(f"Scaled numerical features: {numerical_features_present}")
    
    return X_train_df.values, y_train, X_val_df.values, y_val, dates_train, dates_val, scaler, label_encoder

def preprocess_test_data(test_df: pd.DataFrame, scaler: RobustScaler, label_encoder: LabelEncoder, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    logging.info("Preprocessing test data...")
    if test_df.empty:
        return np.empty((0, len(args.all_features))), np.array([], dtype=int), np.array([], dtype='datetime64[ns]')
    
    if not args.date_col or args.date_col not in test_df.columns:
        raise ValueError(f"Date column '{args.date_col}' not found in test DataFrame or not specified.")

    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(test_df[args.date_col]):
        try:
            test_df[args.date_col] = pd.to_datetime(test_df[args.date_col])
        except Exception as e:
            logging.error(f"Failed to convert date column '{args.date_col}' in test_df to datetime: {e}", exc_info=True)
            raise
    dates_test = test_df[args.date_col].values

    try:
        y_test = label_encoder.transform(test_df[args.target_col])
    except Exception as e:
        logging.error(f"Error encoding test target: {e}", exc_info=True)
        raise
    
    if not all(f in test_df.columns for f in args.all_features):
        missing_features = [f for f in args.all_features if f not in test_df.columns]
        raise ValueError(f"Missing feature columns in test DataFrame: {missing_features}")
    
    X_test_df_features = test_df[args.all_features].copy()
    
    # Scale only numerical features
    numerical_features_present = [f for f in args.numerical_features if f in X_test_df_features.columns]
    if numerical_features_present:
        X_test_df_features.loc[:, numerical_features_present] = scaler.transform(X_test_df_features[numerical_features_present])
    
    return X_test_df_features.values, y_test, dates_test

def create_sequences(X: np.ndarray, y: np.ndarray, seq_length: int, dates: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if X.size == 0 or y.size == 0:
        num_feat = X.shape[1] if X.ndim == 2 and X.shape[0] > 0 else 0
        return np.empty((0, seq_length, num_feat)), np.empty((0,)), (np.empty((0,)) if dates is not None else None)
    n_samples, n_features = X.shape
    if n_samples < seq_length:
        logging.warning(f"Dataset length ({n_samples}) < sequence length ({seq_length}). Returning empty sequences.")
        return np.empty((0, seq_length, n_features)), np.empty((0,)), (np.empty((0,)) if dates is not None else None)
    if not X.flags['C_CONTIGUOUS']: X = np.ascontiguousarray(X)
    shape = (n_samples - seq_length + 1, seq_length, n_features)
    strides = (X.strides[0], X.strides[0], X.strides[1]) 
    try:
        sequences_X = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
    except ValueError as e: logging.error(f"Error with as_strided: {e}", exc_info=True); raise
    sequences_y = y[seq_length - 1:]
    sequences_dates = dates[seq_length - 1:] if dates is not None else None
    if sequences_X.shape[0] != sequences_y.shape[0]:
        raise RuntimeError("Sequence creation mismatch (X and y).")
    if sequences_dates is not None and sequences_X.shape[0] != sequences_dates.shape[0]:
        raise RuntimeError("Sequence creation mismatch (X and dates).")
    return sequences_X.copy(), sequences_y.copy(), sequences_dates.copy() if sequences_dates is not None else None

class ForexDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, dates: Optional[np.ndarray] = None):
        if X.ndim != 3: raise ValueError(f"X must be 3D, got {X.shape}")
        if y.ndim != 1: raise ValueError(f"y must be 1D, got {y.shape}")
        if X.shape[0] != y.shape[0]: raise ValueError("X,y sample count mismatch.")
        if dates is not None and X.shape[0] != dates.shape[0]: raise ValueError("X, dates sample count mismatch.")
        
        if X.shape[0] == 0:
            # Explicitly create on CPU, regardless of default device
            self.X = torch.empty((0, X.shape[1] if X.ndim==3 else 0, X.shape[2] if X.ndim==3 else 0), dtype=torch.float32, device='cpu')
            self.y = torch.empty((0,), dtype=torch.long, device='cpu')
            self.dates = np.array([], dtype='datetime64[ns]') if dates is not None else None
        else:
            # Explicitly create on CPU, regardless of default device
            self.X = torch.tensor(X, dtype=torch.float32, device='cpu')
            self.y = torch.tensor(y, dtype=torch.long, device='cpu')
            self.dates = dates # Store as numpy array of datetimes
            
    def __len__(self) -> int: return self.X.shape[0]
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, np.datetime64]]: 
        if self.dates is not None:
            return self.X[idx], self.y[idx], self.dates[idx]
        return self.X[idx], self.y[idx]

# --- Model Architecture ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        if d_model <= 0 or not (0 <= dropout <= 1) or max_len <= 0: raise ValueError("Invalid PE params")
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0: pe[:, 1::2] = torch.cos(position * div_term)
        else:
            if d_model > 1: pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1): raise ValueError(f"Seq len {x.size(1)} > PE max_len {self.pe.size(1)}")
        if x.size(2) != self.pe.size(2): raise ValueError(f"Input dim {x.size(2)} != PE d_model {self.pe.size(2)}")
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class GTN_v10_4(nn.Module):
    def __init__(self, numerical_dim: int, binary_dim: int,
                 d_model: int, n_heads: int, num_layers: int, d_ff: int, num_classes: int, seq_len_T: int,
                 dropout: float = 0.1, pe_max_len_default: int = 5000,
                 hour_emb_dim: int = 12, minute_emb_dim: int = 12, day_emb_dim: int = 12):
        super().__init__()
        # Validate parameters
        if any(p <= 0 for p in [seq_len_T, d_model, n_heads, num_layers, d_ff, num_classes]):
            raise ValueError("All GTN dimension parameters must be positive.")
        if any(p < 0 for p in [numerical_dim, binary_dim]):
            raise ValueError("Feature dimensions cannot be negative.")
        if d_model % n_heads != 0: raise ValueError("d_model must be divisible by n_heads.")
        
        self.d_model, self.seq_len_T = d_model, seq_len_T
        self.numerical_dim, self.binary_dim = numerical_dim, binary_dim

        # --- Learnable Time Embeddings ---
        self.hour_emb_dim = hour_emb_dim
        self.minute_emb_dim = minute_emb_dim
        self.day_emb_dim = day_emb_dim
        self.hour_embedding = nn.Embedding(24, hour_emb_dim)
        self.minute_embedding = nn.Embedding(60, minute_emb_dim)
        self.day_embedding = nn.Embedding(7, day_emb_dim)
        self.time_dim_embedded = hour_emb_dim + minute_emb_dim + day_emb_dim
        
        self.input_dim_C = numerical_dim + binary_dim + 3 # 3 integer time features

        # --- Grouped Feature Projections for Step-wise Tower ---
        # The total dimension fed into the main transformer will be d_model
        d_model_main_input = self.d_model - self.time_dim_embedded
        
        self.numerical_proj = nn.Linear(self.numerical_dim, d_model_main_input // 2) if self.numerical_dim > 0 else None
        self.binary_proj = nn.Linear(self.binary_dim, d_model_main_input - (d_model_main_input // 2)) if self.binary_dim > 0 else None
        
        self.step_wise_pos_encoder = PositionalEncoding(d_model, dropout, max_len=max(pe_max_len_default, seq_len_T))
        # Using Pre-LN for stability by setting norm_first=True
        step_encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, 'gelu', batch_first=True, norm_first=True)
        self.step_wise_transformer_encoder = nn.TransformerEncoder(step_encoder_layer, num_layers, nn.LayerNorm(d_model))
        
        # --- Channel-wise Tower (remains the same) ---
        self.channel_wise_input_proj = nn.Linear(seq_len_T, d_model)
        self.channel_wise_pos_encoder = PositionalEncoding(d_model, dropout, max_len=max(pe_max_len_default, self.input_dim_C))
        # Using Pre-LN for stability by setting norm_first=True
        channel_encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, 'gelu', batch_first=True, norm_first=True)
        self.channel_wise_transformer_encoder = nn.TransformerEncoder(channel_encoder_layer, num_layers, nn.LayerNorm(d_model))
        
        # --- Gating, Residual, and Output ---
        self.gate_linear = nn.Linear(d_model * 2, d_model) 
        self.final_norm = nn.LayerNorm(d_model) # For residual connection
        self.output_layer = nn.Linear(d_model, num_classes)
        self._init_parameters()

    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias) if m.bias is not None else None
            elif isinstance(m, nn.LayerNorm): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding): nn.init.xavier_uniform_(m.weight)
            
    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # --- Step-wise Tower with Grouped Projections & Time Embeddings ---
        # Split input tensor `x` along the feature dimension
        x_numerical = x[..., :self.numerical_dim]
        x_binary = x[..., self.numerical_dim : self.numerical_dim + self.binary_dim]
        # Time features are integers at the end of the input tensor
        x_time_int = x[..., self.numerical_dim + self.binary_dim:].long()
        
        x_hour, x_minute, x_day = x_time_int[..., 0], x_time_int[..., 1], x_time_int[..., 2]

        # Get time embeddings
        hour_emb = self.hour_embedding(x_hour)
        minute_emb = self.minute_embedding(x_minute)
        day_emb = self.day_embedding(x_day)
        time_embeddings = torch.cat([hour_emb, minute_emb, day_emb], dim=-1)

        # Project numerical and binary features
        projected_parts = []
        if self.numerical_proj: projected_parts.append(self.numerical_proj(x_numerical))
        if self.binary_proj: projected_parts.append(self.binary_proj(x_binary))
        
        # Concatenate projected features and time embeddings to form the final d_model representation
        x_projected_non_time = torch.cat(projected_parts, dim=-1)
        x_projected = torch.cat([x_projected_non_time, time_embeddings], dim=-1)
        
        x_step = self.step_wise_pos_encoder(x_projected)
        h_step = self.step_wise_transformer_encoder(x_step, src_key_padding_mask=src_key_padding_mask)
        
        if src_key_padding_mask is not None:
            expanded_mask = (~src_key_padding_mask).unsqueeze(-1).float() 
            h_step_pooled = (h_step * expanded_mask).sum(dim=1) / torch.clamp(expanded_mask.sum(dim=1), min=1e-9)
        else: 
            h_step_pooled = h_step.mean(dim=1)

        # --- Channel-wise Tower (unmodified from v10.1) ---
        x_channel_permuted = x.permute(0, 2, 1)
        x_channel = self.channel_wise_pos_encoder(self.channel_wise_input_proj(x_channel_permuted))
        h_chan_pooled = self.channel_wise_transformer_encoder(x_channel).mean(dim=1)
        
        # --- Gating, Residual, and Output ---
        concat_features = torch.cat((h_step_pooled, h_chan_pooled), dim=1)
        gate_vals = torch.sigmoid(self.gate_linear(concat_features))
        gated_output = gate_vals * h_step_pooled + (1 - gate_vals) * h_chan_pooled
        
        # Add residual connection from the projected input (before positional encoding)
        input_residual = x_projected.mean(dim=1)
        final_repr = self.final_norm(gated_output + input_residual)

        return self.output_layer(final_repr)


def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    if total_steps <= 0: return optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
    if total_steps <= warmup_steps: return optim.lr_scheduler.LambdaLR(optimizer, lambda s: float(s+1)/float(max(1,total_steps)))
    def lr_lambda(current_step):
        if current_step < warmup_steps: return float(current_step+1)/float(max(1,warmup_steps))
        prog = float(current_step-warmup_steps)/float(max(1,total_steps-warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# --- Training Loop ---
def validate_input_parameters(args):
    """Validate input parameters for the GTN model."""
    if args.sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if args.d_model <= 0:
        raise ValueError("d_model must be positive")
    if args.n_heads <= 0:
        raise ValueError("n_heads must be positive")
    if args.d_model % args.n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")
    if args.num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if args.d_ff <= 0:
        raise ValueError("d_ff must be positive")
    if not 0 <= args.dropout <= 1:
        raise ValueError("dropout must be between 0 and 1")
    if not 0 <= args.val_size < 1:
        raise ValueError("val_size must be between 0 and 1")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if args.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive")
    if args.pe_max_len_default < args.sequence_length:
        raise ValueError("pe_max_len_default should be greater than or equal to sequence_length")


def save_model_checkpoint(model: nn.Module, optimizer: optim.Optimizer, scheduler: Any, path: Path, args: argparse.Namespace, label_encoder: LabelEncoder, current_val_metrics: Dict[str, Any], epoch: int):
    """Save model checkpoint with full metadata."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'args': vars(args),
        'input_dim_C': model.input_dim_C,
        'numerical_dim': model.numerical_dim,
        'binary_dim': model.binary_dim,
        'seq_len_T': model.seq_len_T,
        'num_classes': model.output_layer.out_features,
        'hour_emb_dim': model.hour_emb_dim,
        'minute_emb_dim': model.minute_emb_dim,
        'day_emb_dim': model.day_emb_dim,
        'label_encoder_classes': list(label_encoder.classes_),
        'validation_metrics': current_val_metrics, # Save all validation metrics for this best epoch
        'random_state_torch': torch.get_rng_state(),
        'random_state_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'random_state_numpy': np.random.get_state()
    }
    torch.save(checkpoint, path, _use_new_zipfile_serialization=False)
    logging.info(f"Model checkpoint saved to {path}")

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: Optional[DataLoader], criterion: nn.Module,
                optimizer: optim.Optimizer, scheduler: Any, device: torch.device, epochs: int, patience: int,
                output_dir: Path, grad_accum_steps: int, args: argparse.Namespace, 
                label_encoder: LabelEncoder) -> nn.Module:
    
    # Validate input parameters first
    validate_input_parameters(args)
    
    best_primary_metric = -1.0 # "Average Buy/Sell Precision"
    epochs_no_improve = 0
    best_epoch_details = {} # Initialize to prevent potential UnboundLocalError
    
    adaptive_lambda_scheduler = None
    if args.use_cost_sensitive_loss:
        adaptive_lambda_scheduler = AdaptiveLambda(
            initial_lambda=args.cost_initial_lambda,
            max_lambda=args.cost_max_lambda,
            min_lambda=args.cost_min_lambda,
            patience=args.cost_lambda_patience,
            improvement_threshold=args.cost_lambda_improve_thresh,
            increase_factor=args.cost_lambda_increase_factor,
            decrease_factor=args.cost_lambda_decrease_factor
        )
        logging.info(f"CostSensitiveRegularizedLoss enabled. Initial Lambda: {args.cost_initial_lambda}")

    amp_device = get_amp_device(device)
    use_amp = device.type == 'cuda'  # Only use AMP for CUDA devices
    scaler = torch.amp.GradScaler(amp_device) if use_amp else None
    logging.info(f"Starting GTN training for {epochs} epochs. Patience: {patience}. Primary Metric: Avg BUY/SELL Precision.")
    logging.info(f"Using device: {device.type}, AMP: {use_amp}")
    
    metrics_df_path = output_dir / "training_metrics_history.csv"
    metrics_log_path = output_dir / "training_log.txt"
    best_model_checkpoint_path = output_dir / "model_best.pt" 

    with open(metrics_log_path, 'w') as f:
        f.write(f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model.__class__.__name__}\n")
        f.write(f"Args: {vars(args)}\n")
        log_header_cols = ["epoch","train_loss","train_acc","val_loss","val_acc","val_f1_macro",
                           "val_avg_buy_sell_precision","val_buy_precision","val_sell_precision",
                           f"val_daily_avg_confident_bs_gt{args.daily_threshold}","lr"]
        if args.use_cost_sensitive_loss:
            log_header_cols.append("lambda")
        # Add new cols for composite score
        log_header_cols.extend(["val_precision_deviation", "val_composite_score"])
        log_header = ",".join(log_header_cols) + "\n"
        f.write(log_header)
    
    metrics_history_list = [] # For creating DataFrame at the end

    for epoch in range(epochs):
        if device.type == 'cuda': torch.cuda.empty_cache()
        model.train()
        train_loss_sum, correct_train, total_train = 0.0, 0, 0
        
        for batch_idx, batch_data in enumerate(train_loader):
            inputs, targets = batch_data[0].to(device, non_blocking=True), batch_data[1].to(device, non_blocking=True)
            # dates_batch = batch_data[2] # Dates are now available if needed, e.g., for very specific logging
            
            if use_amp:
                with torch.amp.autocast(amp_device):
                    outputs = model(inputs)
                    if args.use_cost_sensitive_loss and isinstance(criterion, CostSensitiveRegularizedLoss) and adaptive_lambda_scheduler:
                        current_lambda_val = adaptive_lambda_scheduler.current_lambda # Use current, not updated yet for this batch
                        loss_per_sample = criterion(outputs, targets, current_lambda_val)
                    else:
                        loss_per_sample = criterion(outputs, targets)
                    loss_for_step = loss_per_sample / grad_accum_steps
                scaler.scale(loss_for_step).backward()
            else:
                outputs = model(inputs)
                if args.use_cost_sensitive_loss and isinstance(criterion, CostSensitiveRegularizedLoss) and adaptive_lambda_scheduler:
                    current_lambda_val = adaptive_lambda_scheduler.current_lambda
                    loss_per_sample = criterion(outputs, targets, current_lambda_val)
                else:
                    loss_per_sample = criterion(outputs, targets)
                loss_for_step = loss_per_sample / grad_accum_steps
                loss_for_step.backward()
            
            train_loss_sum += loss_per_sample.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5) # Add value clipping
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5) # Add value clipping
                    optimizer.step()
                
                # Schedulers that step per batch/iteration (like OneCycleLR and our Cosine LambdaLR)
                if scheduler and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step()

                optimizer.zero_grad()

        avg_train_loss = train_loss_sum / total_train if total_train > 0 else float('nan')
        train_acc = 100. * correct_train / total_train if total_train > 0 else 0.0
        
        current_val_metrics = {}
        avg_buy_sell_precision_epoch = 0.0
        daily_avg_confident_bs_metric = 0.0 # Renamed for clarity based on dynamic threshold
        current_lambda_for_log = args.cost_initial_lambda if args.use_cost_sensitive_loss and adaptive_lambda_scheduler else 'N/A'
        buy_precision_epoch = 0.0
        sell_precision_epoch = 0.0
        precision_deviation_epoch = 0.0
        composite_score_epoch = 0.0

        if val_loader and len(val_loader) > 0:
            val_results = evaluate_model_enhanced(model, val_loader, criterion, device, list(label_encoder.classes_), "Validation", 
                                                args=args, current_lambda_for_eval=adaptive_lambda_scheduler.current_lambda if adaptive_lambda_scheduler else 0.0)
            current_val_metrics = val_results # Store all returned metrics
            avg_buy_sell_precision_epoch = current_val_metrics.get('avg_buy_sell_precision', 0.0)
            buy_precision_epoch = current_val_metrics.get('buy_precision', 0.0)
            sell_precision_epoch = current_val_metrics.get('sell_precision', 0.0)
            precision_deviation_epoch = abs(buy_precision_epoch - sell_precision_epoch)
            composite_score_epoch = avg_buy_sell_precision_epoch - (args.precision_deviation_penalty * precision_deviation_epoch)
            current_val_metrics['precision_deviation'] = precision_deviation_epoch
            current_val_metrics['composite_score'] = composite_score_epoch

            if args.use_cost_sensitive_loss and adaptive_lambda_scheduler:
                current_lambda_for_log = adaptive_lambda_scheduler.update(composite_score_epoch) # Update lambda based on composite score

            # Calculate daily threshold stats using results from evaluate_model_enhanced
            if val_results.get('all_probs') is not None and val_results.get('all_preds') is not None and val_results.get('all_dates') is not None:
                daily_avg_confident_bs_metric = calculate_daily_threshold_stats( # MODIFIED
                    all_probs=val_results['all_probs'],
                    all_preds=val_results['all_preds'],
                    all_dates=val_results['all_dates'],
                    class_names=list(label_encoder.classes_),
                    threshold=args.daily_threshold # Using the args.daily_threshold
                )
            else:
                logging.warning("Could not calculate daily threshold stats: Missing probs, preds, or dates from validation.")
        
        epoch_metrics = {
            'epoch': epoch + 1, 'train_loss': avg_train_loss, 'train_acc': train_acc,
            'val_loss': current_val_metrics.get('loss', float('nan')), 
            'val_acc': current_val_metrics.get('accuracy', float('nan')),
            'val_f1_macro': current_val_metrics.get('f1_macro', float('nan')),
            'val_avg_buy_sell_precision': avg_buy_sell_precision_epoch,
            'val_buy_precision': buy_precision_epoch,
            'val_sell_precision': sell_precision_epoch,
            f'val_daily_avg_confident_bs_gt{args.daily_threshold}': daily_avg_confident_bs_metric, # MODIFIED: dynamic key
            'lr': optimizer.param_groups[0]['lr'],
            'lambda': current_lambda_for_log if args.use_cost_sensitive_loss else 'N/A',
            'val_precision_deviation': precision_deviation_epoch,
            'val_composite_score': composite_score_epoch
        }
        metrics_history_list.append(epoch_metrics)
        
        with open(metrics_log_path, 'a') as f:
            log_line_values = [
                f"{epoch_metrics['epoch']}", f"{epoch_metrics['train_loss']:.6f}", f"{epoch_metrics['train_acc']:.2f}",
                f"{epoch_metrics['val_loss']:.6f}", f"{epoch_metrics['val_acc']:.2f}",
                f"{epoch_metrics['val_f1_macro']:.4f}", f"{epoch_metrics['val_avg_buy_sell_precision']:.4f}",
                f"{epoch_metrics['val_buy_precision']:.4f}", f"{epoch_metrics['val_sell_precision']:.4f}",
                f"{daily_avg_confident_bs_metric:.2f}", f"{epoch_metrics['lr']:.6e}"
            ]
            if args.use_cost_sensitive_loss:
                lambda_val_str = f"{epoch_metrics['lambda']:.4f}" if isinstance(epoch_metrics['lambda'], float) else str(epoch_metrics['lambda'])
                log_line_values.append(lambda_val_str)
            
            log_line_values.extend([
                f"{epoch_metrics['val_precision_deviation']:.4f}", f"{epoch_metrics['val_composite_score']:.4f}"
            ])
            f.write(",".join(log_line_values) + "\n")
        
        log_msg = f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | " \
                  f"Val Loss: {epoch_metrics['val_loss']:.4f} | Val Acc: {epoch_metrics['val_acc']:.2f}% | " \
                  f"Val F1 Macro: {epoch_metrics['val_f1_macro']:.4f} | " \
                  f"Val AvgBSPrec: {avg_buy_sell_precision_epoch:.4f} (B:{buy_precision_epoch:.4f},S:{sell_precision_epoch:.4f}) Dev:{precision_deviation_epoch:.4f} | CompScore:{composite_score_epoch:.4f} | " \
                  f"Daily Avg (Buy/Sell > {args.daily_threshold}): {daily_avg_confident_bs_metric:.2f}"
        if args.use_cost_sensitive_loss:
            log_msg += f" | Lambda: {epoch_metrics['lambda']:.4f}"
        logging.info(log_msg)

        if val_loader and len(val_loader) > 0:
            # Primary metric for checkpointing and early stopping is now composite_score_epoch
            if composite_score_epoch > best_primary_metric:
                best_primary_metric = composite_score_epoch # This now tracks the best composite score
                epochs_no_improve = 0
                best_epoch_details = { # Capture only necessary details for logging
                    'epoch': epoch + 1,
                    'validation_metrics': current_val_metrics, 
                }
                save_model_checkpoint(model, optimizer, scheduler, best_model_checkpoint_path, args, label_encoder, current_val_metrics, epoch+1)
                logging.info(f"---> New best model (CompositeScore: {best_primary_metric:.4f}, AvgBSPrec: {avg_buy_sell_precision_epoch:.4f}, Deviation: {precision_deviation_epoch:.4f}) at E:{epoch+1}. Saved to {best_model_checkpoint_path}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logging.info(f"Early stopping at E:{epoch+1}. Best CompositeScore: {best_primary_metric:.4f} (AvgBSPrec: {best_epoch_details.get('validation_metrics',{}).get('avg_buy_sell_precision', -1):.4f}) at E:{best_epoch_details.get('epoch', 'N/A')}.")
                    break
            
            # Schedulers that step per epoch, after validation (like ReduceLROnPlateau)
            if scheduler and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(composite_score_epoch)
        
        if val_loader and (epoch+1) % args.benchmark_interval == 0:
             benchmark_model_performance(model, val_loader, criterion, device, list(label_encoder.classes_), epoch+1, output_dir, args, adaptive_lambda_scheduler)


    # Save full metrics history
    metrics_df = pd.DataFrame(metrics_history_list) # MODIFIED: use the df directly
    # Dynamically rename the daily metric column for saving if it exists
    daily_metric_col_name = f'val_daily_avg_confident_bs_gt{args.daily_threshold}'
    if daily_metric_col_name in metrics_df.columns: # Check if column was added
         pass # Already correctly named
    metrics_df.to_csv(metrics_df_path, index=False)
    logging.info(f"Full training metrics history saved to {metrics_df_path}")

    # Plotting
    if os.path.exists(metrics_df_path):
        df_metrics = pd.read_csv(metrics_df_path)
        if not df_metrics.empty:
            plt.figure(figsize=(20, 12)) # Adjusted figure size for more plots
            num_plots = 6
            if 'lambda' in df_metrics.columns and args.use_cost_sensitive_loss: num_plots +=1
            if 'val_composite_score' in df_metrics.columns: num_plots +=1
            
            plot_rows = 2
            plot_cols = (num_plots + plot_rows -1) // plot_rows # Calculate needed columns

            plt.subplot(plot_rows,plot_cols,1); plt.plot(df_metrics['epoch'], df_metrics['train_loss'], label='Train Loss'); plt.plot(df_metrics['epoch'], df_metrics['val_loss'], label='Val Loss'); plt.legend(); plt.title("Loss")
            plt.subplot(plot_rows,plot_cols,2); plt.plot(df_metrics['epoch'], df_metrics['train_acc'], label='Train Acc'); plt.plot(df_metrics['epoch'], df_metrics['val_acc'], label='Val Acc'); plt.legend(); plt.title("Accuracy")
            plt.subplot(plot_rows,plot_cols,3); plt.plot(df_metrics['epoch'], df_metrics['val_f1_macro'], label='Val F1 Macro'); plt.legend(); plt.title("Val F1 Macro")
            plt.subplot(plot_rows,plot_cols,4); plt.plot(df_metrics['epoch'], df_metrics['val_avg_buy_sell_precision'], label='Avg Buy/Sell Prec'); plt.legend(); plt.title("Avg Buy/Sell Precision")
            plt.subplot(plot_rows,plot_cols,5); plt.plot(df_metrics['epoch'], df_metrics['val_buy_precision'], label='BUY Prec'); plt.plot(df_metrics['epoch'], df_metrics['val_sell_precision'], label='SELL Prec'); plt.legend(); plt.title("BUY/SELL Precision")
            
            current_plot_idx = 6
            daily_metric_plot_col = f'val_daily_avg_confident_bs_gt{args.daily_threshold}'
            if daily_metric_plot_col in df_metrics.columns:
                plt.subplot(plot_rows,plot_cols,current_plot_idx); current_plot_idx +=1
                plt.plot(df_metrics['epoch'], df_metrics[daily_metric_plot_col], label=f'Daily Avg Confident BS > {args.daily_threshold}')
                plt.legend(); plt.title(f"Daily Confident BS > {args.daily_threshold}")

            if 'lambda' in df_metrics.columns and args.use_cost_sensitive_loss:
                plt.subplot(plot_rows,plot_cols,current_plot_idx); current_plot_idx +=1
                plt.plot(df_metrics['epoch'], df_metrics['lambda']); plt.title("Lambda (CSRL)");

            if 'val_composite_score' in df_metrics.columns:
                plt.subplot(plot_rows,plot_cols,current_plot_idx); current_plot_idx +=1
                plt.plot(df_metrics['epoch'], df_metrics['val_composite_score'], label='Val Composite Score'); 
                plt.plot(df_metrics['epoch'], df_metrics['val_avg_buy_sell_precision'], label='Val AvgBSPrec', linestyle=':'); # Overlay AvgBSPrec
                plt.legend(); plt.title("Val Composite Score")

            if 'lr' in df_metrics.columns and current_plot_idx <= plot_rows*plot_cols:
                 plt.subplot(plot_rows,plot_cols,current_plot_idx); current_plot_idx +=1
                 plt.plot(df_metrics['epoch'], df_metrics['lr']); plt.title("Learning Rate")

            plt.tight_layout(); plt.savefig(output_dir / 'training_summary_plots.png'); plt.close()

    # Load the best model state for returning, if one was saved
    if os.path.exists(best_model_checkpoint_path):
        best_ckpt = torch.load(best_model_checkpoint_path, map_location=device, weights_only=False)
        # For v10.4, we check for numerical_dim as a proxy for a valid model checkpoint
        if 'numerical_dim' in best_ckpt:
            model.load_state_dict(best_ckpt['model_state_dict'])
            logging.info(f"Loaded best model state from {best_model_checkpoint_path} (Epoch {best_ckpt['epoch']}, CompScore: {best_ckpt['validation_metrics'].get('composite_score', -1):.4f}) for final use.")
        else:
            logging.warning("Checkpoint is not a v10.4 model. Cannot load state_dict. Returning model from last epoch.")

    else:
        logging.warning("No best model checkpoint was saved during training (or no validation was performed). Returning model from last epoch.")
    return model

# --- Evaluation ---
def evaluate_model_enhanced(model: nn.Module, data_loader: Optional[DataLoader], criterion: nn.Module, 
                           device: torch.device, class_names: List[str], dataset_name: str = "Test",
                           args: argparse.Namespace = None, current_lambda_for_eval: float = 0.0) -> Dict[str, Any]:
    results = {'loss': float('nan'), 'accuracy': 0.0, 'f1_macro': 0.0, 'f1_weighted': 0.0, 
               'cm': np.array([]), 'report': "No data or error.", 
               'buy_precision': 0.0, 'sell_precision': 0.0, 'avg_buy_sell_precision': 0.0,
               'all_preds': None, 'all_probs': None, 'all_dates': None} # Ensure keys exist
    
    if not data_loader or (hasattr(data_loader, 'dataset') and len(data_loader.dataset) == 0):
        logging.warning(f"{dataset_name} loader empty. Skipping enhanced eval.")
        return results

    model.eval()
    loss_sum, total_samples_eval = 0.0, 0
    all_preds_np, all_targets_np = np.array([]), np.array([])
    all_dates_np = np.array([], dtype='datetime64[ns]') 
    all_probs_np = np.array([]) 
    
    try:
        with torch.no_grad():
            preds_list, targets_list, dates_list, probs_list = [], [], [], []
            for batch_data in data_loader:
                inputs, targets = batch_data[0], batch_data[1]
                dates_batch = batch_data[2] if len(batch_data) > 2 else None
                
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                
                outputs = None
                if device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        if isinstance(criterion, CostSensitiveRegularizedLoss):
                            loss = criterion(outputs, targets, current_lambda_for_eval)
                        else:
                            loss = criterion(outputs, targets)
                else:
                    outputs = model(inputs)
                    if isinstance(criterion, CostSensitiveRegularizedLoss):
                        loss = criterion(outputs, targets, current_lambda_for_eval)
                    else:
                        loss = criterion(outputs, targets)
                
                loss_sum += loss.item() * inputs.size(0)
                total_samples_eval += inputs.size(0)
                
                probabilities = torch.softmax(outputs, dim=1)
                max_probs, predicted = probabilities.max(1)
                
                preds_list.append(predicted.cpu().numpy())
                targets_list.append(targets.cpu().numpy())
                probs_list.append(probabilities.cpu().numpy()) 
                if dates_batch is not None:
                    dates_list.extend(list(dates_batch)) 

            if preds_list: all_preds_np = np.concatenate(preds_list)
            if targets_list: all_targets_np = np.concatenate(targets_list)
            if probs_list: all_probs_np = np.concatenate(probs_list)
            if dates_list: all_dates_np = np.array(dates_list, dtype='datetime64[ns]')

        if total_samples_eval == 0 or len(all_targets_np) == 0: 
            logging.warning(f"{dataset_name}: No samples/targets evaluated.")
            return results
        
        results['loss'] = loss_sum / total_samples_eval
        results['accuracy'] = 100. * (all_preds_np == all_targets_np).sum() / len(all_targets_np)
        labels_for_metrics = np.arange(len(class_names))
        results['f1_macro'] = f1_score(all_targets_np, all_preds_np, average='macro', zero_division=0, labels=labels_for_metrics)
        results['f1_weighted'] = f1_score(all_targets_np, all_preds_np, average='weighted', zero_division=0, labels=labels_for_metrics)
        results['cm'] = confusion_matrix(all_targets_np, all_preds_np, labels=labels_for_metrics)
        results['report'] = classification_report(all_targets_np, all_preds_np, target_names=class_names, zero_division=0, labels=labels_for_metrics)
        results['all_preds'] = all_preds_np 
        results['all_probs'] = all_probs_np 
        results['all_dates'] = all_dates_np

        prec_scores = precision_score(all_targets_np, all_preds_np, average=None, zero_division=0, labels=labels_for_metrics)
        
        buy_idx, sell_idx = -1, -1
        class_names_lower = [name.lower() for name in class_names]
        try: 
            buy_idx = class_names_lower.index("buy")
        except ValueError: 
            logging.warning("'buy' (case-insensitive) class not found in label_encoder.classes_ for precision calculation.")
        try: 
            sell_idx = class_names_lower.index("sell")
        except ValueError: 
            logging.warning("'sell' (case-insensitive) class not found in label_encoder.classes_ for precision calculation.")

        if buy_idx != -1 and buy_idx < len(prec_scores): 
            results['buy_precision'] = prec_scores[buy_idx]
        if sell_idx != -1 and sell_idx < len(prec_scores): 
            results['sell_precision'] = prec_scores[sell_idx]
        
        num_target_classes_found = 0
        current_sum_precision = 0.0
        if buy_idx != -1: # Only consider 'buy' if it was a target class
            current_sum_precision += results['buy_precision']
            num_target_classes_found += 1
        if sell_idx != -1: # Only consider 'sell' if it was a target class
            current_sum_precision += results['sell_precision']
            num_target_classes_found += 1
            
        if num_target_classes_found > 0:
            results['avg_buy_sell_precision'] = current_sum_precision / num_target_classes_found
        else: # If neither buy nor sell were target classes, or no predictions for them
            results['avg_buy_sell_precision'] = 0.0


        logging.info(f"{dataset_name} -> Loss:{results['loss']:.4f} Acc:{results['accuracy']:.2f}% F1M:{results['f1_macro']:.4f} "
                    f"AvgBSPrec:{results['avg_buy_sell_precision']:.4f} (B:{results['buy_precision']:.4f}, S:{results['sell_precision']:.4f})")
        
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}", exc_info=True)
        return results
        
    return results

def calculate_daily_threshold_stats(all_probs: np.ndarray, all_preds: np.ndarray, all_dates: np.ndarray, 
                                    class_names: List[str], threshold: float = 0.5) -> float:
    """
    Calculates the daily average number of predictions where the model's predicted probability
    for a buy/sell decision is greater than the threshold.
    """
    if all_probs is None or all_preds is None or all_dates is None or \
       all_probs.size == 0 or all_preds.size == 0 or all_dates.size == 0:
        return 0.0

    df = pd.DataFrame({
        'date': pd.to_datetime(all_dates).normalize(), 
        'predicted_class_idx': all_preds
    })

    buy_idx, sell_idx = -1, -1
    class_names_lower = [name.lower() for name in class_names]
    try: buy_idx = class_names_lower.index("buy")
    except ValueError: logging.debug("'buy' class not found for daily threshold analysis.")
    try: sell_idx = class_names_lower.index("sell")
    except ValueError: logging.debug("'sell' class not found for daily threshold analysis.")

    if buy_idx == -1 and sell_idx == -1:
        logging.warning("Neither 'buy' nor 'sell' classes found. Cannot calculate daily threshold stats.")
        return 0.0

    if buy_idx != -1: df['buy_prob'] = all_probs[:, buy_idx]
    else: df['buy_prob'] = 0.0
    if sell_idx != -1: df['sell_prob'] = all_probs[:, sell_idx]
    else: df['sell_prob'] = 0.0

    is_confident_buy = (df['predicted_class_idx'] == buy_idx) & (df['buy_prob'] > threshold) if buy_idx != -1 else pd.Series([False]*len(df))
    is_confident_sell = (df['predicted_class_idx'] == sell_idx) & (df['sell_prob'] > threshold) if sell_idx != -1 else pd.Series([False]*len(df))
    
    df['confident_buy_sell'] = is_confident_buy | is_confident_sell

    daily_counts = df.groupby('date')['confident_buy_sell'].sum()
    
    if daily_counts.empty:
        return 0.0
        
    return daily_counts.mean()

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Path):
    if cm.size == 0: logging.warning("CM empty, skipping plot."); return
    plt.figure(figsize=(max(8,len(class_names)), max(6,len(class_names)*0.8)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names[:cm.shape[1]], yticklabels=class_names[:cm.shape[0]])
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
    plt.tight_layout(); plt.savefig(save_path); plt.close()

def plot_attention_weights(model: GTN_v10_4, sample_input: torch.Tensor, layer_index: int, 
                           output_dir: Path, filename_prefix: str = "attention_heatmap", 
                           feature_names: Optional[List[str]] = None, average_heads: bool = True,
                           encoder_type: str = 'step_wise'):
    """
    Generates and saves a heatmap of attention weights for a specific layer in an encoder using monkey-patching.
    """
    global attention_weights_store
    model.eval() 
    output_dir.mkdir(parents=True, exist_ok=True)
    attention_weights_store.clear() 

    logging.debug(f"plot_attention_weights: Using attention_weights_store with ID: {id(attention_weights_store)}. Store cleared.")

    if sample_input.ndim == 2: 
        sample_input = sample_input.unsqueeze(0) 
    
    if sample_input.size(0) != 1:
        logging.warning(f"plot_attention_weights expects a single sample (batch size 1), got {sample_input.size(0)}. Using first sample.")
        sample_input = sample_input[0].unsqueeze(0)

    target_encoder = None
    if encoder_type == 'step_wise':
        if hasattr(model, 'step_wise_transformer_encoder'):
            target_encoder = model.step_wise_transformer_encoder
        else:
            logging.error("Model does not have 'step_wise_transformer_encoder'. Cannot visualize attention.")
            return
    elif encoder_type == 'channel_wise':
        if hasattr(model, 'channel_wise_transformer_encoder'):
            target_encoder = model.channel_wise_transformer_encoder
        else:
            logging.error("Model does not have 'channel_wise_transformer_encoder'. Cannot visualize attention.")
            return
    else:
        logging.error(f"Invalid encoder_type: {encoder_type}. Choose 'step_wise' or 'channel_wise'.")
        return

    if not (0 <= layer_index < len(target_encoder.layers)):
        logging.error(f"Invalid layer_index {layer_index} for {encoder_type} encoder with {len(target_encoder.layers)} layers.")
        return

    mha_module = target_encoder.layers[layer_index].self_attn
    
    original_mha_forward = mha_module.forward
    attention_key_for_plot = f'{encoder_type}_encoder_layer_{layer_index}_attn'

    def patched_mha_forward(*args, **kwargs):
        kwargs['need_weights'] = True
        kwargs['average_attn_weights'] = False 
        
        output_tuple = original_mha_forward(*args, **kwargs)
        
        if output_tuple is not None and len(output_tuple) == 2 and output_tuple[1] is not None:
            attention_weights_store[attention_key_for_plot] = output_tuple[1].detach().cpu()
        return output_tuple

    mha_module.forward = patched_mha_forward
    # print(f"PLOT_ATTENTION_WEIGHTS: Patched forward method for MHA module in {encoder_type} layer {layer_index}. Store ID: {id(attention_weights_store)}")

    try:
        with torch.no_grad():
            _ = model(sample_input.to(next(model.parameters()).device))
    finally:
        mha_module.forward = original_mha_forward
        # print(f"PLOT_ATTENTION_WEIGHTS: Restored original forward for MHA module in {encoder_type} layer {layer_index}.")

    # print(f"PLOT_ATTENTION_WEIGHTS: Checking store for key '{attention_key_for_plot}'. Current store keys: {list(attention_weights_store.keys())}")
    if attention_key_for_plot not in attention_weights_store:
        logging.error(f"Attention weights for '{attention_key_for_plot}' not found in store after patched forward. Store keys: {list(attention_weights_store.keys())}")
        return

    attn_weights = attention_weights_store[attention_key_for_plot] 
    
    if attn_weights.ndim == 4 and attn_weights.size(0) > 0:
        attn_weights_sample = attn_weights[0] 
    else:
        logging.error(f"Attention weights for '{attention_key_for_plot}' have unexpected shape: {attn_weights.shape}")
        return

    if average_heads:
        attn_weights_to_plot = attn_weights_sample.mean(dim=0).numpy() 
        title = f"Average Attention Weights ({attention_key_for_plot})"
    else:
        attn_weights_to_plot = attn_weights_sample[0].numpy() 
        title = f"Attention Weights - Head 0 ({attention_key_for_plot})"

    seq_len_plot = attn_weights_to_plot.shape[0]
    
    plt.figure(figsize=(max(10, seq_len_plot / 5), max(8, seq_len_plot / 6)))
    sns.heatmap(attn_weights_to_plot, cmap='viridis', cbar=True)
    
    if encoder_type == 'step_wise':
        tick_labels = [f"T{i+1}" for i in range(seq_len_plot)]
        plt.xticks(ticks=np.arange(seq_len_plot) + 0.5, labels=tick_labels, rotation=45, ha='right')
        plt.yticks(ticks=np.arange(seq_len_plot) + 0.5, labels=tick_labels, rotation=0)
        plt.xlabel("Key Time Steps")
        plt.ylabel("Query Time Steps")
    elif encoder_type == 'channel_wise':
        dim_being_attended = model.input_dim_C 
        if feature_names and len(feature_names) == dim_being_attended and seq_len_plot == dim_being_attended:
            tick_labels = feature_names
        else:
            tick_labels = [f"F{i+1}" for i in range(seq_len_plot)]
            if feature_names:
                logging.warning(f"Length of feature_names ({len(feature_names)}) doesn't match attention dim ({seq_len_plot}) or input_dim_C ({dim_being_attended}). Using generic F labels.")
        plt.xticks(ticks=np.arange(seq_len_plot) + 0.5, labels=tick_labels, rotation=90)
        plt.yticks(ticks=np.arange(seq_len_plot) + 0.5, labels=tick_labels, rotation=0)
        plt.xlabel("Key Features")
        plt.ylabel("Query Features")
    else: 
        tick_labels = [str(i) for i in range(seq_len_plot)]
        plt.xticks(ticks=np.arange(seq_len_plot) + 0.5, labels=tick_labels)
        plt.yticks(ticks=np.arange(seq_len_plot) + 0.5, labels=tick_labels)
        plt.xlabel("Key Sequence Position")
        plt.ylabel("Query Sequence Position")

    plt.title(title)
    plt.tight_layout()
    save_image_path = output_dir / f"{filename_prefix}_{attention_key_for_plot}.png"
    plt.savefig(save_image_path)
    plt.close()
    logging.info(f"Saved attention heatmap to {save_image_path}")

def auto_adjust_batch_size(seq_length: int, input_dim: int, initial_batch: int, device: torch.device, utilisation_target: float = 0.70, d_model_ref: Optional[int] = None) -> int:
    if device.type != 'cuda' or input_dim == 0 or seq_length == 0: return initial_batch
    log_gpu_usage(device)
    total_mem_bytes = torch.cuda.get_device_properties(device).total_memory
    eff_feat_dim = d_model_ref if d_model_ref is not None else input_dim
    per_sample_bytes_est = (seq_length * eff_feat_dim * 4 * 30) 
    per_sample_bytes_est = max(per_sample_bytes_est, 1)
    avail_mem = (total_mem_bytes * min(utilisation_target,0.85)) - (0.1*total_mem_bytes)
    max_samples = int(max(0, avail_mem) // per_sample_bytes_est) if per_sample_bytes_est > 0 else 0
    adj_batch = min(initial_batch, max(1, max_samples))
    if adj_batch < initial_batch*0.75 and adj_batch > 1: adj_batch = 2**int(math.log2(adj_batch))
    adj_batch = max(adj_batch,1)
    logging.info(f"[AutoBatch] Initial:{initial_batch} -> Adjusted:{adj_batch} (VRAM Util Target: ~{min(utilisation_target,0.85)*100:.0f}%)")
    return adj_batch


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered technical indicator features to DataFrame."""
    df = df.copy()
    # --- Redundant features commented out ---
    # df['rolling_mean_10'] = df['Close'].rolling(window=10, min_periods=1).mean()
    # df['rolling_std_10'] = df['Close'].rolling(window=10, min_periods=1).std().fillna(0)
    # df['sma_5'] = df['Close'].rolling(window=5, min_periods=1).mean()
    # df['sma_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    # df['sma_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
    
    # Exponential moving averages (KEPT)
    df['ema_5'] = df['Close'].ewm(span=5, adjust=False).mean()  # Short-term
    df['ema_20'] = df['Close'].ewm(span=20, adjust=False).mean()  # Medium-term
    df['ema_50'] = df['Close'].ewm(span=50, adjust=False).mean()  # Long-term
    
    # RSI (KEPT)
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(span=14, adjust=False).mean()
    roll_down = down.ewm(span=14, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD and signal line (KEPT)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # --- Redundant features commented out ---
    # bb_middle = df['Close'].rolling(window=20, min_periods=1).mean()
    # bb_std = df['Close'].rolling(window=20, min_periods=1).std().fillna(0)
    # df['bb_upper'] = bb_middle + 2 * bb_std
    # df['bb_lower'] = bb_middle - 2 * bb_std
    # df['bb_middle'] = bb_middle
    # df['csi_24'] = df['Close'].pct_change().rolling(window=24, min_periods=1).mean().fillna(0)
    
    # Fill any remaining NaNs
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.fillna(0, inplace=True)
    return df

def add_ichimoku_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Ichimoku Cloud features for machine learning, avoiding lookahead bias.
    """
    df = df.copy()
    
    # Tenkan-sen (Conversion Line)
    high_9 = df['High'].rolling(window=9, min_periods=1).max()
    low_9 = df['Low'].rolling(window=9, min_periods=1).min()
    df['tenkan_sen'] = (high_9 + low_9) / 2

    # Kijun-sen (Base Line)
    high_26 = df['High'].rolling(window=26, min_periods=1).max()
    low_26 = df['Low'].rolling(window=26, min_periods=1).min()
    df['kijun_sen'] = (high_26 + low_26) / 2

    # Senkou Span A and B for *current* time step (not shifted for plotting)
    current_senkou_a = ((df['tenkan_sen'] + df['kijun_sen']) / 2)
    high_52 = df['High'].rolling(window=52, min_periods=1).max()
    low_52 = df['Low'].rolling(window=52, min_periods=1).min()
    current_senkou_b = ((high_52 + low_52) / 2)

    # Derived ML Features (all based on current information to avoid lookahead bias)
    df['price_vs_current_cloud_top'] = df['Close'] - pd.concat([current_senkou_a, current_senkou_b], axis=1).max(axis=1)
    df['price_vs_current_cloud_bottom'] = df['Close'] - pd.concat([current_senkou_a, current_senkou_b], axis=1).min(axis=1)
    df['tenkan_minus_kijun'] = df['tenkan_sen'] - df['kijun_sen']
    df['current_cloud_thickness'] = abs(current_senkou_a - current_senkou_b)
    df['price_vs_chikou_comparison_point'] = df['Close'] - df['Close'].shift(26)
    
    # Fill any NaNs that were created
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.fillna(0, inplace=True)
    
    return df

def add_entropy_feature(df: pd.DataFrame, window_size: int = 20, num_bins: int = 10) -> pd.DataFrame:
    """
    Adds a rolling Shannon entropy feature based on price returns to quantify market uncertainty.
    """
    df = df.copy()
    returns = df['Close'].pct_change().fillna(0)
    
    # Using rolling().apply() for an efficient, vectorized-like calculation
    df['entropy_volatility'] = returns.rolling(window=window_size).apply(
        _calculate_entropy_for_rolling_apply, raw=True, kwargs={'num_bins': num_bins}
    )
    
    # Fill any NaNs that were created by the rolling window
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.fillna(0, inplace=True)
    return df

def add_advanced_features(df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    """
    Adds advanced, context-aware features to the DataFrame.
    - ATR for volatility measurement.
    - Volatility-normalized indicators.
    """
    df = df.copy()

    # 1. Calculate Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.ewm(span=atr_period, adjust=False).mean()

    # We need EMA and Bollinger Bands calculated first for these features.
    # To ensure this, we re-calculate them here if they don't exist,
    # or this function can be called after add_technical_indicators.
    if 'ema_20' not in df.columns:
        df['ema_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # Calculate Bollinger Bands width for normalization
    bb_middle = df['Close'].rolling(window=20, min_periods=1).mean()
    bb_std = df['Close'].rolling(window=20, min_periods=1).std().fillna(0)
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std

    # 2. Create Volatility-Normalized Features
    # Avoid division by zero by adding a small epsilon
    atr_safe = df['atr_14'].replace(0, np.nan).ffill().fillna(1e-9)
    df['close_minus_ema20_norm'] = (df['Close'] - df['ema_20']) / atr_safe
    df['bb_width_norm'] = (bb_upper - bb_lower) / atr_safe

    # --- Redundant features commented out ---
    # df['rsi_14_roc_1'] = df['rsi_14'].diff(1)
    # df['macd_roc_1'] = df['macd'].diff(1)

    # Fill any NaNs created by diff() or shift()
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.fillna(0, inplace=True)
    
    return df

def _calculate_entropy_for_rolling_apply(x: np.ndarray, num_bins: int) -> float:
    """Helper for add_entropy_feature to use with pandas rolling apply."""
    if len(x) < 2:
        return 0.0
    hist, _ = np.histogram(x, bins=num_bins, density=False)
    probabilities = hist / len(x)
    probabilities = probabilities[probabilities > 0] # Filter for log(0)
    if len(probabilities) == 0:
        return 0.0
    return entropy(probabilities, base=2)

def add_regime_features(df: pd.DataFrame, n_regimes: int = 2) -> pd.DataFrame:
    """Adds a market regime feature using a Gaussian Hidden Markov Model."""
    if GaussianHMM is None:
        logging.error("hmmlearn library not installed. Please install with 'pip install hmmlearn'. Skipping regime feature.")
        return df

    df = df.copy()
    # Use log returns for better numerical stability
    returns = np.log(df['Close'] / df['Close'].shift(1)).fillna(0).values.reshape(-1, 1)
    
    # Check for sufficient data
    if len(returns) < n_regimes * 25: # Heuristic check
        logging.warning(f"Not enough data ({len(returns)} points) to fit HMM. Skipping regime feature.")
        return df
        
    hmm_model = GaussianHMM(n_components=n_regimes, covariance_type="full", n_iter=300, random_state=42)
    try:
        hmm_model.fit(returns)
        regimes = hmm_model.predict(returns)
        df['market_regime'] = regimes
        logging.info(f"Successfully added market regime feature with {n_regimes} states.")
    except Exception as e:
        logging.error(f"Failed to fit HMM for market regimes: {e}. Skipping feature.")
    
    return df

# --- NEW: Enhanced Feature Functions ---
def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds conceptual market microstructure features.
    NOTE: This is a placeholder. It requires your data to have columns like
    'bid_price', 'bid_volume', 'ask_price', 'ask_volume'.
    """
    df = df.copy()
    logging.info("Attempting to add microstructure features...")
    required_cols = ['bid_price', 'bid_volume', 'ask_price', 'ask_volume']
    if all(col in df.columns for col in required_cols):
        # 1. Weighted Average Price (WAP)
        df['wap'] = (df['bid_price'] * df['ask_volume'] + df['ask_price'] * df['bid_volume']) / (df['bid_volume'] + df['ask_volume'])
        
        # 2. Bid-Ask Spread
        df['spread'] = df['ask_price'] - df['bid_price']
        
        # 3. Order Book Imbalance (OBI) - A very powerful predictor
        df['order_book_imbalance'] = (df['bid_volume'] - df['ask_volume']) / (df['bid_volume'] + df['ask_volume'])
        logging.info("Successfully added WAP, Spread, and OBI features.")
    else:
        # If data is not available, do not add placeholder columns.
        logging.warning("Microstructure data (bid/ask prices/volumes) not found. Skipping these features.")

    df.ffill(inplace=True); df.bfill(inplace=True); df.fillna(0, inplace=True)
    return df

def add_vol_dynamics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds features describing the dynamics of volatility and its interaction with momentum."""
    df = df.copy()
    logging.info("Adding volatility dynamics and interaction features...")
    # 1. Volatility of Volatility (how stable is the volatility?)
    # A 20-period standard deviation of the ATR. Requires atr_14 to exist.
    if 'atr_14' in df.columns:
        df['atr_volatility'] = df['atr_14'].rolling(window=20, min_periods=1).std()
    else:
        logging.warning("`atr_14` not found. Skipping `atr_volatility` feature.")

    # 2. Interaction Feature: Momentum * Volatility
    # Normalizes RSI by a volatility measure (ATR). Requires rsi_14 and atr_14.
    if 'rsi_14' in df.columns and 'atr_14' in df.columns:
        atr_safe = df['atr_14'].replace(0, np.nan).ffill().fillna(1e-9)
        df['rsi_x_atr_norm'] = df['rsi_14'] / atr_safe
    else:
        logging.warning("`rsi_14` or `atr_14` not found. Skipping `rsi_x_atr_norm` feature.")

    df.ffill(inplace=True); df.bfill(inplace=True); df.fillna(0, inplace=True)
    return df

def add_stationary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces raw price features with stationary versions (returns) and keeps
    relative measures like Body and High-Low.
    """
    df = df.copy()
    
    # Calculate returns to make price data more stationary
    # Use previous close for open return to capture overnight gap
    prev_close = df['Close'].shift(1)
    df['open_return'] = (df['Open'] - prev_close) / prev_close
    
    # Intra-candle returns based on Open price
    df['high_return'] = (df['High'] - df['Open']) / df['Open']
    df['low_return'] = (df['Low'] - df['Open']) / df['Open']
    df['close_return'] = (df['Close'] - df['Open']) / df['Open']

    # Keep Body and High-Low as they are already relative measures (price differences)
    # The original script calculates them from the raw data, so we assume they are present
    # or will be calculated before this function is called.
    # df['Body'] = abs(df['Close'] - df['Open'])
    # df['High-Low'] = df['High'] - df['Low']

    # Fill NaNs created by the shift operation
    df.fillna(0, inplace=True)
    
    return df


# --- v10.4 Feature Lists ---
# Features are now stationary (returns) and use integer time features for embeddings.
v10_4_numerical_features = ['open_return', 'high_return', 'low_return', 'close_return', 'Body', 'High-Low']
v10_4_binary_features = ['Is_Doji', 'Is_Spike', 'Is_Long_Shadow', 'gap']
# These are the integer time features that will be created from the date column.
v10_4_time_features = ['hour', 'minute', 'dayofweek']

# The order of features in all_features MUST match the order of concatenation in the model and data processing.
# Numerical -> Binary -> Time
v10_4_all_features = v10_4_numerical_features + v10_4_binary_features + v10_4_time_features

# For the argument parser, numerical features are scaled. Time features are not scaled as they are categorical.
v10_4_numerical_for_scaling = v10_4_numerical_features


# --- Main Execution ---
def main(args: argparse.Namespace):
    try:
        validate_input_parameters(args)
        
        # Set random seeds
        torch.manual_seed(args.random_state)
        if torch.cuda.is_available(): 
            torch.cuda.manual_seed_all(args.random_state)
            # Clear GPU cache at start
            torch.cuda.empty_cache()
            # Set memory allocator settings for Colab
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        np.random.seed(args.random_state)

        logging.info("Starting GTN Training Script v10.4 with Grouped Feature Encoding")
        logging.info(f"Script arguments: {vars(args)}")
        
        device = get_device()
        if device.type == 'cuda':
            # Log initial GPU memory state
            logging.info(f"Initial GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
            logging.info(f"Initial GPU memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f}GB")
        
        if device.type == 'mps':
            optimize_for_mps() 
        
        logging.info(f"Using device: {device}")
        
        if device.type in ['cuda', 'mps']: 
            log_gpu_usage(device)

        try:
            # Use a single, continuous data file and split it temporally
            train_split_file, test_file = create_test_set(args.train_file, months_to_cut=12)
            train_df = load_data(train_split_file)
            test_df = load_data(test_file)

            # Apply sampling to the training and test data to manage memory
            if args.data_sample_frac < 1.0:
                # Sample training data
                sample_size_train = int(len(train_df) * args.data_sample_frac)
                train_df = train_df.tail(sample_size_train) # Use most recent data
                logging.info(f"Using {args.data_sample_frac * 100:.0f}% of training data ({len(train_df)} samples).")
                
                # Sample test data
                sample_size_test = int(len(test_df) * args.data_sample_frac)
                test_df = test_df.tail(sample_size_test) # Use most recent data from test set
                logging.info(f"Using {args.data_sample_frac * 100:.0f}% of test data ({len(test_df)} samples).")

            # --- FULL FEATURE ENGINEERING PIPELINE ---
            # Apply stationary feature transformation
            logging.info("Applying stationary feature engineering (returns)...")
            train_df = add_stationary_features(train_df)
            test_df = add_stationary_features(test_df)

            # Add integer time features for embeddings
            logging.info("Adding integer time features for embeddings...")
            for df in [train_df, test_df]:
                if args.date_col in df.columns:
                    dt_series = pd.to_datetime(df[args.date_col])
                    df['hour'] = dt_series.dt.hour
                    df['minute'] = dt_series.dt.minute
                    df['dayofweek'] = dt_series.dt.dayofweek
                else:
                    raise ValueError(f"Date column '{args.date_col}' not found for creating time features.")
            
            # The other feature engineering steps from the original script are disabled by default
            # but can be re-enabled for experimentation.

            # --- DYNAMICALLY UPDATE FEATURE LISTS ---
            # Update feature lists in args based on columns that were actually created.
            available_columns = set(train_df.columns)
            
            original_all_features = set(args.all_features)

            args.all_features = [f for f in args.all_features if f in available_columns]
            args.numerical_features = [f for f in args.numerical_features if f in available_columns]
            args.binary_features = [f for f in args.binary_features if f in available_columns]
            
            final_all_features = set(args.all_features)
            
            dropped_features = original_all_features - final_all_features
            if dropped_features:
                logging.info(f"Dynamically excluded {len(dropped_features)} unavailable features: {sorted(list(dropped_features))}")

            X_train_np, y_train, X_val_np, y_val, dates_train, dates_val, scaler, label_encoder = preprocess_and_split_data_temporal(train_df, args)
            X_test_np, y_test, dates_test = preprocess_test_data(test_df, scaler, label_encoder, args)
            
            # Aggressively free memory
            del train_df, test_df
            gc.collect()

        except Exception as e:
            logging.error(f"Data loading/preprocessing failed: {e}", exc_info=True)
            return

        num_classes = len(label_encoder.classes_)
        input_dim_C = X_train_np.shape[1] if X_train_np.size > 0 else len(args.all_features)
        
        # For v10.4, get dims of feature groups
        numerical_dim = len([f for f in v10_4_numerical_features if f in args.all_features])
        binary_dim = len([f for f in v10_4_binary_features if f in args.all_features])
        # Time features are now handled by embeddings, so we don't need a separate time_dim here for model input
        
        logging.info(f"Input dims: Total={input_dim_C}, Numerical={numerical_dim}, Binary={binary_dim}, Time (integer cols)={len(v10_4_time_features)}")
        logging.info(f"Seq len T: {args.sequence_length}, Num classes: {num_classes} ({list(label_encoder.classes_)})")


        if args.auto_batch_size:
            args.batch_size = auto_adjust_batch_size(args.sequence_length, input_dim_C, args.batch_size, device, d_model_ref=args.d_model)

        class_weights = None
        if len(y_train) > 0:
            counts = np.bincount(y_train, minlength=num_classes)
            if np.all(counts > 0): 
                class_weights = torch.tensor(len(y_train) / (num_classes * counts.astype(float)), dtype=torch.float32).to(device)
                logging.info(f"Class weights: {class_weights.cpu().numpy().round(4)}")
            else:
                logging.warning(f"Missing classes in y_train (counts: {counts}). Using unweighted loss or default FocalLoss behavior.")
        
        X_train_seq, y_train_seq, dates_train_seq = create_sequences(X_train_np, y_train, args.sequence_length, dates_train)
        X_val_seq, y_val_seq, dates_val_seq = create_sequences(X_val_np, y_val, args.sequence_length, dates_val)
        X_test_seq, y_test_seq, dates_test_seq = create_sequences(X_test_np, y_test, args.sequence_length, dates_test)
        
        # Aggressively free memory
        del X_train_np, y_train, X_val_np, y_val, dates_train, dates_val
        del X_test_np, y_test, dates_test
        gc.collect()

        if X_train_seq.size == 0:
            logging.error("Empty training sequences. Cannot proceed.")
            return

        train_dataset = ForexDataset(X_train_seq, y_train_seq, dates_train_seq)
        val_dataset = ForexDataset(X_val_seq, y_val_seq, dates_val_seq) if X_val_seq.size > 0 else None
        test_dataset = ForexDataset(X_test_seq, y_test_seq, dates_test_seq) if X_test_seq.size > 0 else None
        
        num_workers_val = 0 
        if device.type != 'mps' and args.num_workers > 0:
             num_workers_val = min(os.cpu_count() or 1, args.num_workers)
        
        g = torch.Generator(device='cpu'); g.manual_seed(args.random_state) # Generator for DataLoader should be on CPU
        use_persistent = num_workers_val > 0 and current_torch_version >= version.parse('1.8.0')
        
        common_loader_args = {
            'pin_memory': device.type == 'cuda',
            'num_workers': num_workers_val,
            'persistent_workers': use_persistent if num_workers_val > 0 else False, 
            'prefetch_factor': 2 if num_workers_val > 0 else None 
        }
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, generator=g, collate_fn=custom_collate_fn, **common_loader_args)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn, **common_loader_args) if val_dataset and len(val_dataset) > 0 else None
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn, **common_loader_args) if test_dataset and len(test_dataset) > 0 else None


        model = GTN_v10_4(numerical_dim=numerical_dim, binary_dim=binary_dim,
                        seq_len_T=args.sequence_length, d_model=args.d_model, n_heads=args.n_heads,
                        num_layers=args.num_layers, d_ff=args.d_ff, num_classes=num_classes, dropout=args.dropout,
                        pe_max_len_default=args.pe_max_len_default).to(device)
        
        if device.type == 'mps':
            torch.backends.mps.enable_metal_shader_cache = True 
        
        if args.compile_model and hasattr(torch, 'compile') and device.type in ['cuda', 'mps'] and current_torch_version >= version.parse('2.0.0'):
            try:
                backend = "aot_eager" if device.type == 'mps' else "reduce-overhead"
                model = torch.compile(model, backend=backend)
                logging.info(f"GTN Model compiled for {device.type} using backend '{backend}'.")
            except Exception as e:
                logging.warning(f"torch.compile() failed for {device.type}: {e}", exc_info=True)
        # Gradient checkpointing removed
        
        # Initialize the base criterion first
        base_criterion_instance = FocalLoss(
            gamma=args.focal_gamma,
            weight=class_weights, 
            reduction='mean'
        )
        logging.info(f"Base FocalLoss initialized with gamma={args.focal_gamma}. Class weights: {'Applied' if class_weights is not None else 'Not applied/calculated'}")

        criterion = base_criterion_instance # Default to FocalLoss

        if args.use_cost_sensitive_loss:
            criterion = CostSensitiveRegularizedLoss(base_loss=base_criterion_instance, args=args)
            logging.info(f"CostSensitiveRegularizedLoss is enabled, wrapping {base_criterion_instance.__class__.__name__}.")
            logging.info(f"CSRL Proxies: Volatility={args.cost_volatility_proxy}, Duration={args.cost_pos_duration_proxy}, Direction={args.cost_market_direction_proxy}")
            logging.info(f"CSRL Lambda: Initial={args.cost_initial_lambda}, Max={args.cost_max_lambda}, Min={args.cost_min_lambda}, Patience={args.cost_lambda_patience}")
        else:
            pass 
        
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, amsgrad=True)
        optimizer.zero_grad(set_to_none=True)

        steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps) if len(train_loader)>0 and args.grad_accum_steps>0 else 0
        total_effective_steps = steps_per_epoch * args.epochs
        warmup_steps = int(total_effective_steps * args.warmup_ratio) if total_effective_steps > 0 else 0
        scheduler = None

        if args.lr_scheduler_type == 'cosine':
            if args.use_warmup and warmup_steps > 0 and total_effective_steps > 0:
                scheduler = get_lr_scheduler(optimizer, warmup_steps, total_effective_steps)
                logging.info(f"Using Cosine LR scheduler with {warmup_steps} warmup steps over {total_effective_steps} total effective steps.")
            else:
                scheduler = get_lr_scheduler(optimizer, 0, total_effective_steps if total_effective_steps > 0 else args.epochs * len(train_loader)) 
                logging.info(f"Using Cosine LR scheduler (no explicit warmup) over {total_effective_steps if total_effective_steps > 0 else args.epochs * len(train_loader)} total effective steps.")
        elif args.lr_scheduler_type == 'plateau':
            if val_loader:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='max', 
                    factor=args.lr_factor,
                    patience=args.lr_patience,
                    threshold=args.lr_threshold,
                    threshold_mode=args.lr_threshold_mode
                )
                logging.info(f"Using ReduceLROnPlateau scheduler, monitoring AvgBuySellPrec.")
            else:
                logging.warning("ReduceLROnPlateau scheduler selected, but no validation loader is available. Training will proceed without LR scheduling unless a different scheduler is chosen.")
        elif args.lr_scheduler_type == 'onecycle':
            if total_effective_steps > 0:
                scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=args.max_lr, 
                    total_steps=total_effective_steps,
                    pct_start=args.warmup_ratio, 
                    anneal_strategy='cos',
                    div_factor=50, 
                    final_div_factor=1e4 
                )
                logging.info(f"Using OneCycleLR scheduler with max_lr={args.max_lr}, total_steps={total_effective_steps}, pct_start={args.warmup_ratio}.")
            else:
                logging.warning("OneCycleLR scheduler selected, but total_effective_steps is zero. Training will proceed without LR scheduling.")
        
        if scheduler is None and val_loader and args.lr_scheduler_type != 'plateau': 
             logging.warning(f"Scheduler '{args.lr_scheduler_type}' could not be initialized. Falling back to ReduceLROnPlateau as val_loader is available.")
             scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max', 
                factor=args.lr_factor,
                patience=args.lr_patience,
                threshold=args.lr_threshold,
                threshold_mode=args.lr_threshold_mode
            )
             logging.info(f"Using Fallback ReduceLROnPlateau scheduler, monitoring AvgBuySellPrec.")
        elif scheduler is None:
            logging.warning(f"No LR scheduler active based on current configuration ('{args.lr_scheduler_type}', use_warmup: {args.use_warmup}, val_loader: {val_loader is not None}). Training with constant LR: {args.learning_rate}")


        run_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(args.output_dir) / f"gtn_forex_run_{run_stamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, 
                            args.epochs, args.early_stopping_patience, output_dir, args.grad_accum_steps, args, label_encoder)

        joblib.dump(scaler, output_dir / "scaler.joblib")
        joblib.dump(label_encoder, output_dir / "label_encoder.joblib")
        logging.info(f"Scaler and LabelEncoder saved to {output_dir.resolve()}")

        best_model_checkpoint_path = output_dir / "model_best.pt"
        if os.path.exists(best_model_checkpoint_path):
            logging.info(f"Evaluating the best saved model from: {best_model_checkpoint_path}")
            loaded_best_model, _, loaded_le, loaded_args_ckpt = load_trained_model(best_model_checkpoint_path, device=device) 
            if loaded_best_model and loaded_le:
                eval_criterion = FocalLoss(
                    gamma=loaded_args_ckpt.focal_gamma if hasattr(loaded_args_ckpt, 'focal_gamma') else args.focal_gamma,
                    weight=class_weights, 
                    reduction='mean'
                )
                if hasattr(loaded_args_ckpt, 'use_cost_sensitive_loss') and loaded_args_ckpt.use_cost_sensitive_loss:
                    base_eval_criterion = eval_criterion
                    eval_criterion = CostSensitiveRegularizedLoss(base_loss=base_eval_criterion, args=loaded_args_ckpt) 
                    logging.info(f"BEST MODEL EVAL: CostSensitiveRegularizedLoss enabled, wrapping {base_eval_criterion.__class__.__name__}.")

                if val_loader:
                    val_final_metrics = evaluate_model_enhanced(loaded_best_model, val_loader, eval_criterion, device, list(loaded_le.classes_), "BEST Validation", 
                                                                 args=loaded_args_ckpt, 
                                                                 current_lambda_for_eval=loaded_args_ckpt.cost_initial_lambda if hasattr(loaded_args_ckpt, 'use_cost_sensitive_loss') and loaded_args_ckpt.use_cost_sensitive_loss else 0.0)
                    if val_final_metrics['cm'].size > 0: 
                        plot_confusion_matrix(val_final_metrics['cm'], list(loaded_le.classes_), output_dir / "confusion_matrix_best_val.png")
                    logging.info(f"BEST MODEL Validation Report:\n{val_final_metrics['report']}")
                
                if test_loader:
                    test_final_metrics = evaluate_model_enhanced(loaded_best_model, test_loader, eval_criterion, device, list(loaded_le.classes_), "BEST Test", 
                                                                 args=loaded_args_ckpt, 
                                                                 current_lambda_for_eval=loaded_args_ckpt.cost_initial_lambda if hasattr(loaded_args_ckpt, 'use_cost_sensitive_loss') and loaded_args_ckpt.use_cost_sensitive_loss else 0.0)
                    if test_final_metrics['cm'].size > 0: 
                        plot_confusion_matrix(test_final_metrics['cm'], list(loaded_le.classes_), output_dir / "confusion_matrix_best_test.png")
                    logging.info(f"BEST MODEL Test Report:\n{test_final_metrics['report']}")

                    verify_saved_model(
                        model_path=best_model_checkpoint_path,
                        data_loader=test_loader, 
                        expected_primary_metric=test_final_metrics['avg_buy_sell_precision'], 
                        primary_metric_name="avg_buy_sell_precision",
                        device=device,
                        criterion_for_eval=eval_criterion, 
                        args_for_eval=loaded_args_ckpt, 
                        lambda_for_eval=loaded_args_ckpt.cost_initial_lambda if hasattr(loaded_args_ckpt, 'use_cost_sensitive_loss') and loaded_args_ckpt.use_cost_sensitive_loss else 0.0
                    )
            else:
                logging.error("Failed to load the best model for final evaluation.")
        else:
            logging.warning("No best model checkpoint found to evaluate after training.")

        if os.path.exists(best_model_checkpoint_path) and test_loader and len(test_loader.dataset) > 0:
            logging.info("Attempting to visualize attention for the best model...")
            loaded_model_for_attn, _, loaded_le_for_attn, loaded_args_for_attn = load_trained_model(best_model_checkpoint_path, device=device)
            if loaded_model_for_attn and loaded_le_for_attn and loaded_args_for_attn:
                try:
                    sample_batch = next(iter(test_loader))
                    sample_input_tensor = sample_batch[0][0].unsqueeze(0) 
                    
                    num_step_encoder_layers = loaded_args_for_attn.num_layers
                    last_step_layer_idx = max(0, num_step_encoder_layers - 1) 
                    
                    plot_attention_weights(
                        model=loaded_model_for_attn,
                        sample_input=sample_input_tensor,
                        layer_index=last_step_layer_idx, 
                        output_dir=output_dir,
                        filename_prefix="best_model_test_sample_attention",
                        encoder_type='step_wise' 
                    )
                except StopIteration:
                    logging.warning("Test loader was empty, cannot get sample for attention visualization.")
                except Exception as e:
                    logging.error(f"Error during attention visualization: {e}", exc_info=True)
            else:
                logging.warning("Could not load best model to visualize attention.")

        # --- v10 SHAP Analysis ---
        if os.path.exists(best_model_checkpoint_path) and test_loader and len(test_loader.dataset) > 0:
            logging.info("--- Starting SHAP Analysis on Best Model ---")
            loaded_model_for_shap, _, loaded_le_for_shap, loaded_args_for_shap = load_trained_model(best_model_checkpoint_path, device=device)
            if loaded_model_for_shap:
                analyze_model_with_shap(
                    model=loaded_model_for_shap,
                    data_loader=test_loader,
                    feature_names=loaded_args_for_shap.all_features,
                    label_encoder=loaded_le_for_shap,
                    output_dir=output_dir,
                    device=device
                )
            else:
                logging.warning("Could not load best model for SHAP analysis.")

        logging.info(f"All results and logs saved in: {output_dir.resolve()}")

    except Exception as e:
        logging.error(f"Training failed: {e}", exc_info=True)
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def custom_collate_fn(batch):
    features = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    dates = [item[2] for item in batch] 

    collated_features = torch.utils.data.default_collate(features)
    collated_targets = torch.utils.data.default_collate(targets)

    return collated_features, collated_targets, dates

def verify_saved_model(model_path: Path, data_loader: DataLoader, 
                       expected_primary_metric: float, primary_metric_name: str,
                       device: torch.device, criterion_for_eval: nn.Module, 
                       args_for_eval: argparse.Namespace, 
                       lambda_for_eval: float,          
                       tolerance: float = 1e-4) -> bool: 
    logging.info(f"Verifying saved model: {model_path} against metric '{primary_metric_name}' (expected: {expected_primary_metric:.4f})")
    try:
        loaded_model, _, loaded_le, loaded_args_ckpt_verify = load_trained_model(model_path, device=device)
        if not loaded_model or not loaded_le: 
            logging.error("Verification failed: Could not load model or label encoder."); return False
        
        current_criterion_for_eval = criterion_for_eval
        if hasattr(args_for_eval, 'use_cost_sensitive_loss') and args_for_eval.use_cost_sensitive_loss:
            if not isinstance(criterion_for_eval, CostSensitiveRegularizedLoss):
                logging.warning("verify_saved_model: criterion_for_eval was not CSRL but args suggest it should be. Reconstructing.")
                base_crit = FocalLoss(
                    gamma=args_for_eval.focal_gamma,
                    weight=None, 
                    reduction='mean'
                )
                current_criterion_for_eval = CostSensitiveRegularizedLoss(base_loss=base_crit, args=args_for_eval)

        eval_metrics = evaluate_model_enhanced(
            loaded_model, 
            data_loader, 
            current_criterion_for_eval, 
            device, 
            list(loaded_le.classes_), 
            "Verification",
            args=args_for_eval, 
            current_lambda_for_eval=lambda_for_eval 
        )
        
        loaded_metric_value = eval_metrics.get(primary_metric_name)
        if loaded_metric_value is None:
            logging.error(f"Verification FAILED: Metric '{primary_metric_name}' not found in evaluation results of loaded model.")
            return False

        diff = abs(loaded_metric_value - expected_primary_metric)
        if diff > tolerance:
            logging.error(f"Verification FAILED: Metric '{primary_metric_name}' mismatch. Expected: {expected_primary_metric:.6f}, Got: {loaded_metric_value:.6f}. Diff: {diff:.6e}")
            return False
        else:
            logging.info(f"Verification PASSED: Metric '{primary_metric_name}' matches within tolerance ({expected_primary_metric:.6f} vs {loaded_metric_value:.6f}).")
            return True
    except Exception as e:
        logging.error(f"Verification FAILED with error: {e}", exc_info=True); return False

def benchmark_model_performance(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, device: torch.device,
                                class_names: List[str], epoch: int, output_dir: Path, args: argparse.Namespace,
                                adaptive_lambda_scheduler: Optional['AdaptiveLambda'] = None) -> Dict[str, Any]:
    logging.info(f"Benchmarking model at epoch {epoch}...")
    current_lambda_for_benchmark = 0.0
    if args.use_cost_sensitive_loss and adaptive_lambda_scheduler:
        current_lambda_for_benchmark = adaptive_lambda_scheduler.current_lambda
    
    metrics = evaluate_model_enhanced(model, data_loader, criterion, device, class_names, f"Benchmark-E{epoch}", 
                                                                args=args, current_lambda_for_eval=current_lambda_for_benchmark)
    if metrics['cm'].size != 0:
        plot_confusion_matrix(metrics['cm'], class_names, output_dir / f"confusion_matrix_epoch_{epoch}.png")
    
    benchmark_hist_file = output_dir / "benchmark_run_history.csv"
    df_benchmark = pd.DataFrame([{
        'epoch': epoch, 'loss': metrics['loss'], 'accuracy': metrics['accuracy'], 
        'f1_macro': metrics['f1_macro'], 'avg_buy_sell_precision': metrics['avg_buy_sell_precision'],
        'buy_precision': metrics['buy_precision'], 'sell_precision': metrics['sell_precision']
    }])
    if not benchmark_hist_file.exists(): df_benchmark.to_csv(benchmark_hist_file, index=False)
    else: df_benchmark.to_csv(benchmark_hist_file, mode='a', header=False, index=False)
    return metrics

def analyze_model_with_shap(model: nn.Module, data_loader: DataLoader,
                            feature_names: List[str], label_encoder: LabelEncoder,
                            output_dir: Path, device: torch.device):
    """
    Performs SHAP analysis using KernelExplainer for robustness and generates a feature importance report.
    """
    if shap is None:
        logging.error("SHAP library not installed. Please install with 'pip install shap'. Skipping analysis.")
        return

    logging.info("--- Starting SHAP Analysis (KernelExplainer) on Best Model ---")
    if not data_loader or (hasattr(data_loader, 'dataset') and len(data_loader.dataset) == 0):
        logging.warning("SHAP analysis skipped: data_loader is empty.")
        return

    model.eval()
    model.to(device)

    # --- Data Collection ---
    # Collect a small, representative set of samples for the explainer
    MAX_BACKGROUND_SAMPLES = 50  # For summarizing the data distribution
    MAX_SAMPLES_TO_EXPLAIN = 20  # Samples we want to find SHAP values for

    background_samples_list = []
    samples_to_explain_list = []
    
    with torch.no_grad():
        for batch_data in data_loader:
            inputs, _, _ = batch_data
            if len(background_samples_list) == 0: # Collect one batch for background
                 background_samples_list.append(inputs.cpu())

            if len(samples_to_explain_list) == 0: # Collect one batch for explaining
                 samples_to_explain_list.append(inputs.cpu())
            
            if len(background_samples_list) > 0 and len(samples_to_explain_list) > 0:
                break
    
    if not background_samples_list:
        logging.warning("SHAP analysis skipped: could not collect background data from loader.")
        return

    background_samples = torch.cat(background_samples_list)[:MAX_BACKGROUND_SAMPLES]
    samples_to_explain = torch.cat(samples_to_explain_list)[:MAX_SAMPLES_TO_EXPLAIN] if samples_to_explain_list else torch.empty(0)
    
    if samples_to_explain.shape[0] == 0:
        logging.warning("SHAP analysis skipped: could not collect samples to explain.")
        return
        
    logging.info(f"Collected {background_samples.shape[0]} background and {samples_to_explain.shape[0]} explanation samples.")

    # --- SHAP KernelExplainer Setup ---
    def predict_proba_wrapper(x_numpy: np.ndarray) -> np.ndarray:
        """Wrapper to make the PyTorch model compatible with SHAP KernelExplainer."""
        num_samples, seq_len, input_dim = x_numpy.shape[0], model.seq_len_T, model.input_dim_C
        x_reshaped = x_numpy.reshape(num_samples, seq_len, input_dim)
        x_tensor = torch.tensor(x_reshaped, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(x_tensor)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    background_2d = background_samples.numpy().reshape(background_samples.shape[0], -1)
    samples_to_explain_2d = samples_to_explain.numpy().reshape(samples_to_explain.shape[0], -1)
    
    background_summary = shap.kmeans(background_2d, 25).data
    logging.info(f"Created a SHAP background summary of shape {background_summary.shape} using K-Means.")
    
    logging.info("Creating SHAP KernelExplainer...")
    explainer = shap.KernelExplainer(predict_proba_wrapper, background_summary)
    
    logging.info(f"Calculating SHAP values for {samples_to_explain_2d.shape[0]} samples. This may take a while...")
    shap_values_list_2d = explainer.shap_values(samples_to_explain_2d, nsamples='auto')
    
    # --- Reporting ---
    seq_len, input_dim = model.seq_len_T, model.input_dim_C
    class_names = list(label_encoder.classes_)
    shap_values_list_3d = [arr.reshape(-1, seq_len, input_dim) for arr in shap_values_list_2d]

    for i, class_name in enumerate(class_names):
        shap_values_agg_time = np.abs(shap_values_list_3d[i]).sum(axis=1)
        feature_values_agg_time = samples_to_explain.numpy().mean(axis=1)
        
        plt.figure(figsize=(12, max(8, len(feature_names) * 0.3)))
        shap.summary_plot(shap_values_agg_time, features=feature_values_agg_time, feature_names=feature_names, show=False, plot_type='bar')
        plt.title(f"Aggregated SHAP Feature Importance for class: {class_name}")
        plt.tight_layout()
        save_path = output_dir / f"shap_summary_{class_name}.png"
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Saved SHAP summary plot for '{class_name}' to {save_path}")

    # --- Generate Text Report ---
    logging.info("Generating feature importance report...")
    valid_shap_arrays = [np.abs(arr) for arr in shap_values_list_3d if arr is not None and isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[0] > 0]
    
    if not valid_shap_arrays:
        report_str = ("# SHAP Feature Importance Report\n\nNo valid SHAP values were computed.\n")
    else:
        mega_shap_array = np.concatenate(valid_shap_arrays, axis=0)
        global_importance = mega_shap_array.mean(axis=(0, 1))
        importance_df = pd.DataFrame({'feature': feature_names, 'importance': global_importance})
        importance_df = importance_df.sort_values(by='importance', ascending=False).reset_index(drop=True)
        pruning_count = int(len(importance_df) * 0.20)
        features_to_prune = importance_df.tail(pruning_count)
        
        report_str = ("# SHAP Feature Importance and Pruning Report\n\n"
                      "## Ranked Feature Importance (Most to Least Important)\n\n")
        for idx, row in importance_df.iterrows():
            report_str += f"{idx + 1:2d}. {row['feature']:<35} | Importance: {row['importance']:.6f}\n"
        
        report_str += (f"\n## Recommended Features to Exclude (Bottom {pruning_count / len(importance_df) * 100:.0f}%)\n\n"
                       "[\n")
        for feature in features_to_prune['feature']:
            report_str += f"    '{feature}',\n"
        report_str += "]\n"

    report_path = output_dir / "feature_importance_report.txt"
    with open(report_path, "w") as f:
        f.write(report_str)
    logging.info(f"Feature importance and pruning report saved to {report_path}")

def load_trained_model(checkpoint_path: Union[str, Path], device: Union[str, torch.device] = 'cpu') -> Tuple[Optional[GTN_v10_4], Optional[RobustScaler], Optional[LabelEncoder], Optional[argparse.Namespace]]:
    try:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            logging.error(f"Checkpoint file not found: {checkpoint_path}")
            return None, None, None, None
            
        current_device = torch.device(device) 
        
        ckpt = torch.load(checkpoint_path, map_location=current_device, weights_only=False)
        
        # Check for v10.4 specific keys
        required_keys = ['model_state_dict', 'args', 'numerical_dim', 'binary_dim', 'seq_len_T', 'num_classes', 'label_encoder_classes', 'hour_emb_dim', 'minute_emb_dim', 'day_emb_dim']
        missing_keys = [key for key in required_keys if key not in ckpt]
        if missing_keys:
            logging.error(f"Checkpoint missing required keys for v10.4 model: {missing_keys}. Found: {list(ckpt.keys())}")
            return None, None, None, None

        args_from_ckpt_dict = ckpt.get('args')
        if not args_from_ckpt_dict:
            logging.error("Args not found in checkpoint.")
            return None, None, None, None
            
        args_ns = argparse.Namespace(**args_from_ckpt_dict)
        
        dropout_val = args_ns.dropout if hasattr(args_ns, 'dropout') else 0.0 

        model = GTN_v10_4(
            numerical_dim=ckpt['numerical_dim'], 
            binary_dim=ckpt['binary_dim'],
            seq_len_T=ckpt['seq_len_T'],
            d_model=args_ns.d_model, 
            n_heads=args_ns.n_heads, 
            num_layers=args_ns.num_layers,
            d_ff=args_ns.d_ff, 
            num_classes=ckpt['num_classes'], 
            dropout=dropout_val,
            pe_max_len_default=args_ns.pe_max_len_default,
            hour_emb_dim=ckpt['hour_emb_dim'],
            minute_emb_dim=ckpt['minute_emb_dim'],
            day_emb_dim=ckpt['day_emb_dim']
        ).to(current_device) 
        
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        scaler_path = checkpoint_path.parent / 'scaler.joblib'
        label_encoder_path = checkpoint_path.parent / 'label_encoder.joblib'
        
        scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        label_encoder_obj = joblib.load(label_encoder_path) if label_encoder_path.exists() else None 
        
        if not scaler: logging.warning(f"Scaler not found at {scaler_path}")
        if not label_encoder_obj:
            logging.warning(f"LabelEncoder file not found at {label_encoder_path}. Reconstructing from checkpoint.")
            if ckpt.get('label_encoder_classes'):
                label_encoder_obj = LabelEncoder()
                label_encoder_obj.classes_ = np.array(ckpt['label_encoder_classes'])
            else:
                logging.error("Cannot reconstruct LabelEncoder: 'label_encoder_classes' not in checkpoint.")
                return model, scaler, None, args_ns 

        return model, scaler, label_encoder_obj, args_ns
        
    except Exception as e:
        logging.error(f"Error loading model from {checkpoint_path}: {e}", exc_info=True)
        return None, None, None, None

def generate_signals(model: nn.Module, inputs: torch.Tensor, threshold: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits = model(inputs.to(next(model.parameters()).device))
        probs = torch.softmax(logits, dim=-1)
        max_probs, preds = torch.max(probs, dim=1)
        mask = max_probs >= threshold
        return preds[mask].cpu().numpy(), max_probs[mask].cpu().numpy()

def optimize_for_mps():
    """Configure PyTorch for optimal MPS performance."""
    if torch.backends.mps.is_available():
        # This is often enabled by default in recent versions, but explicit is good.
        torch.backends.mps.enable_metal_shader_cache = True
        logging.info("MPS optimizations (shader cache) enabled.")

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    def __init__(self, gamma: float = 2.5, weight: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(inputs, dim=1)
        p = torch.exp(logp)
        logpt = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        loss = -((1 - pt) ** self.gamma) * logpt 

        if self.weight is not None:
            w = self.weight.to(targets.device)
            if targets.max() >= len(w) or targets.min() < 0 :
                 logging.error(f"Target indices out of bounds for weights. Targets: {targets.unique()}, Num_weights: {len(w)}")
            else:
                at = w[targets]
                loss = loss * at 
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# --- Custom Loss Components ---

class AdaptiveLambda:
    """
    Dynamically adjusts the lambda hyperparameter for Cost-Sensitive Regularization.
    It intelligently increases or decreases lambda based on performance trends to escape
    local optima and avoid oscillation.
    """
    def __init__(self, initial_lambda: float = 5.0, max_lambda: float = 20.0, min_lambda: float = 0.5,
                 patience: int = 2, improvement_threshold: float = 0.001,
                 increase_factor: float = 1.25, decrease_factor: float = 0.8):

        self.current_lambda = initial_lambda
        self.max_lambda = max_lambda
        self.min_lambda = min_lambda
        self.patience = patience
        self.improvement_threshold = improvement_threshold
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor

        self.best_metric = -float('inf')
        self.best_lambda = initial_lambda
        self.epochs_no_improve = 0
        
        # State to track adjustment direction and avoid oscillation
        self.last_adjustment_direction = None # Can be 'increase' or 'decrease'
        self.metric_before_adjustment = -float('inf')

        logging.info(
            f"Initialized AdaptiveLambda: initial_lambda={initial_lambda}, patience={patience}, "
            f"threshold={improvement_threshold}, inc_factor={increase_factor}, dec_factor={decrease_factor}"
        )

    def update(self, val_metric: float) -> float:
        """
        Updates lambda based on validation metric performance.
        - If metric improves, reset patience.
        - If metric stagnates, intelligently adjust lambda to escape plateau.
        """
        is_beating_best = val_metric > self.best_metric + self.improvement_threshold

        if is_beating_best:
            logging.info(f"[AdaptiveLambda] New best metric {val_metric:.4f} > {self.best_metric:.4f}. Lambda remains {self.current_lambda:.2f}.")
            self.best_metric = val_metric
            self.best_lambda = self.current_lambda
            self.epochs_no_improve = 0
            self.last_adjustment_direction = None # Clear direction on new best
        else:
            self.epochs_no_improve += 1
            logging.info(
                f"[AdaptiveLambda] Metric {val_metric:.4f} did not improve over best {self.best_metric:.4f}. "
                f"Patience: {self.epochs_no_improve}/{self.patience}. Lambda: {self.current_lambda:.2f}."
            )

            if self.epochs_no_improve >= self.patience:
                # Patience is exhausted, time to adjust lambda.

                # TREND-AWARE LOGIC: Check if the last adjustment showed *any* positive momentum.
                was_last_adjustment_helpful = self.last_adjustment_direction and val_metric > self.metric_before_adjustment

                if was_last_adjustment_helpful:
                    logging.warning(
                        f"[AdaptiveLambda] Last adjustment ({self.last_adjustment_direction}) showed positive trend "
                        f"(from {self.metric_before_adjustment:.4f} to {val_metric:.4f}). Continuing in the same direction."
                    )
                    # Continue the successful trend from the *current* lambda value
                    if self.last_adjustment_direction == 'increase':
                        self._increase_lambda(base=self.current_lambda, aggressive=False)
                    else: # last was 'decrease'
                        self._decrease_lambda(base=self.current_lambda, aggressive=False)
                else:
                    # The last move was not helpful or there was no last move.
                    # Revert to the aggressive reversal strategy from the best known point.
                    logging.warning(
                        f"[AdaptiveLambda] Last adjustment ({self.last_adjustment_direction}) was not helpful "
                        f"(from {self.metric_before_adjustment:.4f} to {val_metric:.4f}). Reversing aggressively from BEST lambda ({self.best_lambda:.2f})."
                    )
                    if self.last_adjustment_direction == 'increase':
                        # Last attempt was to increase, it failed, so now decrease from best.
                        self._decrease_lambda(base=self.best_lambda, aggressive=True)
                    else:
                        # Last attempt was to decrease (or was None), so now increase from best.
                        self._increase_lambda(base=self.best_lambda, aggressive=True)

                # Reset patience and record the metric *before* this new adjustment takes effect.
                self.epochs_no_improve = 0
                self.metric_before_adjustment = val_metric
        
        return self.current_lambda

    def _increase_lambda(self, base: float, aggressive: bool = False):
        factor = self.increase_factor ** 1.5 if aggressive else self.increase_factor
        new_lambda = base * factor
        self.current_lambda = min(self.max_lambda, new_lambda)
        self.last_adjustment_direction = 'increase'
        logging.warning(
            f"[AdaptiveLambda] {'Aggressively ' if aggressive else ''}Increasing lambda to {self.current_lambda:.2f} from base {base:.2f}."
        )

    def _decrease_lambda(self, base: float, aggressive: bool = False):
        # A more symmetric aggressive decrease using the inverse of the squared increase factor.
        factor = 1 / (self.increase_factor ** 1.5) if aggressive else self.decrease_factor
        new_lambda = base * factor
        self.current_lambda = max(self.min_lambda, new_lambda)
        self.last_adjustment_direction = 'decrease'
        logging.warning(
            f"[AdaptiveLambda] {'Aggressively ' if aggressive else ''}Decreasing lambda to {self.current_lambda:.2f} from base {base:.2f}."
        )

def create_cost_matrix(
    volatility_proxy: float, 
    pos_duration_proxy: int,
    device: torch.device,
    args: argparse.Namespace # Added to access base cost args
    ) -> torch.Tensor:
    """
    Creates a dynamic cost matrix.
    Args:
        volatility_proxy: Proxy for normalized volatility (e.g., 0.0 to 1.0). 
                          A higher value might indicate higher perceived risk.
        pos_duration_proxy: Proxy for position duration in bars.
        device: The torch device to place the tensor on.
        args: The argparse namespace to access base cost parameters.
    Returns:
        A (3,3) cost matrix tensor for [buy, keep, sell].
    """
    base_costs = torch.tensor([
        [0.0, args.cost_base_buy_keep, args.cost_base_buy_sell],       # True Buy
        [args.cost_base_keep_buy_sell, 0.0, args.cost_base_keep_buy_sell], # True Keep
        [args.cost_base_sell_buy, args.cost_base_sell_keep, 0.0]        # True Sell
    ], dtype=torch.float32, device=device)

    volatility_factor_float = 1.0 + 0.2 * (2 * volatility_proxy - 1.0) 
    volatility_factor_tensor = torch.tensor(volatility_factor_float, dtype=torch.float32, device=device)
    clamped_volatility_factor = torch.clamp(volatility_factor_tensor, 0.5, 1.5)
    
    time_decay_arg = -float(pos_duration_proxy) / 20.0
    time_decay_factor = torch.exp(torch.tensor(time_decay_arg, dtype=torch.float32, device=device)) 

    dynamic_costs = base_costs * clamped_volatility_factor * torch.clamp(time_decay_factor, 0.5, 1.0)
    return dynamic_costs

def cost_sensitive_regularization(
    probs: torch.Tensor, 
    targets: torch.Tensor,
    cost_matrix: torch.Tensor,
    market_direction_proxy: int
    ) -> torch.Tensor:
    """
    Calculates the cost-sensitive regularization term.
    Args:
        probs: Predicted probabilities (batch_size, num_classes).
        targets: True target labels (batch_size,).
        cost_matrix: The (num_classes, num_classes) cost matrix.
        market_direction_proxy: Proxy for market direction (e.g., 1 for up, -1 for down, 0 for neutral).
    Returns:
        Scalar tensor for the mean regularization term for the batch.
    """
    sample_specific_costs = cost_matrix[targets] 
    expected_cost_per_sample = torch.sum(probs * sample_specific_costs, dim=1)

    directional_penalty = torch.ones_like(expected_cost_per_sample)
    if market_direction_proxy == 1: 
        directional_penalty += 0.1 * (targets != 2).float() 
    elif market_direction_proxy == -1: 
        directional_penalty += 0.1 * (targets != 0).float()
        
    regularization_term = torch.mean(expected_cost_per_sample * directional_penalty)
    return regularization_term

class CostSensitiveRegularizedLoss(nn.Module):
    """
    Combines a base loss (e.g., FocalLoss) with a cost-sensitive regularization term.
    Lambda for the regularization term is passed in the forward method.
    """
    def __init__(self, base_loss: nn.Module, 
                 args: argparse.Namespace 
                 ):
        super().__init__()
        self.base_loss = base_loss
        self.args = args
        logging.info(f"Initialized CostSensitiveRegularizedLoss with base: {base_loss.__class__.__name__}")
        logging.info(f"CSRL using proxy volatility: {args.cost_volatility_proxy}, duration: {args.cost_pos_duration_proxy}, direction: {args.cost_market_direction_proxy}")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, current_lambda: float) -> torch.Tensor:
        base_loss_value = self.base_loss(inputs, targets)
        
        current_cost_matrix = create_cost_matrix(
            volatility_proxy=self.args.cost_volatility_proxy,
            pos_duration_proxy=self.args.cost_pos_duration_proxy,
            device=inputs.device,
            args=self.args 
        )
        
        probs = F.softmax(inputs, dim=1)
        
        reg_term = cost_sensitive_regularization(
            probs,
            targets,
            current_cost_matrix,
            market_direction_proxy=self.args.cost_market_direction_proxy
        )
        
        total_loss = base_loss_value + current_lambda * reg_term
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logging.warning(f"NaN/Inf loss detected in CSRL! Base: {base_loss_value.item()}, Reg: {reg_term.item()}, Lambda: {current_lambda}")
            return base_loss_value 

        return total_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gated Transformer Network (GTN) for Time Series Classification - v10.4 Advanced")
    
    # Data paths
    parser.add_argument('--train_file', default='new_signal_4h_results/train_4hr_signal.csv',type=str, help="Path to training data CSV")
    parser.add_argument('--test_file', default='new_signal_4h_results/test_4hr_signal.csv',type=str, help="Path to test data CSV")
    parser.add_argument('--output_dir', type=str, default="output_gtn", help="Base directory for run outputs.")
    parser.add_argument('--date_col', type=str, default='Date', help="Name of the date/timestamp column in input CSVs for daily analysis.")
    
    # Data & Preprocessing
    parser.add_argument('--target_col', type=str, default='signal', help="Target column name")
    parser.add_argument('--all_features', type=str, nargs='+', default=v10_4_all_features, help="All feature columns")
    parser.add_argument('--numerical_features', type=str, nargs='+', default=v10_4_numerical_for_scaling, help="Numerical features to scale")
    parser.add_argument('--binary_features', type=str, nargs='+', default=v10_4_binary_features, help="Binary features (not to be scaled)")
    parser.add_argument('--target_classes', type=str, nargs='+', default=['buy', 'keep', 'sell'], help="Expected target class names in order")
    parser.add_argument('--sequence_length', type=int, default=120, help="Sequence length T (lookback window)") 
    parser.add_argument('--val_size', type=float, default=0.1, help="Validation set proportion (temporal split)")
    parser.add_argument('--data_sample_frac', type=float, default=0.25, help="Fraction of training data. 0.01 is very low; 0.05-0.1 is better for stability if memory allows.")
    
    # HMM Regimes
    parser.add_argument('--n_hmm_regimes', type=int, default=3, help="Number of market regimes for HMM feature.")
    
    # GTN Model
    parser.add_argument('--d_model', type=int, default=256, help="Model dimension.")
    parser.add_argument('--n_heads', type=int, default=8, help="Number of attention heads.")
    parser.add_argument('--num_layers', type=int, default=3, help="Transformer layers per tower.")
    parser.add_argument('--d_ff', type=int, default=512, help="Feed-forward dimension.")
    parser.add_argument('--dropout', type=float, default=0.15, help="Dropout rate (set to 0 to disable regularization).")
    parser.add_argument('--pe_max_len_default', type=int, default=5000, help="Max length for Positional Encoding. Should be >= sequence_length.")
    
    # Training
    parser.add_argument('--epochs', type=int, default=50, help="Max training epochs.")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size.")
    parser.add_argument('--learning_rate', type=float, default=5e-05, help="Initial learning rate for AdamW, not the peak for OneCycleLR.")
    parser.add_argument('--max_lr', type=float, default=0.0002, help="Max learning rate for OneCycleLR scheduler.")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay for regularization.")
    parser.add_argument('--grad_accum_steps', type=int, default=2, help="Gradient accumulation.")
    parser.add_argument('--early_stopping_patience', type=int, default=10, help="Patience for early stopping on val_composite_score.")
    parser.add_argument('--use_warmup', action='store_true', default=True, help="Use LR warmup for 'cosine' scheduler. For 'onecycle', warmup_ratio controls pct_start.")
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help="Warmup step ratio for 'cosine' or pct_start for 'onecycle'.")
    parser.add_argument('--lr_patience', type=int, default=2, help="ReduceLROnPlateau patience.") 
    parser.add_argument('--lr_factor', type=float, default=0.1, help="LR reduction factor for ReduceLROnPlateau scheduler.")
    parser.add_argument('--lr_threshold', type=float, default=0.001, help="Threshold for measuring plateau in ReduceLROnPlateau scheduler.")
    parser.add_argument('--lr_threshold_mode', type=str, choices=['rel','abs'], default='abs', help="Threshold mode for ReduceLROnPlateau scheduler (rel or abs).")
    parser.add_argument('--lr_scheduler_type', type=str, default='onecycle', choices=['cosine', 'plateau', 'onecycle'], help="Type of LR scheduler to use.")
    
    # System & Performance
    parser.add_argument('--num_workers', type=int, default=0, help="DataLoader workers. Set to 0 for MPS or if issues arise.")
    parser.add_argument('--compile_model', action='store_true', help="Use torch.compile().")
    parser.add_argument('--auto_batch_size', type=bool, default=True, help="Auto-adjust batch size based on GPU memory.")
    
    # Logging & Reproducibility
    parser.add_argument('--random_state', type=int, default=42, help="Global random seed.")
    parser.add_argument('--benchmark_interval', type=int, default=5, help="Benchmark every N epochs.")
    parser.add_argument('--signal_threshold', type=float, default=0.9, help='Signal generation threshold (used in generate_signals function).')
    parser.add_argument('--daily_threshold', type=float, default=0.9, help='Threshold for daily threshold stats calculation (0.0 to 1.0).') 
    
    # Loss function parameters
    parser.add_argument('--focal_gamma', type=float, default=1.5, help='Focal loss gamma parameter')
    
    # Cost-Sensitive Regularized Loss parameters
    parser.add_argument('--use_cost_sensitive_loss', default=True,action='store_true', help='Enable CostSensitiveRegularizedLoss.')
    parser.add_argument('--cost_initial_lambda', type=float, default=6.0, help='Initial lambda for CSRL.')
    parser.add_argument('--cost_max_lambda', type=float, default=20.0, help='Maximum lambda for CSRL.')
    parser.add_argument('--cost_min_lambda', type=float, default=1.0, help='Minimum lambda for CSRL.')
    parser.add_argument('--cost_lambda_patience', type=int, default=4, help='Patience for AdaptiveLambda scheduler (epochs).')
    parser.add_argument('--cost_lambda_improve_thresh', type=float, default=0.0005, help='Improvement threshold for AdaptiveLambda.')
    parser.add_argument('--cost_lambda_increase_factor', type=float, default=1.25, help='Factor to increase lambda by.')
    parser.add_argument('--cost_lambda_decrease_factor', type=float, default=0.8, help='Factor to decrease lambda by.')
    
    parser.add_argument('--cost_volatility_proxy', type=float, default=0.5, help='Proxy for normalized volatility (0.0 to 1.0) for dynamic cost matrix.')
    parser.add_argument('--cost_pos_duration_proxy', type=int, default=5, help='Proxy for position duration (bars) for dynamic cost matrix.')
    parser.add_argument('--cost_market_direction_proxy', type=int, default=0, choices=[-1, 0, 1], help='Proxy for market direction (-1: down, 0: neutral, 1: up) for cost regularization.')

    # Base costs for cost matrix (new)
    parser.add_argument('--cost_base_buy_keep', type=float, default=1.5, help='Base cost for True:Buy, Pred:Keep.')
    parser.add_argument('--cost_base_buy_sell', type=float, default=2.0, help='Base cost for True:Buy, Pred:Sell.')
    parser.add_argument('--cost_base_keep_buy_sell', type=float, default=3.0, help='Base cost for True:Keep, Pred:Buy/Sell.') 
    parser.add_argument('--cost_base_sell_buy', type=float, default=2.0, help='Base cost for True:Sell, Pred:Buy.')
    parser.add_argument('--cost_base_sell_keep', type=float, default=1.5, help='Base cost for True:Sell, Pred:Keep.')

    # Model Selection Metric Modifier
    parser.add_argument('--precision_deviation_penalty', type=float, default=0.25, help='Penalty factor for deviation between buy/sell precision in composite score.')

    if any(arg.startswith('-f') for arg in sys.argv):
        args = parser.parse_args(args=[])
        logging.info("Running in Jupyter/Colab cell, using default or in-notebook defined args.")
    else:
        args = parser.parse_args()
        logging.info("Running as standard script, parsing command-line arguments.")

    main(args)
    
