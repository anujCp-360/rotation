import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from typing import Dict, List, Tuple, Optional
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader, Dataset
import pywt
from scipy import signal
from scipy.stats import norm

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class AdaptiveNormalization(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveNormalization, self).__init__()
        self.layer_norm = nn.LayerNorm(num_features, eps=eps)
        self.instance_norm = nn.InstanceNorm1d(num_features, momentum=momentum)
        self.gate = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # Reshape for instance norm
        b, t, c = x.size()
        x_reshaped = x.transpose(1, 2)  # [B, C, T]
        
        # Apply normalizations
        x1 = self.layer_norm(x)
        x2 = self.instance_norm(x_reshaped).transpose(1, 2)  # Back to [B, T, C]
        
        # Adaptive combination
        gate = torch.sigmoid(self.gate)
        return gate * x1 + (1 - gate) * x2

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiScaleFourierLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_harmonics=8):
        super(MultiScaleFourierLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_harmonics = n_harmonics
        
        # Learnable frequencies
        self.frequencies = nn.Parameter(torch.randn(n_harmonics) * 0.01)
        self.amplitudes = nn.Parameter(torch.ones(n_harmonics) * 0.1)
        self.phases = nn.Parameter(torch.zeros(n_harmonics))
        
        self.projection = nn.Linear(input_dim + 2*n_harmonics, hidden_dim)
        
    def forward(self, x, time_idx):
        # time_idx should be normalized timestamps
        batch_size, seq_len, _ = x.shape
        time_idx = time_idx.view(batch_size, seq_len, 1)
        
        # Calculate Fourier features
        cos_features = []
        sin_features = []
        
        for i in range(self.n_harmonics):
            freq = torch.abs(self.frequencies[i]) + 0.1  # Ensure positive frequency
            amp = torch.abs(self.amplitudes[i])
            phase = self.phases[i]
            
            cos_features.append(amp * torch.cos(freq * time_idx + phase))
            sin_features.append(amp * torch.sin(freq * time_idx + phase))
        
        cos_features = torch.cat(cos_features, dim=-1)
        sin_features = torch.cat(sin_features, dim=-1)
        
        # Concatenate original features with Fourier features
        enhanced_features = torch.cat([x, cos_features, sin_features], dim=-1)
        
        return self.projection(enhanced_features)

class WaveletTransformLayer(nn.Module):
    def __init__(self, wavelet_type='db4', decomposition_level=3, learnable=True):
        super(WaveletTransformLayer, self).__init__()
        self.wavelet_type = wavelet_type
        self.decomposition_level = decomposition_level
        self.learnable = learnable
        
        # Get wavelet filters
        wavelet = pywt.Wavelet(wavelet_type)
        self.filter_length = len(wavelet.dec_lo)
        
        if learnable:
            # Initialize with wavelet filters but make them learnable
            self.dec_lo = nn.Parameter(torch.FloatTensor(wavelet.dec_lo))
            self.dec_hi = nn.Parameter(torch.FloatTensor(wavelet.dec_hi))
        else:
            self.register_buffer('dec_lo', torch.FloatTensor(wavelet.dec_lo))
            self.register_buffer('dec_hi', torch.FloatTensor(wavelet.dec_hi))
            
        # Projections for each decomposition level
        self.projections = nn.ModuleList(
            [nn.Linear(2**i, 2**i) for i in range(1, decomposition_level+1)]
        )
        
    def dwt(self, x):
        """Discrete wavelet transform using convolution"""
        batch_size, seq_len, features = x.shape
        x = x.reshape(batch_size * features, 1, seq_len)
        
        # Pad to ensure proper convolution
        pad_size = self.filter_length - 1
        x_padded = F.pad(x, (pad_size, 0))
        
        # Low-pass and high-pass filtering
        lo = F.conv1d(x_padded, self.dec_lo.view(1, 1, -1), stride=2)
        hi = F.conv1d(x_padded, self.dec_hi.view(1, 1, -1), stride=2)
        
        # Reshape back
        lo = lo.view(batch_size, features, -1).transpose(1, 2)
        hi = hi.view(batch_size, features, -1).transpose(1, 2)
        
        return lo, hi
    
    def forward(self, x):
        """Apply wavelet decomposition"""
        batch_size, seq_len, features = x.shape
        
        # Ensure sequence length is sufficient for decomposition
        min_length = 2**self.decomposition_level
        if seq_len < min_length:
            pad_len = min_length - seq_len
            x = F.pad(x, (0, 0, 0, pad_len))
            seq_len = min_length
        
        # Multi-level wavelet decomposition
        coeffs = []
        approx = x
        
        for i in range(self.decomposition_level):
            approx, detail = self.dwt(approx)
            # Apply learnable projection to each detail coefficient
            detail = self.projections[i](detail)
            coeffs.append(detail)
        
        coeffs.append(approx)
        
        # Concatenate all coefficients along feature dimension
        result = torch.cat(coeffs, dim=-1)
        
        return result

class TemporalConvBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, dilation=1, dropout=0.2):
        super(TemporalConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            input_dim, 
            hidden_dim, 
            kernel_size=kernel_size, 
            padding=(kernel_size-1)*dilation, 
            dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            hidden_dim, 
            hidden_dim, 
            kernel_size=kernel_size, 
            padding=(kernel_size-1)*dilation, 
            dilation=dilation
        )
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        
    def forward(self, x):
        # x shape: [batch, seq_len, channels]
        residual = self.projection(x)
        
        # Conv1D expects [batch, channels, seq_len]
        x_conv = x.transpose(1, 2)
        
        # First conv block
        out = self.conv1(x_conv)
        out = F.gelu(out)
        out = out.transpose(1, 2)  # Back to [batch, seq_len, channels]
        out = self.layer_norm1(out)
        out = self.dropout(out)
        
        # Second conv block
        out = out.transpose(1, 2)  # To [batch, channels, seq_len]
        out = self.conv2(out)
        out = F.gelu(out)
        out = out.transpose(1, 2)  # Back to [batch, seq_len, channels]
        out = self.layer_norm2(out)
        out = self.dropout(out)
        
        # Residual connection
        out = out + residual
        
        return out

class TemporalConvNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, num_layers=4, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        
        # Exponentially increasing dilation
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(
                TemporalConvBlock(
                    input_dim if i == 0 else hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class MultiHeadAttentionWithProbMask(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, sparse_attn=True):
        super(MultiHeadAttentionWithProbMask, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.sparse_attn = sparse_attn
        self.factor = 5  # Sparsity factor (c in Informer paper)
        
    def _prob_QK(self, Q, K, sample_k=None, n_top=None):
        """
        Compute attention with probabilistic sparse matrix
        Args:
            Q, K: query and key tensors
            sample_k: number of queries to sample
            n_top: number of top queries to preserve
        """
        B, L_Q, H, D = Q.shape
        _, L_K, _, _ = K.shape
        
        # Calculate query sparsity
        sample_k = L_Q if sample_k is None else min(sample_k, L_Q)
        n_top = n_top if n_top is not None else int(np.ceil(np.log(L_K)))
        
        # Reshape for matrix multiplication
        K_reshaped = K.transpose(1, 2).reshape(B, H, D, L_K)
        Q_reshaped = Q.transpose(1, 2).reshape(B, H, L_Q, D)
        
        # Q_K calculation
        Q_K = torch.matmul(Q_reshaped, K_reshaped)  # B, H, L_Q, L_K
        
        # Find top-k queries (most informative queries)
        if n_top < L_Q:
            M = Q_K.max(-1)[0] - torch.div(Q_K.sum(-1), L_K)
            M_top = M.topk(n_top, sorted=False)[1]
            Q_reduce = torch.zeros(B, H, n_top, D, device=Q.device)
            
            # Gather top queries
            for b in range(B):
                for h in range(H):
                    Q_reduce[b, h] = Q[b, M_top[b, h], h]
            
            # Recalculate Q_K with reduced queries
            Q_K_reduce = torch.matmul(Q_reduce, K_reshaped)
            
            # Get attention scores with reduced computation
            return Q_K_reduce
        else:
            return Q_K.transpose(1, 2)  # Return full attention (B, H, L_Q, L_K)
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        if self.sparse_attn:
            # Reshape for sparse attention calculation
            B, L_Q, D = query.shape
            _, L_K, _ = key.shape
            H = self.mha.num_heads
            
            # Reshape query, key for sparse attention
            q = query.view(B, L_Q, H, -1)
            k = key.view(B, L_K, H, -1)
            
            # Apply probabilistic attention masking
            sparse_attn = self._prob_QK(q, k, n_top=int(np.log(L_K)))
            
            # Apply attention with the sparse mask
            output, _ = self.mha(query, key, value, attn_mask=sparse_attn, 
                                key_padding_mask=key_padding_mask)
        else:
            # Standard multihead attention
            output, _ = self.mha(query, key, value, attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask)
        
        return output

class StackedBidirectionalLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.2):
        super(StackedBidirectionalLSTM, self).__init__()
        self.num_layers = len(hidden_dims)
        
        # Create LSTM layers with decreasing hidden sizes
        self.lstm_layers = nn.ModuleList()
        
        # Input layer
        self.lstm_layers.append(
            nn.LSTM(input_dim, hidden_dims[0], bidirectional=True, batch_first=True)
        )
        
        # Hidden layers
        for i in range(1, self.num_layers):
            self.lstm_layers.append(
                nn.LSTM(hidden_dims[i-1]*2, hidden_dims[i], bidirectional=True, batch_first=True)
            )
        
        # Layer normalization between LSTM layers
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hid_dim*2) for hid_dim in hidden_dims]
        )
        
        # Dropout layers
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout) for _ in range(self.num_layers)]
        )
        
    def forward(self, x):
        outputs = []
        current_input = x
        
        for i in range(self.num_layers):
            # Apply LSTM layer
            lstm_out, _ = self.lstm_layers[i](current_input)
            
            # Apply layer normalization
            normalized = self.layer_norms[i](lstm_out)
            
            # Apply dropout
            current_input = self.dropouts[i](normalized)
            
            # Store layer output
            outputs.append(current_input)
        
        return outputs[-1]  # Return final layer output

class TemporalMemoryNetwork(nn.Module):
    def __init__(self, input_dim, memory_dim, memory_size=128, topk=5):
        super(TemporalMemoryNetwork, self).__init__()
        self.memory_dim = memory_dim
        self.memory_size = memory_size
        self.topk = topk
        
        # Initialize memory with learnable patterns
        self.memory_keys = nn.Parameter(torch.randn(memory_size, memory_dim))
        self.memory_values = nn.Parameter(torch.randn(memory_size, memory_dim))
        
        # Input projection to memory dimension
        self.query_projection = nn.Linear(input_dim, memory_dim)
        
        # Output projection
        self.output_projection = nn.Linear(memory_dim, input_dim)
        
    def forward(self, query):
        """
        Args:
            query: Input tensor of shape [batch_size, seq_len, input_dim]
        Returns:
            Memory-augmented output of shape [batch_size, seq_len, input_dim]
        """
        batch_size, seq_len, _ = query.shape
        
        # Project input to memory dimension
        query_proj = self.query_projection(query)  # [B, T, memory_dim]
        
        # Reshape for batch matrix multiplication
        query_reshaped = query_proj.view(batch_size * seq_len, 1, self.memory_dim)
        memory_keys_expanded = self.memory_keys.unsqueeze(0).expand(
            batch_size * seq_len, self.memory_size, self.memory_dim
        )
        
        # Calculate attention scores
        attention = torch.bmm(query_reshaped, memory_keys_expanded.transpose(1, 2))
        attention = attention.squeeze(1)  # [B*T, memory_size]
        
        # Get top-k memory slots
        topk_weights, topk_indices = torch.topk(attention, self.topk, dim=1)
        topk_weights = F.softmax(topk_weights, dim=1)  # [B*T, topk]
        
        # Gather top-k memory values
        memory_values_expanded = self.memory_values.unsqueeze(0).expand(
            batch_size * seq_len, self.memory_size, self.memory_dim
        )
        
        # Gather and weight memory values
        batch_indices = torch.arange(batch_size * seq_len).unsqueeze(1).expand(-1, self.topk)
        top_memory_values = memory_values_expanded[batch_indices.flatten(), 
                                               topk_indices.flatten()].view(
            batch_size * seq_len, self.topk, self.memory_dim
        )
        
        weighted_values = top_memory_values * topk_weights.unsqueeze(-1)
        memory_output = weighted_values.sum(dim=1)  # [B*T, memory_dim]
        
        # Reshape and project back to input dimension
        memory_output = memory_output.view(batch_size, seq_len, self.memory_dim)
        output = self.output_projection(memory_output)
        
        # Residual connection
        return output + query

class NBeatsBlock(nn.Module):
    def __init__(self, input_dim, theta_dim, basis_functions, share_weights=False):
        super(NBeatsBlock, self).__init__()
        self.input_dim = input_dim
        self.theta_dim = theta_dim
        self.basis_functions = basis_functions
        self.share_weights = share_weights
        
        # Fully connected stack for backcast and forecast
        self.fc1 = nn.Linear(input_dim, 2*input_dim)
        self.fc2 = nn.Linear(2*input_dim, 4*input_dim)
        self.fc3 = nn.Linear(4*input_dim, 2*input_dim)
        self.fc4 = nn.Linear(2*input_dim, theta_dim)
        
        # Separate FC layers for backcast and forecast if not sharing weights
        if not share_weights:
            self.fc_forecast = nn.Linear(2*input_dim, theta_dim)
    
    def forward(self, x):
        # Forward pass through FC layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # Calculate theta for backcast (and forecast if sharing weights)
        theta = self.fc4(x)
        
        # Calculate separate theta for forecast if not sharing weights
        if self.share_weights:
            theta_b = theta_f = theta
        else:
            theta_b = theta
            theta_f = self.fc_forecast(x)
        
        # Apply basis functions
        backcast = self.basis_functions.backward(theta_b)
        forecast = self.basis_functions.forward(theta_f)
        
        return backcast, forecast

class TrendBasis:
    def __init__(self, degree, forecast_length, backcast_length):
        self.degree = degree
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        
        # Precompute polynomial basis
        self.backcast_basis = self._polynomial_basis(backcast_length)
        self.forecast_basis = self._polynomial_basis(forecast_length)
        
    def _polynomial_basis(self, length):
        # Generate powers of x for polynomial basis
        x = np.arange(length) / length
        powers = np.power(x.reshape(-1, 1), np.arange(self.degree + 1))
        return torch.tensor(powers, dtype=torch.float32)
    
    def forward(self, theta):
        """Compute forecast values using polynomial basis"""
        return torch.matmul(self.forecast_basis.to(theta.device), theta.unsqueeze(-1)).squeeze(-1)
    
    def backward(self, theta):
        """Compute backcast values using polynomial basis"""
        return torch.matmul(self.backcast_basis.to(theta.device), theta.unsqueeze(-1)).squeeze(-1)

class SeasonalBasis:
    def __init__(self, harmonics, forecast_length, backcast_length):
        self.harmonics = harmonics
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        
        # Precompute seasonal basis
        self.backcast_basis = self._fourier_basis(backcast_length)
        self.forecast_basis = self._fourier_basis(forecast_length)
    
    def _fourier_basis(self, length):
        # Generate Fourier basis with different frequencies
        x = np.arange(length) / length
        basis = np.zeros((length, 2 * self.harmonics))
        
        for i in range(self.harmonics):
            basis[:, 2*i] = np.cos(2 * np.pi * (i+1) * x)
            basis[:, 2*i+1] = np.sin(2 * np.pi * (i+1) * x)
            
        return torch.tensor(basis, dtype=torch.float32)
    
    def forward(self, theta):
        """Compute forecast using Fourier basis"""
        return torch.matmul(self.forecast_basis.to(theta.device), theta.unsqueeze(-1)).squeeze(-1)
    
    def backward(self, theta):
        """Compute backcast using Fourier basis"""
        return torch.matmul(self.backcast_basis.to(theta.device), theta.unsqueeze(-1)).squeeze(-1)

class NBeatsNetwork(nn.Module):
    def __init__(
        self, 
        input_dim,
        forecast_length, 
        backcast_length,
        stack_types=['trend', 'seasonality'],
        num_blocks=[3, 3],
        trend_degree=4,
        seasonality_harmonics=8
    ):
        super(NBeatsNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.stack_types = stack_types
        self.num_blocks = num_blocks
        
        # Initialize basis functions
        trend_basis = TrendBasis(trend_degree, forecast_length, backcast_length)
        seasonal_basis = SeasonalBasis(seasonality_harmonics, forecast_length, backcast_length)
        
        # Create stacks and blocks
        self.blocks = nn.ModuleList([])
        
        for stack_id, stack_type in enumerate(stack_types):
            for block_id in range(num_blocks[stack_id]):
                # Select appropriate basis function
                if stack_type == 'trend':
                    basis = trend_basis
                    theta_dim = trend_degree + 1
                elif stack_type == 'seasonality':
                    basis = seasonal_basis
                    theta_dim = 2 * seasonality_harmonics
                else:
                    raise ValueError(f"Unknown stack type: {stack_type}")
                
                # Add block to network
                self.blocks.append(
                    NBeatsBlock(
                        input_dim=input_dim,
                        theta_dim=theta_dim,
                        basis_functions=basis,
                        share_weights=(block_id == 0)  # Share weights in first block of each stack
                    )
                )
    
    def forward(self, x):
        """
        Args:
            x: Input of shape [batch_size, backcast_length, input_dim]
        Returns:
            forecast: Output of shape [batch_size, forecast_length]
        """
        # Get batch size
        batch_size = x.shape[0]
        
        # Initialize backcast and forecast
        backcast = x.clone()
        forecast = torch.zeros(
            (batch_size, self.forecast_length), dtype=torch.float32, device=x.device
        )
        
        # Apply each block sequentially
        for block in self.blocks:
            # Get block's backcast and forecast
            block_backcast, block_forecast = block(backcast)
            
            # Update backcast and forecast
            backcast = backcast - block_backcast
            forecast = forecast + block_forecast
        
        return forecast

class DeepARLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_samples=100):
        super(DeepARLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_samples = n_samples
        
        # LSTM for modeling temporal dependencies
        self.lstm = nn.LSTM(input_dim, hidden_dim,
