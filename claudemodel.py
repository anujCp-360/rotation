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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ------------------------------------------------------------------------------
# Multi-Source Input Layer
# ------------------------------------------------------------------------------
class InputLayer(nn.Module):
    def __init__(self, input_dim, calendar_dim, exo_dim, meta_dim, embedding_dim):
        super(InputLayer, self).__init__()
        self.primary_linear = nn.Linear(input_dim, embedding_dim)
        self.calendar_embedding = nn.Embedding(calendar_dim, embedding_dim)
        self.exo_linear = nn.Linear(exo_dim, embedding_dim)
        self.meta_linear = nn.Linear(meta_dim, embedding_dim)

    def forward(self, primary_time_series, calendar_features, exo_variables, historical_events, meta_features):
        primary_embedding = self.primary_linear(primary_time_series)
        calendar_embedding = self.calendar_embedding(calendar_features)
        exo_embedding = self.exo_linear(exo_variables)
        meta_embedding = self.meta_linear(meta_features)
        
        # Assuming historical_events are already embedded or processed
        return primary_embedding + calendar_embedding + exo_embedding + historical_events + meta_embedding

# ------------------------------------------------------------------------------
# Feature Processing
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# Hierarchical Time-Scale Processing
# ------------------------------------------------------------------------------
class MultiResolutionFeaturePyramid(nn.Module):
    def __init__(self, embedding_dim, num_levels=5):
        super(MultiResolutionFeaturePyramid, self).__init__()
        self.num_levels = num_levels
        self.feature_extractors = nn.ModuleList([
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=2**i, stride=2**i) for i in range(num_levels)
        ])

    def forward(self, x):
        feature_maps = [x]
        for i in range(self.num_levels):
            x = x.transpose(1, 2)  # Reshape for Conv1d
            feature = self.feature_extractors[i](x)
            feature = feature.transpose(1, 2)  # Transpose back
            feature_maps.append(feature)
        return feature_maps

class CrossScaleAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads=8):
        super(CrossScaleAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads)

    def forward(self, feature_maps):
        # Concatenate feature maps from different resolutions
        x = torch.cat(feature_maps, dim=1)
        # Apply cross-scale attention
        attn_output, _ = self.attention(x, x, x)
        return attn_output

# ------------------------------------------------------------------------------
# Advanced Temporal Encoding
# ------------------------------------------------------------------------------
class TemporalConvolutionalNetwork(nn.Module):
    def __init__(self, embedding_dim, num_layers=3, dilation_base=2):
        super(TemporalConvolutionalNetwork, self).__init__()
        self.tcn_layers = nn.ModuleList([
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, dilation=dilation_base**i, padding=dilation_base**i)
            for i in range(num_layers)
        ])

    def forward(self, x):
        x = x.transpose(1, 2)  # Reshape for Conv1d
        for layer in self.tcn_layers:
            x = F.relu(layer(x))
        x = x.transpose(1, 2)  # Transpose back
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads=16):
        super(MultiHeadSelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return attn_output

class InformerStyleSparseAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads=8, attention_factor=5):
        super(InformerStyleSparseAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads)
        self.attention_factor = attention_factor

    def forward(self, x):
        # Implement sparse attention mechanism (Informer-style)
        # This is a simplified version; the actual Informer attention is more complex
        seq_len = x.size(1)
        num_selected = min(self.attention_factor * torch.log(torch.tensor(seq_len)).int(), seq_len)
        
        # Select important queries (simplified selection)
        indices = torch.randperm(seq_len)[:num_selected]
        sparse_x = x[:, indices, :]
        
        attn_output, _ = self.attention(sparse_x, x, x)
        return attn_output

class DeepARProbabilisticLayer(nn.Module):
    def __init__(self, embedding_dim):
        super(DeepARProbabilisticLayer, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, batch_first=True)
        self.linear_mu = nn.Linear(embedding_dim, 1)
        self.linear_sigma = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        mu = self.linear_mu(lstm_out)
        sigma = F.softplus(self.linear_sigma(lstm_out))  # Ensure sigma is positive
        return mu, sigma

# ------------------------------------------------------------------------------
# Recurrent Memory Modules
# ------------------------------------------------------------------------------
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

class GRUMemoryCells(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(GRUMemoryCells, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        out, _ = self.gru(x)
        return out

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

class NBeatsBasisExpansion(nn.Module):
    def __init__(self, input_dim, theta_dim=8, num_blocks=3):
        super(NBeatsBasisExpansion, self).__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_dim, theta_dim) for _ in range(num_blocks)
        ])

    def forward(self, x):
        residuals = x
        forecast = 0
        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = residuals - backcast
            forecast = forecast + block_forecast
        return forecast

class NBeatsBlock(nn.Module):
    def __init__(self, input_dim, theta_dim):
        super(NBeatsBlock, self).__init__()
        self.linear1 = nn.Linear(input_dim, theta_dim)
        self.linear2 = nn.Linear(theta_dim, input_dim)

    def forward(self, x):
        theta = self.linear1(x)
        backcast = self.linear2(theta)
        forecast = self.linear2(theta)
        return backcast, forecast

# ------------------------------------------------------------------------------
# Knowledge Distillation
# ------------------------------------------------------------------------------
class KnowledgeDistillationLayer(nn.Module):
    def __init__(self, student_dim, teacher_dim):
        super(KnowledgeDistillationLayer, self).__init__()
        self.linear = nn.Linear(student_dim, teacher_dim)

    def forward(self, student_output, teacher_output):
        # Distillation loss (e.g., MSE between student and teacher outputs)
        student_logits = self.linear(student_output)
        loss = F.mse_loss(student_logits, teacher_output)
        return loss

# ------------------------------------------------------------------------------
# Multi-Task Learning
# ------------------------------------------------------------------------------
class MultiTaskHeads(nn.Module):
    def __init__(self, input_dim, num_quantiles=9):
        super(MultiTaskHeads, self).__init__()
        self.point_forecast = nn.Linear(input_dim, 4320)
        self.quantile_forecast = nn.Linear(input_dim, 4320 * num_quantiles)
        self.anomaly_detection = nn.Linear(input_dim, 1)  # Binary classification
        self.pattern_recognition = nn.Linear(input_dim, 10)  # Example: 10 pattern classes

    def forward(self, x):
        point_forecast = self.point_forecast(x)
        quantile_forecast = self.quantile_forecast(x)
        anomaly_score = torch.sigmoid(self.anomaly_detection(x))  # Sigmoid for binary classification
        pattern_logits = self.pattern_recognition(x)  # Logits for pattern classes
        return point_forecast, quantile_forecast, anomaly_score, pattern_logits

# ------------------------------------------------------------------------------
# Decoder Architecture
# ------------------------------------------------------------------------------
class TransformerDecoder(nn.Module):
    def __init__(self, embedding_dim, num_layers=12, num_heads=8):
        super(TransformerDecoder, self).__init__()
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(embedding_dim, num_heads) for _ in range(num_layers)
        ])
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layers, num_layers)

    def forward(self, x, memory):
        # x: target sequence, memory: encoder output
        tgt = x.transpose(0, 1)  # (S, N, E) where S is sequence length, N is batch size, E is embedding dimension
        memory = memory.transpose(0, 1)
        output = self.transformer_decoder(tgt, memory)
        return output.transpose(0, 1)  # (N, S, E)

class TemporalMixtureDensityNetwork(nn.Module):
    def __init__(self, input_dim, num_components=5):
        super(TemporalMixtureDensityNetwork, self).__init__()
        self.linear_mu = nn.Linear(input_dim, num_components)
        self.linear_sigma = nn.Linear(input_dim, num_components)
        self.linear_pi = nn.Linear(input_dim, num_components)

    def forward(self, x):
        mu = self.linear_mu(x)
        sigma = torch.exp(self.linear_sigma(x))  # Ensure sigma is positive
        pi = F.softmax(self.linear_pi(x), dim=-1)  # Mixture coefficients
        return mu, sigma, pi

class MultiHorizonAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads=8):
        super(MultiHorizonAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return attn_output

class HierarchicalRefinementNetwork(nn.Module):
    def __init__(self, input_dim, num_layers=3):
        super(HierarchicalRefinementNetwork, self).__init__()
        self.refinement_layers = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.refinement_layers:
            x = F.relu(layer(x))
        return x

# ------------------------------------------------------------------------------
# Uncertainty Quantification
# ------------------------------------------------------------------------------
class MonteCarloDropout(nn.Module):
    def __init__(self, p=0.5):
        super(MonteCarloDropout, self).__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        return self.dropout(x)

class DeepEnsembleIntegration(nn.Module):
    def __init__(self):
        super(DeepEnsembleIntegration, self).__init__()
        # This layer doesn't have any learnable parameters
        pass

    def forward(self, ensemble_outputs):
        # Average the outputs from multiple models
        return torch.mean(torch.stack(ensemble_outputs), dim=0)

class ConformalPredictionLayer(nn.Module):
    def __init__(self):
        super(ConformalPredictionLayer, self).__init__()
        # Placeholder for conformal prediction logic
        pass

    def forward(self, predictions, calibration_data):
        # Implement conformal prediction to produce confidence intervals
        # This is a placeholder; actual implementation depends on the chosen method
        return predictions  # Return predictions as-is for now

# ------------------------------------------------------------------------------
# Output Layer
# ------------------------------------------------------------------------------
class OutputLayer(nn.Module):
    def __init__(self, input_dim, num_quantiles=9):
        super(OutputLayer, self).__init__()
        self.point_forecast = nn.Linear(input_dim, 4320)
        self.quantile_forecast = nn.Linear(input_dim, 4320 * num_quantiles)
        self.decompose_linear = nn.Linear(input_dim, 4320 * 3)  # Example: 3 components

    def forward(self, x):
        point_forecast = self.point_forecast(x)
        quantile_forecast = self.quantile_forecast(x)
        decomposed_components = self.decompose_linear(x)
        return point_forecast, quantile_forecast, decomposed_components

# ------------------------------------------------------------------------------
# Main Model
# ------------------------------------------------------------------------------
class TimeSeriesModel(nn.Module):
    def __init__(self, input_dim, calendar_dim, exo_dim, meta_dim, embedding_dim, num_quantiles=9):
        super().__init__()
        
        # --- Input Processing ---
        self.input_layer = InputLayer(input_dim, calendar_dim, exo_dim, meta_dim, embedding_dim)
        self.workforce_pattern = WorkforcePatternLayer(embedding_dim, embedding_dim)
        self.seasonal_decomp = SeasonalDecompositionLayer(seq_length=4320)
        
        # --- Feature Engineering ---
        self.adaptive_norm = AdaptiveNormalization(embedding_dim)
        self.fourier_transform = MultiScaleFourierLayer(embedding_dim, embedding_dim)
        self.wavelet_transform = WaveletTransformLayer(learnable=True)
        self.feature_fusion = AdaptiveFeatureFusion(embedding_dim, num_features=5)
        
        # --- Temporal Processing ---
        self.multi_resolution_pyramid = MultiResolutionFeaturePyramid(embedding_dim)
        self.tcn = TemporalConvNet(embedding_dim, embedding_dim)
        self.self_attention = MultiHeadAttentionWithProbMask(embedding_dim, num_heads=16)
        self.sparse_attention = InformerStyleSparseAttention(embedding_dim)
        self.hyper_attention = HyperAttention(embedding_dim)
        
        # --- Memory & Pattern Learning ---
        self.stacked_lstm = StackedBidirectionalLSTM(embedding_dim, [256, 128, 64])
        self.gru_memory = GRUMemoryCells(embedding_dim)
        self.temporal_memory = TemporalMemoryNetwork(embedding_dim, memory_dim=embedding_dim)
        self.nbeats = NBeatsNetwork(input_dim=embedding_dim)
        
        # --- Refinement & Ensemble ---
        self.residual_blocks = nn.ModuleList([
            ResidualGatedBlock(embedding_dim) for _ in range(4)
        ])
        self.deepar = DeepARProbabilisticLayer(embedding_dim)
        self.ensemble_models = nn.ModuleList([
            self._create_submodel() for _ in range(3)
        ])
        self.calibration = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
        # --- Output Generation ---
        self.multi_task_heads = MultiTaskHeads(embedding_dim, num_quantiles)
        self.monte_carlo_dropout = MonteCarloDropout(p=0.1)
        self.output_layer = OutputLayer(embedding_dim, num_quantiles)

    def _create_submodel(self):
        """Creates a smaller version of the main model for ensemble"""
        return nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )

    def forward(self, primary_time_series, calendar_features, exo_variables, 
                historical_events, meta_features, time_idx, day_of_week=None, 
                hour_of_day=None, is_holiday=None):
        
        # --- Input Processing ---
        x = self.input_layer(primary_time_series, calendar_features, exo_variables, 
                            historical_events, meta_features)
        x_daily, x_weekly, daily_pattern, weekly_pattern = self.seasonal_decomp(x)
        x = self.workforce_pattern(x, is_holiday, day_of_week, hour_of_day)
        x = x + x_daily + x_weekly
        
        # --- Feature Engineering ---
        x = self.adaptive_norm(x)
        fourier_features = self.fourier_transform(x, time_idx)
        wavelet_features = self.wavelet_transform(x)
        features = [x, fourier_features, wavelet_features]
        x = self.feature_fusion(features)
        
        # --- Temporal Processing ---
        feature_maps = self.multi_resolution_pyramid(x)
        x = self.tcn(x)
        x = self.self_attention(x, x, x)
        x = self.sparse_attention(x, x, x)
        x = self.hyper_attention(x)
        
        # --- Memory & Pattern Learning ---
        x = self.stacked_lstm(x)
        x = self.gru_memory(x)
        x = self.temporal_memory(x)
        x = self.nbeats(x)
        
        # --- Refinement & Ensemble ---
        for block in self.residual_blocks:
            x = block(x)
        mu, sigma = self.deepar(x)
        
        ensemble_outputs = []
        for model in self.ensemble_models:
            ensemble_outputs.append(model(x))
        ensemble_output = torch.cat(ensemble_outputs, dim=-1)
        x = x + self.calibration(ensemble_output)
        
        # --- Output Generation ---
        point_forecast, quantile_forecast, anomaly_score, pattern_logits = self.multi_task_heads(x)
        
        # Monte Carlo Dropout for uncertainty estimation
        mc_outputs = []
        for _ in range(10):  # Number of MC samples
            mc_x = self.monte_carlo_dropout(x)
            mc_output = self.output_layer(mc_x)
            mc_outputs.append(mc_output[0])
        
        mc_mean = torch.stack(mc_outputs).mean(dim=0)
        mc_std = torch.stack(mc_outputs).std(dim=0)
        
        # Final output with uncertainty
        point_forecast, quantile_forecast, decomposed_components = self.output_layer(x)
        
        return {
            "point_forecast": point_forecast,
            "quantile_forecast": quantile_forecast,
            "anomaly_score": anomaly_score,
            "pattern_logits": pattern_logits,
            "mu": mu,
            "sigma": sigma,
            "decomposed_components": decomposed_components,
            "uncertainty": {
                "mc_mean": mc_mean,
                "mc_std": mc_std,
                "confidence_intervals": torch.stack([mc_mean - 2*mc_std, mc_mean + 2*mc_std])
            },
            "patterns": {
                "daily": daily_pattern,
                "weekly": weekly_pattern
            }
        }

    def training_step(self, batch, batch_idx):
        outputs = self.forward(**batch)
        
        # Enhanced loss computation
        losses = {
            'mse': self.mse_loss(outputs["point_forecast"], batch["target"]),
            'quantile': self._compute_quantile_loss(outputs, batch),
            'pattern': self._compute_pattern_loss(outputs),
            'consistency': self._compute_consistency_loss(outputs),
            'ensemble': self._compute_ensemble_loss(outputs),
            'calibration': self._compute_calibration_loss(outputs, batch)
        }
        
        # Dynamic loss weighting
        weights = self._compute_loss_weights(losses)
        total_loss = sum(w * l for w, l in zip(weights.values(), losses.values()))
        
        return total_loss
        
    def _compute_loss_weights(self, losses):
        """Dynamically adjust loss weights based on training progress"""
        # Convert losses to tensor
        loss_tensor = torch.tensor(list(losses.values()))
        
        # Calculate weights using softmax for automatic normalization
        weights = F.softmax(loss_tensor, dim=0)
        
        return dict(zip(losses.keys(), weights))
    


    
# Add these new specialized layers at the appropriate location in the file

class HyperAttention(nn.Module):
    """Advanced attention mechanism with dynamic kernel selection"""
    def __init__(self, dim, num_heads=8, num_kernels=4):
        super().__init__()
        self.num_kernels = num_kernels
        self.kernel_weights = nn.Parameter(torch.ones(num_kernels) / num_kernels)
        self.attention_heads = nn.ModuleList([
            MultiHeadAttentionWithProbMask(dim, num_heads) for _ in range(num_kernels)
        ])
        self.kernel_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, num_kernels)
        )
        
    def forward(self, x):
        # Dynamic kernel selection
        kernel_scores = self.kernel_mlp(x.mean(dim=1))  # [batch_size, num_kernels]
        kernel_weights = F.softmax(kernel_scores, dim=-1)
        
        outputs = []
        for i in range(self.num_kernels):
            out = self.attention_heads[i](x, x, x)
            outputs.append(out)
            
        # Weighted combination of kernel outputs
        weighted_output = torch.stack([
            outputs[i] * kernel_weights[:, i:i+1, None] for i in range(self.num_kernels)
        ]).sum(dim=0)
        
        return weighted_output

class AdaptiveFeatureFusion(nn.Module):
    """Dynamically fuses features based on their importance"""
    def __init__(self, dim, num_features):
        super().__init__()
        self.feature_gates = nn.Parameter(torch.ones(num_features))
        self.feature_attention = nn.MultiheadAttention(dim, num_heads=4)
        self.feature_importance = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1)
        )
        
    def forward(self, features):
        # Calculate feature importance scores
        importance_scores = [self.feature_importance(f).squeeze(-1) for f in features]
        attention_weights = F.softmax(torch.stack(importance_scores, dim=1), dim=1)
        
        # Weighted feature fusion
        fused_features = sum([
            f * w.unsqueeze(-1) * g for f, w, g in 
            zip(features, attention_weights.unbind(1), F.softplus(self.feature_gates))
        ])
        
        return fused_features

class ResidualGatedBlock(nn.Module):
    """Enhanced residual block with gating mechanism"""
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.transform = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
        self.layer_norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        gate_values = self.gate(x)
        transformed = self.transform(x)
        return x + self.layer_norm(gate_values * transformed)


# ------------------------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    # Model parameters
    input_dim = 10
    calendar_dim = 31  # Example: days in a month
    exo_dim = 5
    meta_dim = 3
    embedding_dim = 64
    num_quantiles = 9

    # Create the model
    model = TimeSeriesModel(input_dim, calendar_dim, exo_dim, meta_dim, embedding_dim, num_quantiles)

    # Load the model (replace with your actual model loading code)
    model.load_state_dict(torch.load("path/to/your/trained_model.pth"))  # Load the saved weights
    model.eval()  # Set the model to evaluation mode

    # Generate random input data
    batch_size = 32
    seq_len = 100
    primary_time_series = torch.randn(batch_size, seq_len, input_dim)
    calendar_features = torch.randint(0, calendar_dim, (batch_size, seq_len)).long()
    exo_variables = torch.randn(batch_size, seq_len, exo_dim)
    historical_events = torch.randn(batch_size, seq_len, embedding_dim)  # Assuming already embedded
    meta_features = torch.randn(batch_size, seq_len, meta_dim)
    time_idx = torch.rand(batch_size, seq_len)  # Normalized time index

    # Pass the input through the model
    output = model(primary_time_series, calendar_features, exo_variables, historical_events, meta_features, time_idx)

    # Print the output shapes
    print("Point Forecast Shape:", output["point_forecast"].shape)
    print("Quantile Forecast Shape:", output["quantile_forecast"].shape)
    print("Anomaly Score Shape:", output["anomaly_score"].shape)
    print("Pattern Logits Shape:", output["pattern_logits"].shape)
    print("Mu Shape:", output["mu"].shape)
    print("Sigma Shape:", output["sigma"].shape)
    print("Decomposed Components Shape:", output["decomposed_components"].shape)
    print("Uncertainty Shape:", output["uncertainty"]["confidence_intervals"].shape)
