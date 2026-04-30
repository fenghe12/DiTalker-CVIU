# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init as init
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import BasicTransformerBlock, FeedForward
from diffusers.models.embeddings import (PatchEmbed,
    PixArtAlphaTextProjection, TimestepEmbedding, Timesteps)
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.utils import (USE_PEFT_BACKEND, BaseOutput, is_torch_version,
                             logging)
from diffusers.utils.torch_utils import maybe_allow_in_graph
from einops import rearrange
from ditalker.models.scale_adapter import ScaleAdapter
from torch import nn
from .arch_transformer import ScaleEncoder
from .attention import (HunyuanDiTBlock, HunyuanTemporalTransformerBlock,
                        SelfAttentionTemporalTransformerBlock,
                        TemporalTransformerBlock)
from .embeddings import HunyuanCombinedTimestepTextSizeStyleEmbedding
from .norm import AdaLayerNormSingle
from .patch import (CasualPatchEmbed3D, Patch1D, PatchEmbed3D, PatchEmbedF3D,
                    TemporalUpsampler3D, UnPatch1D)
from .resampler import Resampler

try:
    from diffusers.models.embeddings import PixArtAlphaTextProjection
except:
    from diffusers.models.embeddings import \
        CaptionProjection as PixArtAlphaTextProjection



try:
    import torch_npu
    from opensora.npu_config import npu_config
except:
    torch_npu = None
    npu_config = None
    

from .transformer import (
    TransformerEncoder,
    TransformerEncoderLayer,
    PositionalEncoding,
    TransformerDecoderLayer,
    TransformerDecoder,
)


def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module

def build_mlp(hidden_size=1152, projector_dim=4096, z_dim=768):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )


class Block(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) or isinstance(t, list) else ((t,) * length)

class CausalConv3d(Block):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        enable_cached=True,
        bias=True,
        **kwargs,
    ):
        super().__init__()
        self.kernel_size = cast_tuple(kernel_size, 3)
        self.time_kernel_size = self.kernel_size[0]
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.stride = kwargs.pop("stride", 1)
        self.padding = kwargs.pop("padding", 0)
        self.padding = list(cast_tuple(self.padding, 3))
        self.padding[0] = 0
        self.stride = cast_tuple(self.stride, 3)
        # print(self.kernel_size)
        # input("test")
        self.conv = nn.Conv3d(
            chan_in,
            chan_out,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=bias
        )
        self.enable_cached = enable_cached
        self.causal_cached = None
        self.cache_offset = 0

    def forward(self, x):
        x_dtype = x.dtype
        if self.causal_cached is None:
            first_frame_pad = x[:, :, :1, :, :].repeat(
                (1, 1, self.time_kernel_size - 1, 1, 1)
            )
        else:
            first_frame_pad = self.causal_cached
        x = torch.concatenate((first_frame_pad, x), dim=2)
        # print(x.shape)

        if self.enable_cached and self.time_kernel_size != 1:
            if (self.time_kernel_size - 1) // self.stride[0] != 0:
                if self.cache_offset == 0:
                    self.causal_cached = x[:, :, -(self.time_kernel_size - 1) // self.stride[0]:]
                else:
                    self.causal_cached = x[:, :, :-self.cache_offset][:, :, -(self.time_kernel_size - 1) // self.stride[0]:]
            else:
                self.causal_cached = x[:, :, 0:0, :, :]

        if npu_config is not None and npu_config.on_npu:
            return npu_config.run_conv3d(self.conv, x, x_dtype)
        else:
            x = self.conv(x)
            return x
from torch.utils.checkpoint import checkpoint

class CausalConv3d_GC(CausalConv3d):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int]],
        init_method="random",
        **kwargs
    ):
        super().__init__(chan_in, chan_out, kernel_size, init_method, **kwargs)

    def forward(self, x):
        # 1 + 16   16 as video, 1 as image
        first_frame_pad = x[:, :, :1, :, :].repeat(
            (1, 1, self.time_kernel_size - 1, 1, 1)
        )  # b c t h w
        x = torch.concatenate((first_frame_pad, x), dim=2)  # 3 + 16
        return checkpoint(self.conv, x)

class InflatedConv3d(nn.Conv2d):
    def forward(self, x):
        video_length = x.shape[2]

        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = super().forward(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)

        return x

class PoseGuider(ModelMixin):
    def __init__(
        self,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 64, 128),
    ):
        super().__init__()


        # self.blocks = nn.ModuleList([])
        # self.blocks.append(CausalConv3d(3,16,(3,3,3),stride=1))
        # self.blocks.append(nn.Conv3d(16,32,(2,3,3),stride=2))
        # self.blocks.append(CausalConv3d(32,32,(3,3,3),stride=1))
        # self.blocks.append(nn.Conv3d(32,16,(2,3,3),stride=1))
        # self.blocks.append(CausalConv3d(16,8,(3,3,3),stride=1))
        # 3-layer 3D CNN as described in paper (Sec 3.1, Pose Adapter)
        self.blocks = nn.ModuleList([
            nn.Conv3d(3, 16, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            nn.Conv3d(16, 32, kernel_size=(2,3,3), stride=(2,2,2), padding=(0,1,1)),
            nn.Conv3d(32, 8, kernel_size=(2,3,3), stride=(1,1,1), padding=(0,1,1)),
        ])


        # for i in range(len(block_out_channels) - 1):
        #     channel_in = block_out_channels[i]
        #     channel_out = block_out_channels[i + 1]
        #     self.blocks.append(
        #         InflatedConv3d(channel_in, channel_in, kernel_size=3, padding=1)
        #     )
        #     self.blocks.append(
        #         InflatedConv3d(
        #             channel_in, channel_out, kernel_size=3, padding=1, stride=2
        #         )
        #     )

        self.conv_out = zero_module(
            nn.Conv3d(
            in_channels=8, 
            out_channels=4, 
            kernel_size=(3, 3, 3), 
            stride=(2, 2, 2), 
            padding=(1, 1, 1)
        )
        )

    def forward(self, conditioning):
        # embedding = self.conv_in(conditioning)
        embedding = F.silu(conditioning)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)
        embedding = F.pad(embedding, (0, 4, 0, 4, 0, 0))

        return embedding




def get_audio_window(audio, win_size:int=5):
    """

    Args:
        audio (numpy.ndarray): (N,)

    Returns:
        audio_wins (numpy.ndarray): (N, W)
    """
    num_frames = len(audio)
    ph_frames = []
    for rid in range(0, num_frames):
        ph = []
        for i in range(rid - win_size, rid + win_size + 1):
            if i < 0:
                ph.append(74)
            elif i >= num_frames:
                ph.append(74)
            else:
                ph.append(audio[i])

        ph_frames.append(ph)

    audio_wins = np.array(ph_frames)

    return audio_wins
def _reset_parameters(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """

    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Sequential(nn.Linear(input_dim, input_dim), Mish(), nn.Linear(input_dim, 1))
        self.softmax = nn.functional.softmax

    def forward(self, batch_rep, att_mask=None):
        """
        N: batch size, T: sequence length, H: Hidden dimension
        input:
            batch_rep : size (N, T, H)
        attention_weight:
            att_w : size (N, T, 1)
        att_mask:
            att_mask: size (N, T): if True, mask this item.
        return:
            utter_rep: size (N, H)
        """

        att_logits = self.W(batch_rep).squeeze(-1)
        # (N, T)
        if att_mask is not None:
            att_mask_logits = att_mask.to(dtype=batch_rep.dtype) * -100000.0
            # (N, T)
            att_logits = att_mask_logits + att_logits

        att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep




class StyleEncoder(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        pos_embed_len=256,
        input_dim=64,
        aggregate_method="average",
    ):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        _reset_parameters(self.encoder)

        self.pos_embed = PositionalEncoding(d_model, pos_embed_len)

        self.increase_embed_dim = nn.Linear(input_dim, d_model)

        self.aggregate_method = None
        if aggregate_method == "self_attention_pooling":
            self.aggregate_method = SelfAttentionPooling(d_model)
        elif aggregate_method == "average":
            pass
        else:
            raise ValueError(f"Invalid aggregate method {aggregate_method}")

    def forward(self, x, pad_mask=None):
        """

        Args:
            x (_type_): (B, num_frames(L), C_exp)
            pad_mask: (B, num_frames)

        Returns:
            style_code: (B, C_model)
        """
        x = self.increase_embed_dim(x)
        # (B, L, C)
        x = x.permute(1, 0, 2)
        # (L, B, C)

        pos = self.pos_embed(x.shape[0])
        pos = pos.permute(1, 0, 2)
        # (L, 1, C)

        style = self.encoder(x, pos=pos, src_key_padding_mask=pad_mask)
        # (L, B, C)

        if self.aggregate_method is not None:
            permute_style = style.permute(1, 0, 2)
            # (B, L, C)
            style_code = self.aggregate_method(permute_style, pad_mask)
            return style_code

        if pad_mask is None:
            style = style.permute(1, 2, 0)
            # (B, C, L)
            style_code = style.mean(2)
            # (B, C)
        else:
            permute_style = style.permute(1, 0, 2)
            # (B, L, C)
            permute_style[pad_mask] = 0
            sum_style_code = permute_style.sum(dim=1)
            # (B, C)
            valid_token_num = (~pad_mask).sum(dim=1).unsqueeze(-1)
            # (B, 1)
            style_code = sum_style_code / valid_token_num
            # (B, C)

        return style_code



class ContentEncoder(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        pos_embed_len=11,
        ph_embed_dim=128,
        num_phonemes = 41
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        _reset_parameters(self.encoder)

        self.pos_embed = PositionalEncoding(d_model, pos_embed_len)

        self.ph_embedding = nn.Embedding(num_phonemes, ph_embed_dim)
        self.increase_embed_dim = nn.Linear(ph_embed_dim, d_model)

    def forward(self, x):
        """

        Args:
            x (_type_): (B, num_frames, window)

        Returns:
            content: (B, num_frames, window, C_dmodel)
        """
        x_embedding = self.ph_embedding(x)
        x_embedding = self.increase_embed_dim(x_embedding)
        # (B, N, W, C)
        B, N, W, C = x_embedding.shape
        x_embedding = x_embedding.reshape(B * N, W, C)
        x_embedding = x_embedding.permute(1, 0, 2)
        # (W, B*N, C)

        pos = self.pos_embed(W)
        pos = pos.permute(1, 0, 2)
        # (W, 1, C)

        content = self.encoder(x_embedding, pos=pos)
        # (W, B*N, C)
        content = content.permute(1, 0, 2).reshape(B, N, W, C)
        # (B, N, W, C)

        return content


from scipy.io import loadmat
def get_video_style_clip(video_path, style_max_len, start_idx="random", dtype=torch.float32):
    if video_path[-3:] == "mat":
        face3d_all = loadmat(video_path)["coeff"]
        face3d_exp = face3d_all[:, 80:144]  # expression 3DMM range
    elif video_path[-3:] == "txt":
        face3d_exp = np.loadtxt(video_path)
    else:
        raise ValueError("Invalid 3DMM file extension")

    face3d_exp = torch.tensor(face3d_exp, dtype=dtype)

    length = face3d_exp.shape[0]
    if length >= style_max_len:
        clip_num_frames = style_max_len
        if start_idx == "random":
            clip_start_idx = np.random.randint(low=0, high=length - clip_num_frames + 1)
        elif start_idx == "middle":
            clip_start_idx = (length - clip_num_frames + 1) // 2
        elif isinstance(start_idx, int):
            clip_start_idx = start_idx
        else:
            raise ValueError(f"Invalid start_idx {start_idx}")

        face3d_clip = face3d_exp[clip_start_idx : clip_start_idx + clip_num_frames]
        pad_mask = torch.tensor([False] * style_max_len)
    else:
        padding = torch.zeros(style_max_len - length, face3d_exp.shape[1])
        face3d_clip = torch.cat((face3d_exp, padding), dim=0)
        pad_mask = torch.tensor([False] * length + [True] * (style_max_len - length))

    return face3d_clip, pad_mask


class FP32LayerNorm(nn.LayerNorm):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        return F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(origin_dtype)

import random
class Audio2Latents(nn.Module):
    def __init__(self,style_code_dim:int=64,audio_emb_channels:int=128,time_emb_dim:int=6912):
        super().__init__()
        self.style_code_dim = style_code_dim
        self.audio_emb_channels = audio_emb_channels
        self.time_emb_dim = time_emb_dim
        # self.style_code_proj = nn.Sequential(nn.Linear(style_code_dim*style_code_dim, style_code_dim*8),
        #                                     nn.SiLU(),
        #                                     nn.Linear(style_code_dim*8,self.time_emb_dim),)
        
        # self.audio_emb_proj = nn.Linear(audio_emb_dim, time_emb_dim)
        self.audio_emb_proj = nn.Sequential(nn.Conv1d(in_channels=self.audio_emb_channels, out_channels=6, kernel_size=1),
                                            nn.SiLU(),)
        self.silu = nn.SiLU()
        self.K=nn.Parameter(torch.randn(1, time_emb_dim))
        self.V=nn.Parameter(torch.randn(1, time_emb_dim))
        # self.Q_norm = FP32LayerNorm(normalized_shape=time_emb_dim)
        # self.K_norm = FP32LayerNorm(normalized_shape=time_emb_dim)
        # self.V_norm = FP32LayerNorm(normalized_shape=time_emb_dim)

        self.Q_norm = nn.LayerNorm(normalized_shape=time_emb_dim)
        self.K_norm = nn.LayerNorm(normalized_shape=time_emb_dim)
        self.V_norm = nn.LayerNorm(normalized_shape=time_emb_dim)
        pass
    def forward(self,  audio_emb, time_emb, is_training:bool = False,):
        batch_size = time_emb.shape[0]
        # if is_training:
        #         Q = self.silu(self.audio_emb_proj(audio_emb).view(batch_size, -1))
        #         Q = self.silu(self.style_code_proj(style_code.view(batch_size, -1)))
        # else:
        #     Q = self.silu(self.audio_emb_proj(audio_emb).view(batch_size, -1))


        Q = self.silu(self.audio_emb_proj(audio_emb).view(batch_size, -1))

        output = nn.functional.scaled_dot_product_attention(self.Q_norm(Q), self.K_norm(self.K), self.V_norm(self.V))

        return output



class Decoder(nn.Module):

    """
    parameters:
    d_model: 256
    nhead: 8
    num_decoder_layers: 3
    dim_feedforward: 1024
    dropout: 0.1
    activation: relu
    normalize_before: False
    return_intermediate_dec: False
    pos_embed_len: 11
    output_dim: 64
    """
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_decoder_layers=3,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        pos_embed_len=11,
        output_dim=64,
        **_,
    ) -> None:
        super().__init__()
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )
        _reset_parameters(self.decoder)

        self.pos_embed = PositionalEncoding(d_model, pos_embed_len)

        tail_hidden_dim = d_model // 2
        self.tail_fc = nn.Sequential(
            nn.Linear(d_model, tail_hidden_dim),
            nn.ReLU(),
            nn.Linear(tail_hidden_dim, tail_hidden_dim),
            nn.ReLU(),
            nn.Linear(tail_hidden_dim, output_dim),
        )
        # print(f"d_model: {d_model}")
        # print(f"nhead: {nhead}")
        # print(f"num_decoder_layers: {num_decoder_layers}")
        # print(f"dim_feedforward: {dim_feedforward}")
        # print(f"dropout: {dropout}")
        # print(f"activation: {activation}")
        # print(f"normalize_before: {normalize_before}")
        # print(f"return_intermediate_dec: {return_intermediate_dec}")
        # print(f"pos_embed_len: {pos_embed_len}")
        # print(f"output_dim: {output_dim}")



    def forward(self, content, style_code):
        """

        Args:
            content (_type_): (B, num_frames, window, C_dmodel)
            style_code (_type_): (B, C_dmodel)

        Returns:
            face3d: (B, num_frames, C_3dmm)
        """
        B, N, W, C = content.shape
        style = style_code.reshape(B, 1, 1, C).expand(B, N, W, C)
        style = style.permute(2, 0, 1, 3).reshape(W, B * N, C)
        # (W, B*N, C)

        content = content.permute(2, 0, 1, 3).reshape(W, B * N, C)
        # (W, B*N, C)
        tgt = torch.zeros_like(style)
        pos_embed = self.pos_embed(W)
        pos_embed = pos_embed.permute(1, 0, 2)
        face3d_feat = self.decoder(tgt, content, pos=pos_embed, query_pos=style)[0]
        # (W, B*N, C)
        face3d_feat = face3d_feat.permute(1, 0, 2).reshape(B, N, W, C)[:, :, W // 2, :]
        # (B, N, C)
        face3d = self.tail_fc(face3d_feat)
        # (B, N, C_exp)
        return face3d


########################################################################################################################################################






class CLIPProjection(nn.Module):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_features, hidden_size, num_tokens=120):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)
        self.act_1 = nn.GELU(approximate="tanh")
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)
        self.linear_2 = zero_module(self.linear_2)
    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states






class AudioProjection(nn.Module):
    def __init__(self, f=64, inner_dim=9216):
        # print("Initializing AudioProjection")
        super(AudioProjection, self).__init__()
        self.f = f
        self.inner_dim = inner_dim
        self.conv1 = nn.Conv1d(in_channels=f*50, out_channels=128, kernel_size=1)
        self.fc1 = nn.Linear( 384, self.inner_dim)
        self.fc2 = nn.Linear( self.inner_dim, 1152)
        # self.fc2 =zero_module(self.fc2)
        self.gelu = nn.GELU(approximate="tanh")

    def forward(self, x):
        # print("Running forward pass")

        # print(x.shape)
        
        b = x.size(0)
        f = x.size(1)
        # h = x.size(2)
        # w = x.size(3)


        

        # padded_x = torch.zeros(b, self.f, h, w).to(x.device,dtype=x.dtype)
        # padded_x[:, :f] = x
            
        # x = padded_x
        
        # if f != self.expected_f:
        #     if f < self.expected_f:
        #         padding = (0, 0, 0, 0, (self.expected_f - f) // 2 , (self.expected_f - f + 1) // 2)
        #         x = torch.nn.functional.pad(x, padding)
        #     else:
        #         raise ValueError(f"Expected input with at most {self.expected_f} channels but got {f}")

            # raise ValueError(f"Expected input with f={self.f}, but got f={f}")
        x=x.view(b,f*50,384) 
        # print(x.shape)


        # x=x.unsqueeze(-l)#（b，f*50，384，l） 
        x=self.conv1(x)#（b，f*50,l28,l） 
        x=self.gelu(x) 
        # print(x.shape)
        # x=x.squeeze(-l)#（b,f*50,l28）

        # Transpose and flatten for the fully connected layer 
        # x=x.permute(O,2,l).contiguous()#（b,l28,f*50） 
        # x=x.view(b,l28,-l)

        # Apply fully connected layer to map the last dimension from 50*l28 to ll52 
        x=self.fc1(x)#（b,l28,ll52）
        x=self.gelu(x)
        x=self.fc2(x)
        # print(x.shape)

        return x






class StyleProjection(nn.Module):
    def __init__(self, f, inner_dim=384):
        super(StyleProjection, self).__init__()
        self.f = f
        self.inner_dim = inner_dim
        self.conv1 = nn.Conv1d(in_channels=f, out_channels=128, kernel_size=1)
        self.fc1 = nn.Linear(64, self.inner_dim)
        self.fc2 = nn.Linear(self.inner_dim, 1152)
        # self.fc2 =zero_module(self.fc2)
        self.gelu = nn.GELU(approximate="tanh")

    def forward(self, x):
        b = x.size(0)
        f = x.size(1)

        # Reshape input for convolution
        # x = x.view(b, f*64, 384)

        # Convolution + activation
        x = self.conv1(x)
        x = self.gelu(x)

        # Fully connected layers
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)

        return x

####################################################################################################################
####################################################################################################################
class TimePositionalEncoding(nn.Module):
    def __init__(
        self, 
        d_model, 
        dropout = 0., 
        max_len = 24
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        b, c, f, h, w = x.size()
        x = rearrange(x, "b c f h w -> (b h w) f c")
        x = x + self.pe[:, :x.size(1)]
        x = rearrange(x, "(b h w) f c -> b c f h w", b=b, h=h, w=w)
        return self.dropout(x)
    
@dataclass
class Transformer3DModelOutput(BaseOutput):
    """
    The output of [`Transformer2DModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [`Transformer2DModel`] is discrete):
            The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability
            distributions for the unnoised latent pixels.
    """

    sample: torch.FloatTensor


class Transformer3DModel(ModelMixin, ConfigMixin):
    """
    A 3D Transformer model for image-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        sample_size (`int`, *optional*): The width of the latent images (specify if the input is **discrete**).
            This is fixed during training since it is used to learn a number of position embeddings.
        num_vector_embeds (`int`, *optional*):
            The number of classes of the vector embeddings of the latent pixels (specify if the input is **discrete**).
            Includes the class for the masked latent pixel.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to use in feed-forward.
        num_embeds_ada_norm ( `int`, *optional*):
            The number of diffusion steps used during training. Pass if at least one of the norm_layers is
            `AdaLayerNorm`. This is fixed during training since it is used to learn a number of embeddings that are
            added to the hidden states.

            During inference, you can denoise for up to but not more steps than `num_embeds_ada_norm`.
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlocks` attention should contain a bias parameter.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        num_vector_embeds: Optional[int] = None,
        patch_size: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
        caption_channels: int = None,
        # block type
        basic_block_type: str = "motionmodule",
        # enable_uvit
        enable_uvit: bool = False,
        enable_audio_attention =False,
        enable_style_attn=False,
        enable_scale_fusion= False,
        enable_audio_scale_fusion =False,
        enable_scale_time_embedding = False,
        enable_pose_adapter = False,
        enable_eca = False,
        enable_repa_1 = False,
        enable_repa_2 = False,
        encoder_depth :int = 8,
        # 3d patch params
        patch_3d: bool = False,
        fake_3d: bool = False,
        time_patch_size: Optional[int] = None,

        casual_3d: bool = False,
        casual_3d_upsampler_index: Optional[list] = None,

        # motion module kwargs
        motion_module_type = "VanillaGrid",
        motion_module_kwargs = None,
        motion_module_kwargs_odd = None,
        motion_module_kwargs_even = None,

        # time position encoding
        time_position_encoding_before_transformer = False,

        qk_norm = False,
        after_norm = False,
        
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.enable_uvit = enable_uvit
        inner_dim = num_attention_heads * attention_head_dim
        self.basic_block_type = basic_block_type
        self.patch_3d = patch_3d
        self.fake_3d = fake_3d
        self.casual_3d = casual_3d
        self.casual_3d_upsampler_index = casual_3d_upsampler_index

        conv_cls = nn.Conv2d if USE_PEFT_BACKEND else LoRACompatibleConv
        linear_cls = nn.Linear if USE_PEFT_BACKEND else LoRACompatibleLinear

        assert sample_size is not None, "Transformer3DModel over patched input must provide sample_size"

        self.height = sample_size
        self.width = sample_size

        self.patch_size = patch_size
        self.time_patch_size = self.patch_size if time_patch_size is None else time_patch_size
        interpolation_scale = self.config.sample_size // 64  # => 64 (= 512 pixart) has interpolation scale 1
        interpolation_scale = max(interpolation_scale, 1)
        self.enable_audio_attention = enable_audio_attention
        self.enable_style_attn = enable_style_attn
        self.enable_scale_time_embedding=enable_scale_time_embedding
        self.enable_scale_fusion = enable_scale_fusion
        self.enable_audio_scale_fusion = enable_audio_scale_fusion
        self.enable_pose_adapter = enable_pose_adapter
        self.enable_eca = enable_eca
        self.enable_repa_1 = enable_repa_1
        self.enable_repa_2 = enable_repa_2
        self.encoder_depth = encoder_depth
        self.encoder_depth = 8
        # print(enable_audio_attention)
        # print(self.enable_audio_attention)
        if self.casual_3d:
            self.pos_embed = CasualPatchEmbed3D(
                height=sample_size,
                width=sample_size,
                patch_size=patch_size,
                time_patch_size=self.time_patch_size,
                in_channels=in_channels,
                embed_dim=inner_dim,
                interpolation_scale=interpolation_scale,
            )
        elif self.patch_3d:
            if self.fake_3d:
                self.pos_embed = PatchEmbedF3D(
                    height=sample_size,
                    width=sample_size,
                    patch_size=patch_size,
                    in_channels=in_channels,
                    embed_dim=inner_dim,
                    interpolation_scale=interpolation_scale,
                )
            else:
                self.pos_embed = PatchEmbed3D(
                    height=sample_size,
                    width=sample_size,
                    patch_size=patch_size,
                    time_patch_size=self.time_patch_size,
                    in_channels=in_channels,
                    embed_dim=inner_dim,
                    interpolation_scale=interpolation_scale,
                )
        else:
            self.pos_embed = PatchEmbed(
                height=sample_size,
                width=sample_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=inner_dim,
                interpolation_scale=interpolation_scale,
            )
        if self.enable_scale_time_embedding:
            self.audio2latents = Audio2Latents(audio_emb_channels=128)

        # Scale Adapter: Q-Former architecture (Paper Sec 3.2)
        # Replaces two separate ScaleEncoders with a single ScaleAdapter.
        # Shared KV from cat(c_a, c_s), d_b learnable queries, outputs s_phi and s_alpha.
        # Zero-init weight + unit bias ensures balanced contributions at early training.
        if self.enable_scale_fusion or self.enable_audio_scale_fusion:
            self.scale_adapter = ScaleAdapter(
                feature_dim=cross_attention_dim,
                num_heads=self.config.num_attention_heads,
                num_layers=num_layers,
                qformer_layers=3,
            )


        # 3. Define transformers blocks
        if self.basic_block_type == "motionmodule":
            self.transformer_blocks = nn.ModuleList(
                [
                    TemporalTransformerBlock(
                        inner_dim,
                        num_attention_heads,
                        attention_head_dim,
                        dropout=dropout,
                        cross_attention_dim=cross_attention_dim,
                        activation_fn=activation_fn,
                        num_embeds_ada_norm=num_embeds_ada_norm,
                        attention_bias=attention_bias,
                        only_cross_attention=only_cross_attention,
                        double_self_attention=double_self_attention,
                        upcast_attention=upcast_attention,
                        norm_type=norm_type,
                        norm_elementwise_affine=norm_elementwise_affine,
                        norm_eps=norm_eps,
                        attention_type=attention_type,
                        motion_module_type=motion_module_type,
                        motion_module_kwargs=motion_module_kwargs,
                        qk_norm=qk_norm,
                        after_norm=after_norm,
                    )
                    for d in range(num_layers)
                ]
            )
        elif self.basic_block_type == "global_motionmodule":
            self.transformer_blocks = nn.ModuleList(
                [
                    TemporalTransformerBlock(
                        inner_dim,
                        num_attention_heads,
                        attention_head_dim,
                        dropout=dropout,
                        cross_attention_dim=cross_attention_dim,
                        activation_fn=activation_fn,
                        num_embeds_ada_norm=num_embeds_ada_norm,
                        attention_bias=attention_bias,
                        only_cross_attention=only_cross_attention,
                        double_self_attention=double_self_attention,
                        upcast_attention=upcast_attention,
                        norm_type=norm_type,
                        norm_elementwise_affine=norm_elementwise_affine,
                        norm_eps=norm_eps,
                        attention_type=attention_type,
                        motion_module_type=motion_module_type,
                        motion_module_kwargs=motion_module_kwargs_even if d % 2 == 0 else motion_module_kwargs_odd,
                        qk_norm=qk_norm,
                        after_norm=after_norm,
                        enable_audio_attention=self.enable_audio_attention,
                        enable_style_attn=self.enable_style_attn,
                        enable_scale_fusion=self.enable_scale_fusion, 
                        enable_audio_scale_fusion=self.enable_audio_scale_fusion,
                        enable_eca=self.enable_eca    
                    )
                    for d in range(num_layers)
                ]
            )
        elif self.basic_block_type == "kvcompression_motionmodule":
            self.transformer_blocks = nn.ModuleList(
                [
                    TemporalTransformerBlock(
                        inner_dim,
                        num_attention_heads,
                        attention_head_dim,
                        dropout=dropout,
                        cross_attention_dim=cross_attention_dim,
                        activation_fn=activation_fn,
                        num_embeds_ada_norm=num_embeds_ada_norm,
                        attention_bias=attention_bias,
                        only_cross_attention=only_cross_attention,
                        double_self_attention=double_self_attention,
                        upcast_attention=upcast_attention,
                        norm_type=norm_type,
                        norm_elementwise_affine=norm_elementwise_affine,
                        norm_eps=norm_eps,
                        attention_type=attention_type,
                        kvcompression=False if d < 14 else True,
                        motion_module_type=motion_module_type,
                        motion_module_kwargs=motion_module_kwargs,
                        qk_norm=qk_norm,
                        after_norm=after_norm,
                    )
                    for d in range(num_layers)
                ]
            )
        elif self.basic_block_type == "selfattentiontemporal":
            self.transformer_blocks = nn.ModuleList(
                [
                    SelfAttentionTemporalTransformerBlock(
                        inner_dim,
                        num_attention_heads,
                        attention_head_dim,
                        dropout=dropout,
                        cross_attention_dim=cross_attention_dim,
                        activation_fn=activation_fn,
                        num_embeds_ada_norm=num_embeds_ada_norm,
                        attention_bias=attention_bias,
                        only_cross_attention=only_cross_attention,
                        double_self_attention=double_self_attention,
                        upcast_attention=upcast_attention,
                        norm_type=norm_type,
                        norm_elementwise_affine=norm_elementwise_affine,
                        norm_eps=norm_eps,
                        attention_type=attention_type,
                        qk_norm=qk_norm,
                        after_norm=after_norm,
                    )
                    for d in range(num_layers)
                ]
            )
        else:
            self.transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        inner_dim,
                        num_attention_heads,
                        attention_head_dim,
                        dropout=dropout,
                        cross_attention_dim=cross_attention_dim,
                        activation_fn=activation_fn,
                        num_embeds_ada_norm=num_embeds_ada_norm,
                        attention_bias=attention_bias,
                        only_cross_attention=only_cross_attention,
                        double_self_attention=double_self_attention,
                        upcast_attention=upcast_attention,
                        norm_type=norm_type,
                        norm_elementwise_affine=norm_elementwise_affine,
                        norm_eps=norm_eps,
                        attention_type=attention_type,
                    )
                    for d in range(num_layers)
                ]
            )
        
        if self.casual_3d:
            self.unpatch1d = TemporalUpsampler3D()
        elif self.patch_3d and self.fake_3d:
            self.unpatch1d = UnPatch1D(inner_dim, True)



        if self.enable_uvit:
            self.long_connect_fc = nn.ModuleList(
                [
                    nn.Linear(inner_dim, inner_dim, True) for d in range(13)
                ]
            )
            for index in range(13):
                self.long_connect_fc[index] = zero_module(self.long_connect_fc[index])
        if self.enable_audio_attention:
            # print("hahahah")
            self.audio_projection = AudioProjection(f=64)

        if self.enable_style_attn:
            self.style_model = StyleEncoder()
            self.style_content_model = ContentEncoder(num_phonemes=75)
            self.style_content_decoder=Decoder()
            self.style_projection = StyleProjection(f=64,inner_dim=384)
        if self.enable_pose_adapter:
            self.pose_adapter = PoseGuider()
            # print(self.audio_projection(torch.randn(2,64,50,384)).shape)
            # print(self.audio_projection)
        # 4. Define output layers
        if self.enable_repa_1 or self.enable_repa_2:
            self.repa_proj = build_mlp()
        self.out_channels = in_channels if out_channels is None else out_channels
        if norm_type != "ada_norm_single":
            self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
            self.proj_out_1 = nn.Linear(inner_dim, 2 * inner_dim)
            if self.patch_3d and not self.fake_3d:
                self.proj_out_2 = nn.Linear(inner_dim, self.time_patch_size * patch_size * patch_size * self.out_channels)
            else:
                self.proj_out_2 = nn.Linear(inner_dim, patch_size * patch_size * self.out_channels)
        elif norm_type == "ada_norm_single":
            self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
            self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
            if self.patch_3d and not self.fake_3d:
                self.proj_out = nn.Linear(inner_dim, self.time_patch_size * patch_size * patch_size * self.out_channels)
            else:
                self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * self.out_channels)

        # 5. PixArt-Alpha blocks.
        self.adaln_single = None
        self.use_additional_conditions = False
        if norm_type == "ada_norm_single":
            self.use_additional_conditions = self.config.sample_size == 128
            # TODO(Sayak, PVP) clean this, for now we use sample size to determine whether to use
            # additional conditions until we find better name
            self.adaln_single = AdaLayerNormSingle(inner_dim, use_additional_conditions=self.use_additional_conditions)

        self.caption_projection = None
        self.clip_projection = None
        if caption_channels is not None:
            self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=inner_dim)
            if in_channels == 12:
                self.clip_projection = CLIPProjection(in_features=768, hidden_size=inner_dim * 8)

        self.gradient_checkpointing = False
        
        self.time_position_encoding_before_transformer = time_position_encoding_before_transformer
        if self.time_position_encoding_before_transformer:
            self.t_pos = TimePositionalEncoding(max_len = 4096, d_model = inner_dim)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        inpaint_latents: torch.Tensor = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        clip_encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        clip_attention_mask: Optional[torch.Tensor] = None,
        audio_encoder_hidden_states: Optional[torch.Tensor] = None,
        phoneme_win :Optional[torch.Tensor] = None,
        style_clip:Optional[torch.Tensor] = None,
        pad_mask:Optional[torch.Tensor] = None,
        pixel_values_pose :Optional[torch.Tensor] = None,
        c_emo: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        The [`Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                Input `hidden_states`.
            encoder_hidden_states ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            attention_mask ( `torch.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer3DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        hidden_states = hidden_states.to(encoder_hidden_states.dtype)
        
        if self.enable_pose_adapter:
            # pose driving
            # print(hidden_states.shape)
            # # 
            # print("前",pixel_values_pose.shape)
            # print("后",self.pose_adapter(pixel_values_pose).shape)
            # print(hidden_states.shape)
            pixel_values_pose = pixel_values_pose.to(encoder_hidden_states.dtype)
            pixel_values_pose=self.pose_adapter(pixel_values_pose)
            # Paper Eq.3: z_0 = z_0 + w_pose * z_pose, w_pose = 0.5
            pixel_values_pose = pixel_values_pose * 0.5
            pixel_values_pose= pixel_values_pose[:,:,:hidden_states.shape[2],:hidden_states.shape[3],:hidden_states.shape[4]]
            pad_z = hidden_states.shape[2] - pixel_values_pose.shape[2]
            pad_h = hidden_states.shape[3] - pixel_values_pose.shape[3]
            pad_w = hidden_states.shape[4] - pixel_values_pose.shape[4]
            pixel_values_pose = F.pad(pixel_values_pose, (0, pad_w, 0, pad_h, 0, pad_z))
            
            # print("后", pixel_values_pose.shape)
            hidden_states = hidden_states + pixel_values_pose

            # hidden_states = hidden_states + (self.pose_adapter(pixel_values_pose)/5)[:,:,:hidden_states.shape[2]]
        # print(hidden_states.shape)


        # print(audio_encoder_hidden_states.shape)
        # input("test")

        

        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        if clip_attention_mask is not None:
            encoder_attention_mask = torch.cat([encoder_attention_mask, clip_attention_mask], dim=1)
        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(encoder_hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        if inpaint_latents is not None:
            hidden_states = torch.concat([hidden_states, inpaint_latents], 1)
        # 1. Input
        if self.casual_3d:
            video_length, height, width = (hidden_states.shape[-3] - 1) // self.time_patch_size + 1, hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
        elif self.patch_3d:
            video_length, height, width = hidden_states.shape[-3] // self.time_patch_size, hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
        else:
            video_length, height, width = hidden_states.shape[-3], hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
            hidden_states = rearrange(hidden_states, "b c f h w ->(b f) c h w")
        # print("video_length",video_length)
        hidden_states = self.pos_embed(hidden_states)
        if self.adaln_single is not None:
            if self.use_additional_conditions and added_cond_kwargs is None:
                raise ValueError(
                    "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
                )
            batch_size = hidden_states.shape[0] // video_length
            timestep, embedded_timestep = self.adaln_single(
                timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
            )

        hidden_states = rearrange(hidden_states, "(b f) (h w) c -> b c f h w", f=video_length, h=height, w=width)

        # hidden_states
        # bs, c, f, h, w => b (f h w ) c
        if self.time_position_encoding_before_transformer:
            hidden_states = self.t_pos(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        if self.enable_audio_attention:
            # audio_encoder_hidden_states = torch.randn(2,64,50,384).to("cuda")
            audio_encoder_hidden_states = self.audio_projection(audio_encoder_hidden_states)
            # print(audio_encoder_hidden_states.shape)
        else :
            audio_encoder_hidden_states = None
        if self.enable_style_attn:
            # print("style_clip",style_clip.shape)
            # print("pad_mask",pad_mask.shape)
            # print("phoneme_win",phoneme_win.shape)
            # style_code=self.style_model(style_clip, pad_mask if pad_mask is not None else None)
            # style_clip = style_clip.to(self.style_model.dtype)
            style_code=self.style_model(style_clip,  None)
            content = self.style_content_model(phoneme_win)
            gen_exp_stack = self.style_content_decoder(content, style_code)

            

            # style_code_tmp = gen_exp_stack.clone()


            style_encoder_hidden_states = self.style_projection(gen_exp_stack)


            # self.style_model = StyleEncoder()
            # self.style_content_model = ContentEncoder()
            # self.style_content_decoder=Decoder()
            # self.style_projection = StyleProjection(f=64,inner_dim=384)
            style_encoder_hidden_states = style_encoder_hidden_states.to(hidden_states.dtype)
            # print(style_encoder_hidden_states.shape)
        else :
            style_code_tmp = None
            style_encoder_hidden_states = None
        




        # 2. Blocks
        if self.caption_projection is not None:
            batch_size = hidden_states.shape[0]
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

        if clip_encoder_hidden_states is not None and encoder_hidden_states is not None:
            batch_size = hidden_states.shape[0]
            # print(clip_encoder_hidden_states.shape)
            clip_encoder_hidden_states = self.clip_projection(clip_encoder_hidden_states)
            # print(clip_encoder_hidden_states.shape)
            clip_encoder_hidden_states = clip_encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])
            # print(clip_encoder_hidden_states.shape)

            # Build c_emo = cat(c_text, c_ref) for ECA (Paper Sec 3.3, Eq.5)
            # c_text = projected T5 embeddings (encoder_hidden_states after caption_projection)
            # c_ref = projected CLIP embeddings (clip_encoder_hidden_states after clip_projection)
            if self.enable_eca and c_emo is None:
                c_emo = torch.cat([encoder_hidden_states, clip_encoder_hidden_states], dim=1)

            encoder_hidden_states = torch.cat([encoder_hidden_states, clip_encoder_hidden_states], dim = 1)
        if self.enable_scale_time_embedding:
            timestep = self.audio2latents(audio_emb=audio_encoder_hidden_states,time_emb=timestep,is_training =self.training )
            # print("hahaha")
            # print(style_code_tmp.shape,audio_encoder_hidden_states.shape,)
            # timestep = self.audio2latents(style_code = style_code_tmp,audio_emb=audio_encoder_hidden_states,time_emb=timestep,is_training =self.training )
            # print(timestep.shape)
            pass
        # Scale Adapter: predict s_phi (style) and s_alpha (audio) scaling factors
        # Paper Sec 3.2: shared Q-Former with c_a and c_s as KV
        if self.enable_scale_fusion or self.enable_audio_scale_fusion:
            # s_phi: (B, d_b, 1), s_alpha: (B, d_b, 1)
            scale_scalar, audio_scale_scalar = self.scale_adapter(
                style_encoder_hidden_states if style_encoder_hidden_states is not None else torch.zeros(hidden_states.shape[0], 1, hidden_states.shape[-1], device=hidden_states.device),
                audio_encoder_hidden_states if audio_encoder_hidden_states is not None else torch.zeros(hidden_states.shape[0], 1, hidden_states.shape[-1], device=hidden_states.device),
            )
        else:
            scale_scalar = torch.ones(hidden_states.shape[0], 28, 1, device=hidden_states.device)
            audio_scale_scalar = torch.ones(hidden_states.shape[0], 28, 1, device=hidden_states.device)
        skips = []
        skip_index = 0
        for index, block in enumerate(self.transformer_blocks):
            if self.enable_uvit:
                if index >= 15:
                    long_connect = self.long_connect_fc[skip_index](skips.pop())
                    hidden_states = hidden_states + long_connect
                    skip_index += 1

            if self.casual_3d_upsampler_index is not None and index in self.casual_3d_upsampler_index:
                hidden_states = rearrange(hidden_states, "b (f h w) c -> b c f h w", f=video_length, h=height, w=width)
                hidden_states = self.unpatch1d(hidden_states)
                video_length = (video_length - 1) * 2 + 1
                hidden_states = rearrange(hidden_states, "b c f h w -> b (f h w) c", f=video_length, h=height, w=width)
            # print("video_length",video_length)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs,**kwargs):
                        if return_dict is not None:
                            return module(*inputs, **kwargs,return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                args = {
                    "basic": [],
                    "motionmodule": [video_length, height, width],
                    "global_motionmodule": [video_length, height, width],
                    "selfattentiontemporal": [],
                    "kvcompression_motionmodule": [video_length, height, width],
                }[self.basic_block_type]
                # print("video_length",video_length)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                    audio_encoder_hidden_states,
                    style_encoder_hidden_states,
                    c_emo,
                    scale_scalar,
                    audio_scale_scalar,
                    index,
                    *args,
                    **ckpt_kwargs,          
                )
                # print(self.enable_repa_1 )
                # print(self.enable_repa_2 )
                # print(index + 1 ==8 and self.enable_repa_1)

                if ((index + 1) ==8) and self.enable_repa_1  :   
                    # print("1")                
                    hidden_states_repa = hidden_states.clone()
                    with torch.no_grad():
                        hidden_states_repa = self.proj_out(hidden_states_repa)
                    hidden_states_repa = hidden_states_repa.squeeze(1)
                    hidden_states_repa = hidden_states_repa.reshape(
                        shape=(-1, video_length, height, width, self.patch_size, self.patch_size, self.out_channels)
                     )
                    hidden_states_repa = torch.einsum("nfhwpqc->ncfhpwq", hidden_states_repa)
                    out_repa = hidden_states_repa.reshape(
                        shape=(-1, self.out_channels, video_length, height * self.patch_size, width * self.patch_size)
                     )
                    
                    lo_index = random.randint(0,out_repa.shape[2]-1)%16
                    out_repa = out_repa[:,:,lo_index].unsqueeze(2).chunk(2, dim=1)[0]
                    # video_length, height, width = out_repa.shape[-3], hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
                    # print("out_repa",out_repa.shape)
                    out_repa = rearrange(out_repa,"b c f h w -> (b f) c h w")
                    # print("out_repa",out_repa.shape)
                    with torch.no_grad():
                        out_repa = self.pos_embed(out_repa.repeat(1,3,1,1))
                    out_repa = out_repa[:,:256,:]
                    out_repa = self.repa_proj(out_repa)
                    print(out_repa.shape)

                    # print(out_repa.shape)
                if index + 1 ==self.encoder_depth and self.enable_repa_2 :
                    # print("2")  
                    lo_index = random.randint(0,video_length)%16
                    out_repa = hidden_states[:,256*(lo_index):256*(lo_index+1):,:]
                    out_repa = self.repa_proj(out_repa)
                # else:
                #     out_repa = None

            else:
                kwargs = {
                    "basic": {},
                    "motionmodule": {"num_frames":video_length, "height":height, "width":width},
                    "global_motionmodule": {"num_frames":video_length, "height":height, "width":width},
                    "selfattentiontemporal": {},
                    "kvcompression_motionmodule": {"num_frames":video_length, "height":height, "width":width},
                }[self.basic_block_type]
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                    audio_encoder_hidden_states=audio_encoder_hidden_states,
                    style_encoder_hidden_states=style_encoder_hidden_states,
                    scale_scalar=scale_scalar,
                    audio_scale_scalar=audio_scale_scalar,
                    c_emo=c_emo,
                    layer_index=index,
                    **kwargs
                )
                if index + 1 ==self.encoder_depth and self.enable_repa_1  :                   
                    hidden_states_repa = hidden_states.clone()
                    with torch.no_grad():
                        hidden_states_repa = self.proj_out(hidden_states_repa)
                    hidden_states_repa = hidden_states_repa.squeeze(1)
                    hidden_states_repa = hidden_states_repa.reshape(
                        shape=(-1, video_length, height, width, self.patch_size, self.patch_size, self.out_channels)
                     )
                    hidden_states_repa = torch.einsum("nfhwpqc->ncfhpwq", hidden_states_repa)
                    out_repa = hidden_states_repa.reshape(
                        shape=(-1, self.out_channels, video_length, height * self.patch_size, width * self.patch_size)
                     )
                    
                    lo_index = random.randint(0,out_repa.shape[2])%16
                    out_repa = out_repa[:,:,lo_index].unsqueeze(2).chunk(2, dim=1)[0]
                    # video_length, height, width = out_repa.shape[-3], hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
                    print("out_repa",out_repa.shape)
                    out_repa = rearrange(out_repa,"b c f h w -> (b f) c h w")
                    # print("out_repa",out_repa.shape)
                    with torch.no_grad():
                        out_repa = self.pos_embed(out_repa.repeat(1,3,1,1))
                    out_repa = out_repa[:,:256,:]
                    out_repa = self.repa_proj(out_repa)
                    z_s = out_repa
                    # print(out_repa.shape)
                if index + 1 ==self.encoder_depth and self.enable_repa_2  :
                    lo_index = random.randint(0,video_length)%16
                    out_repa = hidden_states[:,256*(lo_index):256*(lo_index+1):,:]
                    out_repa = self.repa_proj(out_repa)
                    z_s = out_repa
                # else:
                #     out_repa = None
                #     z_s = out_repa

           
            if self.enable_uvit:
                if index < 13:
                    skips.append(hidden_states)

        if self.fake_3d and self.patch_3d:
            hidden_states = rearrange(hidden_states, "b (f h w) c -> (b h w) c f", f=video_length, w=width, h=height)
            hidden_states = self.unpatch1d(hidden_states)
            hidden_states = rearrange(hidden_states, "(b h w) c f -> b (f h w) c", w=width, h=height)

        # 3. Output
        if self.config.norm_type != "ada_norm_single":
            conditioning = self.transformer_blocks[0].norm1.emb(
                timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
            shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
            hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
            hidden_states = self.proj_out_2(hidden_states)
        elif self.config.norm_type == "ada_norm_single":
            shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
            hidden_states = self.norm_out(hidden_states)
            # Modulation
            hidden_states = hidden_states * (1 + scale) + shift
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.squeeze(1)

        # unpatchify
        if self.adaln_single is None:
            height = width = int(hidden_states.shape[1] ** 0.5)
        if self.patch_3d:
            if self.fake_3d:
                hidden_states = hidden_states.reshape(
                    shape=(-1, video_length * self.patch_size, height, width, self.patch_size, self.patch_size, self.out_channels)
                )
                hidden_states = torch.einsum("nfhwpqc->ncfhpwq", hidden_states)
            else:
                hidden_states = hidden_states.reshape(
                    shape=(-1, video_length, height, width, self.time_patch_size, self.patch_size, self.patch_size, self.out_channels)
                )
                hidden_states = torch.einsum("nfhwopqc->ncfohpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(-1, self.out_channels, video_length * self.time_patch_size, height * self.patch_size, width * self.patch_size)
            )
        else:
            hidden_states = hidden_states.reshape(
                shape=(-1, video_length, height, width, self.patch_size, self.patch_size, self.out_channels)
            )
            hidden_states = torch.einsum("nfhwpqc->ncfhpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(-1, self.out_channels, video_length, height * self.patch_size, width * self.patch_size)
            )
        
        if not (self.enable_repa_1 or self.enable_repa_2):
            out_repa = None
        if not return_dict:
            # print(out_repa.shape)
            return (output,out_repa)

        

        return Transformer3DModelOutput(sample=output)

    @classmethod
    def from_pretrained_2d(cls, pretrained_model_path, subfolder=None, patch_size=2, transformer_additional_kwargs={}):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        print(f"loaded 3D transformer's pretrained weights from {pretrained_model_path} ...")

        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)

        from diffusers.utils import WEIGHTS_NAME
        model = cls.from_config(config, **transformer_additional_kwargs)
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        model_file_safetensors = model_file.replace(".bin", ".safetensors")
        if os.path.exists(model_file_safetensors):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(model_file_safetensors)
        else:
            if not os.path.isfile(model_file):
                raise RuntimeError(f"{model_file} does not exist")
            state_dict = torch.load(model_file, map_location="cpu")
        
        if model.state_dict()['pos_embed.proj.weight'].size() != state_dict['pos_embed.proj.weight'].size():
            new_shape   = model.state_dict()['pos_embed.proj.weight'].size()
            if len(new_shape) == 5:
                state_dict['pos_embed.proj.weight'] = state_dict['pos_embed.proj.weight'].unsqueeze(2).expand(new_shape).clone()
                state_dict['pos_embed.proj.weight'][:, :, :-1] = 0
            else:
                model.state_dict()['pos_embed.proj.weight'][:, :4, :, :] = state_dict['pos_embed.proj.weight']
                model.state_dict()['pos_embed.proj.weight'][:, 4:, :, :] = 0
                state_dict['pos_embed.proj.weight'] = model.state_dict()['pos_embed.proj.weight']
                
        if model.state_dict()['proj_out.weight'].size() != state_dict['proj_out.weight'].size():
            new_shape   = model.state_dict()['proj_out.weight'].size()
            state_dict['proj_out.weight'] = torch.tile(state_dict['proj_out.weight'], [patch_size, 1])

        if model.state_dict()['proj_out.bias'].size() != state_dict['proj_out.bias'].size():
            new_shape   = model.state_dict()['proj_out.bias'].size()
            state_dict['proj_out.bias'] = torch.tile(state_dict['proj_out.bias'], [patch_size])

        tmp_state_dict = {} 
        for key in state_dict:
            if key in model.state_dict().keys() and model.state_dict()[key].size() == state_dict[key].size():
                tmp_state_dict[key] = state_dict[key]
            else:
                print(key, "Size don't match, skip")
        state_dict = tmp_state_dict

        m, u = model.load_state_dict(state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        
        params = [p.numel() if "attn_temporal." in n else 0 for n, p in model.named_parameters()]
        print(f"### Attn temporal Parameters: {sum(params) / 1e6} M")
        
        return model

class HunyuanTransformer3DModel(ModelMixin, ConfigMixin):
    """
    HunYuanDiT: Diffusion model with a Transformer backbone.

    Inherit ModelMixin and ConfigMixin to be compatible with the sampler StableDiffusionPipeline of diffusers.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88):
            The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        patch_size (`int`, *optional*):
            The size of the patch to use for the input.
        activation_fn (`str`, *optional*, defaults to `"geglu"`):
            Activation function to use in feed-forward.
        sample_size (`int`, *optional*):
            The width of the latent images. This is fixed during training since it is used to learn a number of
            position embeddings.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        cross_attention_dim (`int`, *optional*):
            The number of dimension in the clip text embedding.
        hidden_size (`int`, *optional*):
            The size of hidden layer in the conditioning embedding layers.
        num_layers (`int`, *optional*, defaults to 1):
            The number of layers of Transformer blocks to use.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            The ratio of the hidden layer size to the input size.
        learn_sigma (`bool`, *optional*, defaults to `True`):
             Whether to predict variance.
        cross_attention_dim_t5 (`int`, *optional*):
            The number dimensions in t5 text embedding.
        pooled_projection_dim (`int`, *optional*):
            The size of the pooled projection.
        text_len (`int`, *optional*):
            The length of the clip text embedding.
        text_len_t5 (`int`, *optional*):
            The length of the T5 text embedding.
    """
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        patch_size: Optional[int] = None,
        n_query=16,
        projection_dim=768,
        activation_fn: str = "gelu-approximate",
        sample_size=32,
        hidden_size=1152,
        num_layers: int = 28,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = True,
        cross_attention_dim: int = 1024,
        norm_type: str = "layer_norm",
        cross_attention_dim_t5: int = 2048,
        pooled_projection_dim: int = 1024,
        text_len: int = 77,
        text_len_t5: int = 256,
    
        # block type
        basic_block_type: str = "basic",
        # motion module kwargs
        motion_module_type = "VanillaGrid",
        motion_module_kwargs = None,
        motion_module_kwargs_odd = None,
        motion_module_kwargs_even = None,

        time_position_encoding = False,
        after_norm = False,

    ):
        super().__init__()
        # 4. Define output layers
        if learn_sigma:
            self.out_channels = in_channels * 2 if out_channels is None else out_channels
        else:
            self.out_channels = in_channels if out_channels is None else out_channels
        self.enable_inpaint = in_channels * 2 != self.out_channels if learn_sigma else in_channels != self.out_channels
        self.num_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim
        self.basic_block_type = basic_block_type
        self.text_embedder = PixArtAlphaTextProjection(
            in_features=cross_attention_dim_t5,
            hidden_size=cross_attention_dim_t5 * 4,
            out_features=cross_attention_dim,
            act_fn="silu_fp32",
        )

        self.text_embedding_padding = nn.Parameter(
            torch.randn(text_len + text_len_t5, cross_attention_dim, dtype=torch.float32)
        )

        self.pos_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
            patch_size=patch_size,
            pos_embed_type=None,
        )

        self.time_extra_emb = HunyuanCombinedTimestepTextSizeStyleEmbedding(
            hidden_size,
            pooled_projection_dim=pooled_projection_dim,
            seq_len=text_len_t5,
            cross_attention_dim=cross_attention_dim_t5,
        )

        # 3. Define transformers blocks
        if self.basic_block_type == "motionmodule":
            self.blocks = nn.ModuleList(
                [
                    HunyuanTemporalTransformerBlock(
                        dim=self.inner_dim,
                        num_attention_heads=self.config.num_attention_heads,
                        activation_fn=activation_fn,
                        ff_inner_dim=int(self.inner_dim * mlp_ratio),
                        cross_attention_dim=cross_attention_dim,
                        qk_norm=True,  # See http://arxiv.org/abs/2302.05442 for details.
                        skip=layer > num_layers // 2,
                        motion_module_type=motion_module_type,
                        motion_module_kwargs=motion_module_kwargs,
                        after_norm=after_norm,
                    )
                    for layer in range(num_layers)
                ]
            )
        elif self.basic_block_type == "global_motionmodule":
            self.blocks = nn.ModuleList(
                [
                    HunyuanTemporalTransformerBlock(
                        dim=self.inner_dim,
                        num_attention_heads=self.config.num_attention_heads,
                        activation_fn=activation_fn,
                        ff_inner_dim=int(self.inner_dim * mlp_ratio),
                        cross_attention_dim=cross_attention_dim,
                        qk_norm=True,  # See http://arxiv.org/abs/2302.05442 for details.
                        skip=layer > num_layers // 2,
                        motion_module_type=motion_module_type,
                        motion_module_kwargs=motion_module_kwargs_even if layer % 2 == 0 else motion_module_kwargs_odd,
                        after_norm=after_norm,
                    )
                    for layer in range(num_layers)
                ]
            )
        elif self.basic_block_type == "hybrid_attention":
            self.blocks = nn.ModuleList(
                [
                    HunyuanDiTBlock(
                        dim=self.inner_dim,
                        num_attention_heads=self.config.num_attention_heads,
                        activation_fn=activation_fn,
                        ff_inner_dim=int(self.inner_dim * mlp_ratio),
                        cross_attention_dim=cross_attention_dim,
                        qk_norm=True,  # See http://arxiv.org/abs/2302.05442 for details.
                        skip=layer > num_layers // 2,
                        after_norm=after_norm,
                        time_position_encoding=time_position_encoding,
                        is_local_attention=False if layer % 2 == 0 else True,
                        local_attention_frames=2,
                        enable_inpaint=self.enable_inpaint,
                    )
                    for layer in range(num_layers)
                ]
            )
        else:
            # HunyuanDiT Blocks
            self.blocks = nn.ModuleList(
                [
                    HunyuanDiTBlock(
                        dim=self.inner_dim,
                        num_attention_heads=self.config.num_attention_heads,
                        activation_fn=activation_fn,
                        ff_inner_dim=int(self.inner_dim * mlp_ratio),
                        cross_attention_dim=cross_attention_dim,
                        qk_norm=True,  # See http://arxiv.org/abs/2302.05442 for details.
                        skip=layer > num_layers // 2,
                        after_norm=after_norm,
                        time_position_encoding=time_position_encoding,
                        enable_inpaint=self.enable_inpaint,
                    )
                    for layer in range(num_layers)
                ]
            )
            
        self.n_query = n_query
        if self.enable_inpaint:
            self.clip_padding = nn.Parameter(
                torch.randn((self.n_query, cross_attention_dim)) * 0.02
            )
            self.clip_projection = Resampler(
                int(math.sqrt(n_query)), 
                embed_dim=cross_attention_dim,
                num_heads=self.config.num_attention_heads,
                kv_dim=projection_dim,
                norm_layer=nn.LayerNorm,
            )
        else:
            self.clip_padding = None
            self.clip_projection = None

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False
    
    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states=None,
        text_embedding_mask=None,
        encoder_hidden_states_t5=None,
        text_embedding_mask_t5=None,
        image_meta_size=None,
        style=None,
        image_rotary_emb=None,
        inpaint_latents=None,
        clip_encoder_hidden_states: Optional[torch.Tensor]=None,
        clip_attention_mask: Optional[torch.Tensor]=None,
        return_dict=True,
    ):
        """
        The [`HunyuanDiT2DModel`] forward method.

        Args:
        hidden_states (`torch.Tensor` of shape `(batch size, dim, height, width)`):
            The input tensor.
        timestep ( `torch.LongTensor`, *optional*):
            Used to indicate denoising step.
        encoder_hidden_states ( `torch.Tensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
            Conditional embeddings for cross attention layer. This is the output of `BertModel`.
        text_embedding_mask: torch.Tensor
            An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. This is the output
            of `BertModel`.
        encoder_hidden_states_t5 ( `torch.Tensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
            Conditional embeddings for cross attention layer. This is the output of T5 Text Encoder.
        text_embedding_mask_t5: torch.Tensor
            An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. This is the output
            of T5 Text Encoder.
        image_meta_size (torch.Tensor):
            Conditional embedding indicate the image sizes
        style: torch.Tensor:
            Conditional embedding indicate the style
        image_rotary_emb (`torch.Tensor`):
            The image rotary embeddings to apply on query and key tensors during attention calculation.
        return_dict: bool
            Whether to return a dictionary.
        """
        if inpaint_latents is not None:
            hidden_states = torch.concat([hidden_states, inpaint_latents], 1)

        # unpatchify: (N, out_channels, H, W)
        patch_size = self.pos_embed.patch_size
        video_length, height, width = hidden_states.shape[-3], hidden_states.shape[-2] // patch_size, hidden_states.shape[-1] // patch_size
        hidden_states = rearrange(hidden_states, "b c f h w ->(b f) c h w")
        hidden_states = self.pos_embed(hidden_states)
        hidden_states = rearrange(hidden_states, "(b f) (h w) c -> b c f h w", f=video_length, h=height, w=width)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
            
        temb = self.time_extra_emb(
            timestep, encoder_hidden_states_t5, image_meta_size, style, hidden_dtype=timestep.dtype
        )  # [B, D]

        # text projection
        batch_size, sequence_length, _ = encoder_hidden_states_t5.shape
        encoder_hidden_states_t5 = self.text_embedder(
            encoder_hidden_states_t5.view(-1, encoder_hidden_states_t5.shape[-1])
        )
        encoder_hidden_states_t5 = encoder_hidden_states_t5.view(batch_size, sequence_length, -1)

        encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_t5], dim=1)
        text_embedding_mask = torch.cat([text_embedding_mask, text_embedding_mask_t5], dim=-1)
        text_embedding_mask = text_embedding_mask.unsqueeze(2).bool()

        encoder_hidden_states = torch.where(text_embedding_mask, encoder_hidden_states, self.text_embedding_padding)

        if clip_encoder_hidden_states is not None:
            batch_size = encoder_hidden_states.shape[0]

            clip_encoder_hidden_states = self.clip_projection(clip_encoder_hidden_states)
            clip_encoder_hidden_states = clip_encoder_hidden_states.view(batch_size, -1, encoder_hidden_states.shape[-1])

            clip_attention_mask = clip_attention_mask.unsqueeze(2).bool()
            clip_encoder_hidden_states = torch.where(clip_attention_mask, clip_encoder_hidden_states, self.clip_padding)

        skips = []
        for layer, block in enumerate(self.blocks):
            if layer > self.config.num_layers // 2:
                skip = skips.pop()
                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    args = {
                        "basic": [video_length, height, width, clip_encoder_hidden_states],
                        "hybrid_attention": [video_length, height, width, clip_encoder_hidden_states],
                        "motionmodule": [video_length, height, width],
                        "global_motionmodule": [video_length, height, width],
                    }[self.basic_block_type]
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                        skip,
                        *args,
                        **ckpt_kwargs,
                    )
                else:
                    kwargs = {
                        "basic": {"num_frames":video_length, "height":height, "width":width, "clip_encoder_hidden_states":clip_encoder_hidden_states},
                        "hybrid_attention": {"num_frames":video_length, "height":height, "width":width, "clip_encoder_hidden_states":clip_encoder_hidden_states},
                        "motionmodule": {"num_frames":video_length, "height":height, "width":width},
                        "global_motionmodule": {"num_frames":video_length, "height":height, "width":width},
                    }[self.basic_block_type]
                    hidden_states = block(
                        hidden_states,
                        temb=temb,
                        encoder_hidden_states=encoder_hidden_states,
                        image_rotary_emb=image_rotary_emb,  
                        skip=skip,                  
                        **kwargs
                    )  # (N, L, D)
            else:
                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward
                    
                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    args = {
                        "basic": [None, video_length, height, width, clip_encoder_hidden_states],
                        "hybrid_attention": [None, video_length, height, width, clip_encoder_hidden_states],
                        "motionmodule": [None, video_length, height, width],
                        "global_motionmodule": [None, video_length, height, width],
                    }[self.basic_block_type]
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb, 
                        *args,
                        **ckpt_kwargs,
                    )
                else:
                    kwargs = {
                        "basic": {"num_frames":video_length, "height":height, "width":width, "clip_encoder_hidden_states":clip_encoder_hidden_states},
                        "hybrid_attention": {"num_frames":video_length, "height":height, "width":width, "clip_encoder_hidden_states":clip_encoder_hidden_states},
                        "motionmodule": {"num_frames":video_length, "height":height, "width":width},
                        "global_motionmodule": {"num_frames":video_length, "height":height, "width":width},
                    }[self.basic_block_type]
                    hidden_states = block(
                        hidden_states,
                        temb=temb,
                        encoder_hidden_states=encoder_hidden_states,
                        image_rotary_emb=image_rotary_emb,  
                        **kwargs
                    )  # (N, L, D)

            if layer < (self.config.num_layers // 2 - 1):
                skips.append(hidden_states)

        # final layer
        hidden_states = self.norm_out(hidden_states, temb.to(torch.float32))
        hidden_states = self.proj_out(hidden_states)
        # (N, L, patch_size ** 2 * out_channels)

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], video_length, height, width, patch_size, patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nfhwpqc->ncfhpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.out_channels, video_length, height * patch_size, width * patch_size)
        )
        
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    @classmethod
    def from_pretrained_2d(cls, pretrained_model_path, subfolder=None, patch_size=2, transformer_additional_kwargs={}):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        print(f"loaded 3D transformer's pretrained weights from {pretrained_model_path} ...")

        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)

        from diffusers.utils import WEIGHTS_NAME
        model = cls.from_config(config, **transformer_additional_kwargs)
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        model_file_safetensors = model_file.replace(".bin", ".safetensors")
        if os.path.exists(model_file_safetensors):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(model_file_safetensors)
        else:
            if not os.path.isfile(model_file):
                raise RuntimeError(f"{model_file} does not exist")
            state_dict = torch.load(model_file, map_location="cpu")
        
        if model.state_dict()['pos_embed.proj.weight'].size() != state_dict['pos_embed.proj.weight'].size():
            new_shape   = model.state_dict()['pos_embed.proj.weight'].size()
            if len(new_shape) == 5:
                state_dict['pos_embed.proj.weight'] = state_dict['pos_embed.proj.weight'].unsqueeze(2).expand(new_shape).clone()
                state_dict['pos_embed.proj.weight'][:, :, :-1] = 0
            else:
                if model.state_dict()['pos_embed.proj.weight'].size()[1] > state_dict['pos_embed.proj.weight'].size()[1]:
                    model.state_dict()['pos_embed.proj.weight'][:, :state_dict['pos_embed.proj.weight'].size()[1], :, :] = state_dict['pos_embed.proj.weight']
                    model.state_dict()['pos_embed.proj.weight'][:, state_dict['pos_embed.proj.weight'].size()[1]:, :, :] = 0
                    state_dict['pos_embed.proj.weight'] = model.state_dict()['pos_embed.proj.weight']
                else:
                    model.state_dict()['pos_embed.proj.weight'][:, :, :, :] = state_dict['pos_embed.proj.weight'][:, :model.state_dict()['pos_embed.proj.weight'].size()[1], :, :]
                    state_dict['pos_embed.proj.weight'] = model.state_dict()['pos_embed.proj.weight']

        if model.state_dict()['proj_out.weight'].size() != state_dict['proj_out.weight'].size():
            if model.state_dict()['proj_out.weight'].size()[0] > state_dict['proj_out.weight'].size()[0]:
                model.state_dict()['proj_out.weight'][:state_dict['proj_out.weight'].size()[0], :] = state_dict['proj_out.weight']
                state_dict['proj_out.weight'] = model.state_dict()['proj_out.weight']
            else:
                model.state_dict()['proj_out.weight'][:, :] = state_dict['proj_out.weight'][:model.state_dict()['proj_out.weight'].size()[0], :]
                state_dict['proj_out.weight'] = model.state_dict()['proj_out.weight']

        if model.state_dict()['proj_out.bias'].size() != state_dict['proj_out.bias'].size():
            if model.state_dict()['proj_out.bias'].size()[0] > state_dict['proj_out.bias'].size()[0]:
                model.state_dict()['proj_out.bias'][:state_dict['proj_out.bias'].size()[0]] = state_dict['proj_out.bias']
                state_dict['proj_out.bias'] = model.state_dict()['proj_out.bias']
            else:
                model.state_dict()['proj_out.bias'][:, :] = state_dict['proj_out.bias'][:model.state_dict()['proj_out.bias'].size()[0], :]
                state_dict['proj_out.bias'] = model.state_dict()['proj_out.bias']

        tmp_state_dict = {} 
        for key in state_dict:
            if key in model.state_dict().keys() and model.state_dict()[key].size() == state_dict[key].size():
                tmp_state_dict[key] = state_dict[key]
            else:
                print(key, "Size don't match, skip")
        state_dict = tmp_state_dict

        m, u = model.load_state_dict(state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        print(m)
        
        params = [p.numel() if "mamba" in n else 0 for n, p in model.named_parameters()]
        print(f"### Mamba Parameters: {sum(params) / 1e6} M")

        params = [p.numel() if "attn1." in n else 0 for n, p in model.named_parameters()]
        print(f"### attn1 Parameters: {sum(params) / 1e6} M")
        
        return model




if __name__ == "__main__":
    audio_projection = AudioProjection(f=64)
    print(audio_projection(torch.randn(2,64,50,384)).shape)