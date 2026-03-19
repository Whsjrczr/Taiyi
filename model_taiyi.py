from torchvision import models
import torch.nn as nn

res_dict = {"resnet18": models.resnet18, "resnet34": models.resnet34, "resnet50": models.resnet50,
            "resnet101": models.resnet101, "resnet152": models.resnet152, "resnext50": models.resnext50_32x4d, "resnext101": models.resnext101_32x8d}


class ResBase(nn.Module):
    def __init__(self, res_name='resnet18'):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class Model(nn.Module):
    def __init__(self, w=224, h=224, class_num=5):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=5, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(5)
        self.relu2 = nn.ReLU()
        self.l1 = torch.nn.modules.linear.NonDynamicallyQuantizableLinear(w * h * 5, class_num)

    def forward(self, x):
        r1 = self.relu1(self.conv1(x))
        r2 = self.relu2(self.bn(self.conv2(r1)))
        r2 = r2.view(x.size(0), -1)
        l1 = self.l1(r2)
        return l1



import torch
import torch.nn as nn
from einops import rearrange, repeat


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.shape[0]

        # Linearly project the queries, keys, and values.
        qkv = self.qkv_proj(torch.cat([query, key, value], dim=-1))
        q, k, v = qkv.chunk(3, dim=-1)

        # Split queries, keys, and values into multiple heads.
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,
                                                                            2)  # [batch_size, num_heads, seq_len, head_dim]
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,
                                                                            2)  # [batch_size, num_heads, seq_len, head_dim]
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,
                                                                            2)  # [batch_size, num_heads, seq_len, head_dim]

        # Compute the dot-product attention scores.
        scores = torch.matmul(q, k.transpose(-2, -1))  # [batch_size, num_heads, seq_len, seq_len]
        scores = scores / (self.embed_dim ** 0.5)

        # Apply the attention mask, if given.
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))

        # Apply the softmax function to compute the attention weights.
        attn_weights = torch.softmax(scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
        attn_weights = self.dropout(attn_weights)

        # Compute the weighted sum of the values.
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]

        # Concatenate and linearly project the attention output.
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1,
                                                                    self.embed_dim)  # [batch_size, seq_len, embed_dim]
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        attn_stream = src
        attn_branch = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = attn_stream + attn_branch

        mlp_stream = src
        mlp_branch = self.dropout(self.linear2(self.dropout(torch.relu(self.linear1(self.norm1(src))))))
        src = mlp_stream + mlp_branch
        src = self.norm2(src)
        self.residual_states = {
            "attn": {
                "stream": attn_stream,
                "branch": attn_branch,
                "output": attn_stream + attn_branch,
            },
            "mlp": {
                "stream": mlp_stream,
                "branch": mlp_branch,
                "output": src,
            },
        }
        return src


class ViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=5, dim=12, depth=5, heads=4, mlp_dim=5, dropout=0):
        super().__init__()
        assert image_size % patch_size == 0, 'image size must be divisible by patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        # 将输入的图片分成patch
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        # Transformer的encoder部分
        self.transformer = nn.ModuleList([
            TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout)
            for _ in range(depth)
        ])

        # MLP分类器
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        # 提取patch，添加位置编码，添加CLS token
        x = self.patch_embed(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = x + self.pos_embed
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=x.shape[0])
        x = torch.cat([cls_tokens, x], dim=1)

        # 通过Transformer的encoder部分
        for transformer_layer in self.transformer:
            x = transformer_layer(x)

        # 取出CLS token的输出，进行分类
        x = x[:, 0]
        x = self.fc(x)
        return x

if __name__ == '__main__':
    vit = ViT()
    print(vit)


class ResidualMonitorMixin:
    """Minimal helper for exposing residual states to Taiyi."""

    def _set_single_residual_state(self, stream, branch, output):
        self.residual_states = {
            "default": {
                "stream": stream,
                "branch": branch,
                "output": output,
            }
        }

    def _set_multi_residual_state(self, **states):
        self.residual_states = states


class MinimalResidualBlock(nn.Module, ResidualMonitorMixin):
    """Template for a single residual add."""

    def __init__(self, dim, activation=nn.ReLU):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.act = activation()

    def forward(self, x):
        stream = x
        branch = self.act(self.fc(x))
        output = stream + branch
        self._set_single_residual_state(stream, branch, output)
        return output


class MinimalTransformerResidualBlock(nn.Module, ResidualMonitorMixin):
    """Template for blocks that contain multiple residual adds."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        attn_stream = x
        attn_branch = self.attn(self.norm1(x))
        x = attn_stream + attn_branch

        mlp_stream = x
        mlp_branch = self.mlp(self.norm2(x))
        x = mlp_stream + mlp_branch

        self._set_multi_residual_state(
            attn={
                "stream": attn_stream,
                "branch": attn_branch,
                "output": attn_stream + attn_branch,
            },
            mlp={
                "stream": mlp_stream,
                "branch": mlp_branch,
                "output": x,
            },
        )
        return x
