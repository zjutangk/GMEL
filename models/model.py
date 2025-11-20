import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as Init


from modules.multihead_attention import MultiheadAttention
from modules.transformer import TransformerEncoder, Linear, LayerNorm

from modules.transformer import TransformerEncoder


class Earlyfusion(nn.Module):
    def __init__(
        self,
        orig_dim,
        output_dim=1,
        proj_dim=40,
        num_heads=5,
        layers=5,
        relu_dropout=0.1,
        embed_dropout=0.15,
        res_dropout=0.1,
        out_dropout=0.1,
        attn_dropout=0.2,
    ):
        super(Earlyfusion, self).__init__()

        self.proj_dim = proj_dim
        self.orig_dim = orig_dim
        self.num_mod = len(orig_dim)
        self.num_heads = num_heads
        self.layers = layers
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.out_dropout = out_dropout
        self.embed_dropout = embed_dropout

        # Projection Layers
        self.proj = nn.ModuleList(
            [
                nn.Conv1d(self.orig_dim[i], self.proj_dim, kernel_size=1, padding=0)
                for i in range(self.num_mod)
            ]
        )

        # Fusion
        self.fusion = TransformerEncoder(
            embed_dim=proj_dim,
            num_heads=self.num_heads,
            layers=self.layers,
            attn_dropout=self.attn_dropout,
            res_dropout=self.res_dropout,
            relu_dropout=self.relu_dropout,
            embed_dropout=self.embed_dropout,
        )

        # Output layers
        self.out_layer_proj1 = nn.Linear(self.proj_dim, self.proj_dim)
        self.out_layer_proj2 = nn.Linear(self.proj_dim, self.proj_dim)
        self.out_layer = nn.Linear(self.proj_dim, output_dim)

    def forward(self, x):
        """
        dimension [batch_size, seq_len, n_features]
        """
        for i in range(self.num_mod):
            x[i] = x[i].transpose(1, 2)
            x[i] = self.proj[i](x[i])
            x[i] = x[i].permute(2, 0, 1)  # [seq_len, batch_size, proj_dim]

        feature = torch.cat(x) 
        last_hs = self.fusion(feature)[0]
        # A residual block
        last_hs_proj = self.out_layer_proj2(
            F.dropout(
                F.relu(self.out_layer_proj1(last_hs)),
                p=self.out_dropout,
                training=self.training,
            )
        )
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)
        return output, last_hs
    
    def forward_multi_layer(self, x):
        for i in range(self.num_mod):
            x[i] = x[i].transpose(1, 2)
            x[i] = self.proj[i](x[i])
            x[i] = x[i].permute(2, 0, 1)  # [seq_len, batch_size, proj_dim]

        feature = torch.cat(x) 
        last_hs = self.fusion(feature)[0]
        proj1_out = F.relu(self.out_layer_proj1(last_hs))
        # A residual block
        last_hs_proj = self.out_layer_proj2(
            F.dropout(
                proj1_out,
                p=self.out_dropout,
                training=self.training,
            )
        )
        last_hs_proj_ori = last_hs_proj
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)
        
        # return output, embeddings
        return output,last_hs,proj1_out,last_hs_proj_ori


class Latefusion(nn.Module):
    def __init__(
        self,
        orig_dim,
        output_dim=1,
        proj_dim=40,
        num_heads=5,
        layers=5,
        relu_dropout=0.1,
        embed_dropout=0.15,
        res_dropout=0.1,
        out_dropout=0.1,
        attn_dropout=0.2,
        n_rules=5,
    ):
        super(Latefusion, self).__init__()

        self.proj_dim = proj_dim
        self.orig_dim = orig_dim
        self.num_mod = len(orig_dim)
        self.num_heads = num_heads
        self.layers = layers
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.out_dropout = out_dropout
        self.embed_dropout = embed_dropout
        self.n_rule = n_rules

        # Projection Layers
        self.proj = nn.ModuleList(
            [
                nn.Conv1d(self.orig_dim[i], self.proj_dim, kernel_size=1, padding=0)
                for i in range(self.num_mod)
            ]
        )

        # Encoders
        self.encoders = nn.ModuleList(
            [
                TransformerEncoder(
                    embed_dim=proj_dim,
                    num_heads=self.num_heads,
                    layers=self.layers,
                    attn_dropout=self.attn_dropout,
                    res_dropout=self.res_dropout,
                    relu_dropout=self.relu_dropout,
                    embed_dropout=self.embed_dropout,
                )
                for _ in range(self.num_mod)
            ]
        )

        # Output layers
        self.out_layer_proj0 = nn.Linear(3 * self.proj_dim, self.proj_dim)
        self.out_layer_proj1 = nn.Linear(self.proj_dim, self.proj_dim)
        self.out_layer_proj2 = nn.Linear(self.proj_dim, self.proj_dim)
        self.out_layer = nn.Linear(self.proj_dim, output_dim)
        # self.fls = Fuzzifier(self.proj_dim, n_rules=self.n_rule).to(self.device)

    def forward(self, x):
        """
        dimension [batch_size, seq_len, n_features]
        """
        hs = list()

        for i in range(self.num_mod):
            x[i] = x[i].transpose(1, 2)
            x[i] = self.proj[i](x[i])
            x[i] = x[i].permute(2, 0, 1)
            h_tmp = self.encoders[i](x[i])
            hs.append(h_tmp[0])

        last_hs_out = torch.cat(hs, dim=-1)
        # A residual block
        last_hs = F.relu(self.out_layer_proj0(last_hs_out))
        proj1_out = F.relu(self.out_layer_proj1(last_hs))
        last_hs_proj = self.out_layer_proj2(
            F.dropout(
                proj1_out,
                p=self.out_dropout,
                training=self.training,
            )
        )
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)
        # Return intermediate projections along with final output
        proj0_out = last_hs
        proj2_out = last_hs_proj
        
        # return output, F.normalize(last_hs_out, dim=1)
        return output, last_hs_out
    
    def forward_multi_layer(self, x):
        hs = list()

        for i in range(self.num_mod):
            x[i] = x[i].transpose(1, 2)
            x[i] = self.proj[i](x[i])
            x[i] = x[i].permute(2, 0, 1)
            h_tmp = self.encoders[i](x[i])
            hs.append(h_tmp[0])

        last_hs_out = torch.cat(hs, dim=-1)
        # A residual block
        last_hs = F.relu(self.out_layer_proj0(last_hs_out))
        proj1_out = F.relu(self.out_layer_proj1(last_hs))
        last_hs_proj = self.out_layer_proj2(
            F.dropout(
                proj1_out,
                p=self.out_dropout,
                training=self.training,
            )
        )
        last_hs_proj_ori = last_hs_proj
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)
        
        # Combine all projections into a dictionary
        embeddings = {
            "proj0": last_hs,
            "proj1": proj1_out, 
            "proj2": last_hs_proj
        }
        
        # return output, embeddings
        return output,last_hs,proj1_out,last_hs_proj_ori
    
    # def Calculate_Fuzzy_Boundary(self, x):
    #     fz_degree, x = self.fls.fuzzify(x)
    #     return x

class ReliableFusion_late(nn.Module):
    def __init__(self, orig_dim, output_dim=1, proj_dim=40, num_heads=5, layers=5, relu_dropout=0.1, embed_dropout=0.15,
                 res_dropout=0.1, out_dropout=0.1, attn_dropout=0.2):
        super().__init__()
        self.proj_dim = proj_dim
        self.orig_dim = orig_dim
        self.num_mod = len(orig_dim)
        self.num_heads = num_heads
        self.layers = layers
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.out_dropout = out_dropout
        self.embed_dropout = embed_dropout
        self.proj = nn.ModuleList(
            [
                nn.Conv1d(self.orig_dim[i], proj_dim, kernel_size=1, padding=0)
                for i in range(self.num_mod)
            ]
        )
        self.encoders = nn.ModuleList(
            [
                TransformerEncoder(
                    embed_dim=proj_dim,
                    num_heads=self.num_heads,
                    layers=self.layers,
                    attn_dropout=self.attn_dropout,
                    res_dropout=self.res_dropout,
                    relu_dropout=self.relu_dropout,
                    embed_dropout=self.embed_dropout,
                )
                for _ in range(self.num_mod)
            ]
        )
        self.normalize_before = True
        # self.embed_dim = 3 * self.proj_dim
        self.attention = MultiheadAttention(embed_dim=self.proj_dim, num_heads=self.num_heads, attn_dropout=attn_dropout)
        self.fc1 = Linear(self.proj_dim, 4*self.proj_dim)   # The "Add & Norm" part in the paper
        self.fc2 = Linear(4*self.proj_dim, self.proj_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.proj_dim) for _ in range(2)])
        self.norm = nn.LayerNorm(self.proj_dim)
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.proj_dim), nn.Linear(self.proj_dim, output_dim))

    def forward(self, x):
        hs = list()
        for i in range(self.num_mod):
            x[i] = x[i].transpose(1, 2)
            x[i] = self.proj[i](x[i])
            x[i] = x[i].permute(2, 0, 1)
            h_tmp = self.encoders[i](x[i])
            hs.append(h_tmp)

        x = torch.cat(hs, dim=0)
        feature_after_encoder = x

        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.attention(query=x, key=x, value=x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.gelu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        x = x.mean(dim=0)
        # x = x[0]
        x = self.mlp_head(x)
        return x, feature_after_encoder

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x
        
        
        
class Fuzzifier(nn.Module):
    def __init__(self, in_dim, n_rules, fz=True):
        super(Fuzzifier, self).__init__()
        self.in_dim = in_dim
        self.n_rules = n_rules
        self.fz = fz  # whether to do fuzzifying operation
        self.eps = 1e-10
        self.build_model()

    def build_model(self):
        if self.fz == True:  # create antecedent parameters
            self.fz_weight = nn.Parameter(torch.FloatTensor(size=(1, self.n_rules)), requires_grad=True)
            Init.uniform_(self.fz_weight, 0, 1)

            self.Cs = nn.Parameter(torch.FloatTensor(size=(self.in_dim, self.n_rules)), requires_grad=True)
            Init.normal_(self.Cs, mean=0, std=1)
            self.Vs = nn.Parameter(torch.FloatTensor(size=self.Cs.size()), requires_grad=True)
            Init.uniform_(self.Vs, 0, 1)

    # This process has calculated the fuzzy boundary
    def fuzzify(self, features):
        fz_degree = -(features.unsqueeze(dim=2) - self.Cs) ** 2 / ((2 * self.Vs ** 2) + self.eps)

        if self.fz == True:
            fz_degree = torch.exp(fz_degree)
            weighted_fz_degree = torch.min(fz_degree, dim=2)[0]
        fz_features = torch.mul(features, 2. - weighted_fz_degree)

        return weighted_fz_degree, fz_features
