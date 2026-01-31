import torch
import torch.nn as nn
import torch.nn.functional as F
from models import GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelGate(nn.Module):
    def __init__(self, channel: int, reduction: int = 16, enable: bool = True):
        super().__init__()
        self.enable = enable
        hidden = max(1, channel // reduction)

        self.gate = nn.Sequential(
            nn.Linear(channel, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channel),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enable:
            return x
        channel_avg = x.mean(dim=0, keepdim=True)
        gate_weights = self.gate(channel_avg)
        return x * gate_weights

class HDIGRL(nn.Module):
    def __init__(self,
                 r,
                 n_layers,
                 n_features,
                 num_rna,
                 num_dis,
                 n_global_node,
                 use_graph_conv: bool = True,
                 use_channel_gate: bool = True,
                 gate_reduction: int = 16):
        super().__init__()
        self.r = r
        self.n_layers = n_layers
        self.n_features = n_features
        self.num_rna = num_rna
        self.num_dis = num_dis
        self.n_global_node = n_global_node
        self.use_graph_conv = use_graph_conv
        self.use_channel_gate = use_channel_gate
        self.num_rna = num_rna
        self.num_dis = num_dis
        self.n_global_node = n_global_node
        n_node = num_rna + num_dis + n_global_node
        self.w_r_self = nn.Parameter(torch.zeros(size=(num_rna, n_features)))
        nn.init.xavier_uniform_(self.w_r_self.data)
        self.w_r_proj = nn.Parameter(torch.zeros(size=(num_rna, n_features)))
        nn.init.xavier_uniform_(self.w_r_proj.data)
        self.w_d_self = nn.Parameter(torch.zeros(size=(num_dis, n_features)))
        nn.init.xavier_uniform_(self.w_d_self.data)
        self.w_d_proj = nn.Parameter(torch.zeros(size=(num_dis, n_features)))
        nn.init.xavier_uniform_(self.w_d_proj.data)
        self.no_graph_proj = nn.Linear(self.n_features, self.r)

        self.GATLayers = nn.ModuleList([
            GATConv(n_features if i == 0 else r, r) for i in range(n_layers)])
        self.GateLayers = nn.ModuleList([
            ChannelGate(r) for _ in range(n_layers)])


        self.MLP = nn.Sequential(
            nn.Linear(2 * r, r),
            nn.ReLU(),
            nn.Linear(r, int(r / 2)),
            nn.ReLU(),
            nn.Linear(int(r / 2), 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            torch.nn.Linear(8, 1),
            nn.Sigmoid()
        )


        self.GateLayers = nn.ModuleList([
            ChannelGate(r, reduction=gate_reduction, enable=use_channel_gate)
            for _ in range(n_layers)
        ])



    def toggle_channel_gate(self, enable: bool):
        self.use_channel_gate = enable
        for g in self.GateLayers:
            if hasattr(g, "enable"):
                g.enable = enable

    def toggle_graph_conv(self, enable: bool):
        self.use_graph_conv = enable

    def projection_and_aggregation(self, rna_features, drug_features, inter_features_rna, inter_features_drug):
        x_rna1 = torch.mm(inter_features_rna, self.w_r_proj)
        x_rna2 = torch.mm(rna_features, self.w_r_self)
        attention_weights1 = F.softmax(torch.matmul(x_rna1, x_rna2.T), dim=1)
        x1 = torch.matmul(attention_weights1.unsqueeze(1), x_rna2.unsqueeze(0)).squeeze(1)

        x_drug1 = torch.mm(inter_features_drug, self.w_d_proj)
        x_drug2 = torch.mm(drug_features, self.w_d_self)

        attention_weights2 = F.softmax(torch.matmul(x_drug1, x_drug2.T), dim=1)
        x2 = torch.matmul(attention_weights2.unsqueeze(1), x_drug2.unsqueeze(0)).squeeze(1)

        global_feature = torch.zeros(size=(self.n_global_node, x1.shape[1])).to(torch.device(x1.device))
        return torch.vstack([x1, x2, global_feature])

    def embedding(self, x, adj, coo_data):
        if self.use_graph_conv:
            for gat, gate in zip(self.GATLayers, self.GateLayers):
                x = gat(x, adj)
                if self.use_channel_gate:
                    x = gate(x)
                x = F.relu(x)
        else:
            x = self.no_graph_proj(x)
            x = F.relu(x)
            x = F.relu(x)

        x_1 = x[:self.num_rna, :]
        x_2 = x[self.num_rna:(self.num_rna + self.num_dis), :]

        if isinstance(coo_data, torch.Tensor):
            idx1 = coo_data[:, 0].long().to(x.device)
            idx2 = coo_data[:, 1].long().to(x.device)
        else:
            coo_np = coo_data if isinstance(coo_data, (list, tuple)) else coo_data
            coo_np = torch.as_tensor(coo_np, dtype=torch.long, device=x.device)
            idx1 = coo_np[:, 0]
            idx2 = coo_np[:, 1]

        x_new = torch.cat([x_1[idx1], x_2[idx2]], dim=1)
        return x_new

    def forward(self, x, adj, coo_data):
        x_new = self.embedding(x, adj, coo_data)
        x_new = self.MLP(x_new)
        return x_new