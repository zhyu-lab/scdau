import sys
import torch
import torch.nn as nn
sys.path.append("..")
from diffusion.nn import timestep_embedding


class TimeEmbedding(nn.Module):
    def __init__(self, hidden_dim):
        super(TimeEmbedding, self).__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.hidden_dim = hidden_dim

    def forward(self, t):
        return self.time_embed(timestep_embedding(t, self.hidden_dim).squeeze(1))


class CondEmbedding(nn.Module):
    def __init__(self, cond_dim, hidden_dim):
        super(CondEmbedding, self).__init__()
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.hidden_dim = hidden_dim

    def forward(self, cond):
        return self.cond_embed(cond)


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, time_features, cond_features):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.emb_layer1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_features, out_features)
        )
        self.emb_layer2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_features, out_features)
        )
        self.act = nn.SiLU()
        self.drop = nn.Dropout(0)

    def forward(self, x, emb_t, emb_y):
        h = self.fc(x)
        h = h + self.emb_layer1(emb_t) + self.emb_layer2(emb_y)
        h = self.norm(h)
        h = self.act(h)
        h = self.drop(h)
        return h


class Linear_UNet(nn.Module):
    def __init__(self, input_dim=128, cond_dim=64, hidden_dims=[512, 256, 128, 64], dropout=0.1):
        super(Linear_UNet, self).__init__()
        self.hidden_dims = hidden_dims

        self.time_embedding = TimeEmbedding(hidden_dims[0])
        self.cond_embedding = CondEmbedding(cond_dim, hidden_dims[0])

        # Create layers dynamically  
        self.layers = nn.ModuleList()

        self.layers.append(ResidualBlock(input_dim, hidden_dims[0], hidden_dims[0], hidden_dims[0]))

        for i in range(len(hidden_dims) - 1):
            self.layers.append(ResidualBlock(hidden_dims[i], hidden_dims[i + 1], hidden_dims[0], hidden_dims[0]))

        self.reverse_layers = nn.ModuleList()
        for i in reversed(range(len(hidden_dims) - 1)):
            self.reverse_layers.append(ResidualBlock(hidden_dims[i + 1], hidden_dims[i], hidden_dims[0], hidden_dims[0]))

        self.out1 = nn.Linear(hidden_dims[0], int(hidden_dims[1] * 2))
        self.norm_out = nn.LayerNorm(int(hidden_dims[1] * 2))
        self.out2 = nn.Linear(int(hidden_dims[1] * 2), input_dim, bias=True)

        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x_input, t, y=None):
        emb_t = self.time_embedding(t)
        emb_y = self.cond_embedding(y)
        x = x_input.float()

        # Forward pass with history saving
        history = []
        for layer in self.layers:
            x = layer(x, emb_t, emb_y)
            history.append(x)

        history.pop()

        # Reverse pass with skip connections
        for layer in self.reverse_layers:
            x = layer(x, emb_t, emb_y)
            x = x + history.pop()  # Skip connection

        x = self.out1(x)
        x = self.norm_out(x)
        x = self.act(x)
        x = self.out2(x)
        return x


