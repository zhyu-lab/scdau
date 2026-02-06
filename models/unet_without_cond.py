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


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, time_features):
        super(ResidualBlock, self).__init__()
        
        self.fc = nn.Linear(in_features, out_features)  
        self.norm = nn.LayerNorm(out_features)
        self.emb_layer1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_features, out_features)
        )
        self.act = nn.SiLU()
        self.drop = nn.Dropout(0)

    def forward(self, x, emb_t):

        h = self.fc(x)  

        h = h + self.emb_layer1(emb_t)
        h = self.norm(h)
        h = self.act(h)
        h = self.drop(h)
        return h


class Linear_UNet_nocn(nn.Module):
    def __init__(self, input_dim=64, hidden_dims=[512, 256, 128, 64], dropout=0.1):
        super(Linear_UNet_nocn, self).__init__()
        self.hidden_dims = hidden_dims

        self.time_embedding = TimeEmbedding(hidden_dims[0])

        # Create layers dynamically
        self.layers = nn.ModuleList()

        # Adjust the first ResidualBlock layer to ensure the input dimension matches
        self.layers.append(ResidualBlock(input_dim, hidden_dims[0], hidden_dims[0]))

        for i in range(len(hidden_dims) - 1):
            self.layers.append(ResidualBlock(hidden_dims[i], hidden_dims[i + 1], hidden_dims[0]))

        self.reverse_layers = nn.ModuleList()
        for i in reversed(range(len(hidden_dims) - 1)):
            self.reverse_layers.append(ResidualBlock(hidden_dims[i + 1], hidden_dims[i], hidden_dims[0]))

        # Adjust the output layer
        self.out1 = nn.Linear(hidden_dims[0], int(hidden_dims[1] * 2))
        self.norm_out = nn.LayerNorm(int(hidden_dims[1] * 2))
        self.out2 = nn.Linear(int(hidden_dims[1] * 2), input_dim, bias=True)

        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x_input, t):
        emb_t = self.time_embedding(t)
        x = x_input.float()


        # Forward pass with history saving
        history = []
        for layer in self.layers:
            x = layer(x, emb_t)

            history.append(x)

        history.pop()

        # Reverse pass with skip connections
        for layer in self.reverse_layers:
            x = layer(x, emb_t)

            x = x + history.pop()

        x = self.out1(x)

        x = self.norm_out(x)
        x = self.act(x)
        x = self.out2(x)

        return x