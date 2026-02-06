import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.optim as optim


class ConditionEmbedding(nn.Module):
    def __init__(self, cond_dim, embed_dim=128):
        super().__init__()
        self.cond_dim = cond_dim

       
        if cond_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(cond_dim, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
                nn.SiLU()
            )
        else:
           
            self.net = None

    def forward(self, target_dist):
        if self.cond_dim > 0:
            return self.net(target_dist)
        else:
       
            return None


class FiLMLayer(nn.Module):
    def __init__(self, num_features, embed_dim):
        super().__init__()
     
        if embed_dim is not None:
            self.scale = nn.Linear(embed_dim, num_features)
            self.shift = nn.Linear(embed_dim, num_features)

          
            nn.init.constant_(self.scale.weight, 0)
            nn.init.constant_(self.scale.bias, 1)  
            nn.init.constant_(self.shift.weight, 0)
            nn.init.constant_(self.shift.bias, 0)  
        else:
            self.scale = None
            self.shift = None

    def forward(self, x, cond_emb):
       
        if cond_emb is not None:
            gamma = self.scale(cond_emb)
            beta = self.shift(cond_emb)
            return gamma * x + beta
        else:
           
            return x


class UNetLayer(nn.Module):
    def __init__(self, in_dim, out_dim, embed_dim, final_layer=False):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

   
        if embed_dim is not None:
            self.film = FiLMLayer(out_dim, embed_dim)
        else:
            self.film = None

        self.activation = nn.Identity() if final_layer else nn.SiLU()

    def forward(self, x, cond_emb):
        x = self.linear(x)
        x = self.norm(x)

    
        if self.film is not None:
            x = self.film(x, cond_emb)

        return self.activation(x)


class CrossConditionedUNet(nn.Module):
    def __init__(self,
                 source_features=512,
                 target_features=512,
                 cond_features=0,  
                 embed_dim=128,
                 enc_dims=[256, 128, 64, 32],
                 dec_dims=[64, 128, 256, 512]):
        super().__init__()

  
        self.cond_features = cond_features

   
        self.cond_embed = ConditionEmbedding(cond_features, embed_dim) if cond_features > 0 else None


        self.encoder = nn.ModuleList()
        self.encoder.append(UNetLayer(source_features, enc_dims[0],
                                      embed_dim if cond_features > 0 else None))

        for i in range(1, 4):
            self.encoder.append(UNetLayer(enc_dims[i - 1], enc_dims[i],
                                          embed_dim if cond_features > 0 else None))

     
        self.bottleneck = UNetLayer(enc_dims[3], enc_dims[3],
                                    embed_dim if cond_features > 0 else None)

       
        self.decoder = nn.ModuleList()
        self.decoder.append(UNetLayer(enc_dims[3] + enc_dims[3], dec_dims[0],
                                      embed_dim if cond_features > 0 else None))

        for i in range(1, 4):
            self.decoder.append(UNetLayer(dec_dims[i - 1] + enc_dims[3 - i], dec_dims[i],
                                          embed_dim if cond_features > 0 else None))

     
        self.output_layer = nn.Sequential(
            nn.Linear(dec_dims[3], target_features)
        )

    def forward(self, source, target_dist=None):
       
       
        cond_emb = None
        if self.cond_features > 0:
            
            if target_dist is None:
                raise ValueError("模型需要条件输入，但未提供target_dist")

            cond_emb = self.cond_embed(target_dist)

        
        enc_outputs = []
        x = source

        for i in range(4):
            x = self.encoder[i](x, cond_emb)
            enc_outputs.append(x)

        
        x = self.bottleneck(x, cond_emb)

  
        x = torch.cat([x, enc_outputs[3]], dim=1)
        x = self.decoder[0](x, cond_emb)

        for i in range(1, 4):
            x = torch.cat([x, enc_outputs[3 - i]], dim=1)
            x = self.decoder[i](x, cond_emb)

        return self.output_layer(x)


def train_cross_conditioned_unet(model, source_data, target_dist, target_labels, model_dir, epochs=100, batch_size=64,
                                 device='cuda'):
    start_time = time.time()
    torch.manual_seed(42)
    model = model.to(device)

    if model.cond_features > 0:

        dataset = TensorDataset(source_data, target_dist, target_labels)
    else:

        dataset = TensorDataset(source_data, target_labels)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(),
                           lr=1e-4,
                           weight_decay=1e-5)
    criterion = nn.MSELoss()

    history = {'train_loss': []}
    total_samples = len(dataset)

    print(f"Training ({total_samples} samples) for {epochs} epochs...")
    print(f"Model type: {'Conditional' if model.cond_features > 0 else 'Unconditional'}")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
          
            if model.cond_features > 0:
               
                source, dist, target = batch
                source = source.to(device).float()
                dist = dist.to(device).float()
                target = target.to(device).float()

              
                outputs = model(source, dist)
            else:
              
                source, target = batch
                source = source.to(device).float()
                target = target.to(device).float()

             
                outputs = model(source)

         
            loss = criterion(outputs, target)

        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * source.size(0)

        epoch_train_loss = running_loss / total_samples
        history['train_loss'].append(epoch_train_loss)

        print(f"Epoch [{epoch}/{epochs}] | Train Loss: {epoch_train_loss:.6f}")

    total_time = time.time() - start_time
    print(f"\nTraining Time: {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"Final Train Loss: {epoch_train_loss:.6f}")

 
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, 'final_unet.pt'))

    return model, history