import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self, encoder_hidden_dims, decoder_hidden_dims, use_agg=False):
        super(Autoencoder, self).__init__()
        self.use_agg = use_agg
        encoder_layers = []
        for i in range(len(encoder_hidden_dims)):
            if i == 0:
                encoder_layers.append(nn.Linear(512, encoder_hidden_dims[i]))
            else:
                if use_agg:
                    # Agg features are [B, HW, C]; LayerNorm is stable on last dim.
                    encoder_layers.append(nn.LayerNorm(encoder_hidden_dims[i-1]))
                else:
                    encoder_layers.append(torch.nn.BatchNorm1d(encoder_hidden_dims[i-1]))
                encoder_layers.append(nn.ReLU(inplace=True))
                encoder_layers.append(nn.Linear(encoder_hidden_dims[i-1], encoder_hidden_dims[i]))
        self.encoder = nn.Sequential(*encoder_layers)
             
        decoder_layers = []
        for i in range(len(decoder_hidden_dims)):
            if i == 0:
                decoder_layers.append(nn.Linear(encoder_hidden_dims[-1], decoder_hidden_dims[i]))
            else:
                decoder_layers.append(nn.ReLU(inplace=True))
                decoder_layers.append(nn.Linear(decoder_hidden_dims[i-1], decoder_hidden_dims[i]))
        self.decoder = nn.Sequential(*decoder_layers)
        # print(self.encoder, self.decoder)
    def forward(self, x):
        x = self.encoder(x)
        x = F.normalize(x, dim=-1)
        x = self.decoder(x)
        x = F.normalize(x, dim=-1)
        return x
    
    def encode(self, x):
        x = self.encoder(x)
        x = F.normalize(x, dim=-1)
        return x

    def decode(self, x):
        # x [B, HW, c]
        x = self.decoder(x)
        x = F.normalize(x, dim=-1)
        return x
