import torch
import torch.nn as nn
from torch.nn.modules import Transformer, TransformerEncoder, TransformerEncoderLayer, LayerNorm
import math
# get the device available
_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



class PositionalEncoding(nn.Module):
    """
    Positional Encoder for a Transformer Neural Network
    """

    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__() 
        self._device = device      
        pe = torch.zeros(max_len, d_model).to(_device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(self._device)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(self._device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1).to(self._device)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :].to(self._device)
       

class TransAm(nn.Module):
    """
    Transformer neural network for time-series forecasting
    """

    def __init__(self,feature_size=250,num_layers=1,dropout=0.1, heads=10, device=None, max_enc_len=50000):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self._device = device
        self.heads = heads
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size, max_enc_len, self._device)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=self.heads, dropout=dropout).to(self._device)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers).to(self._device)
        self.decoder = nn.Linear(feature_size,1).to(self._device)
        self.init_weights()
        

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            mask = self._generate_square_subsequent_mask(len(src)).to(self._device)
            self.src_mask = mask.to(self._device)

        src = self.pos_encoder(src).to(self._device)
        output = self.transformer_encoder(src,self.src_mask).to(self._device)#, self.src_mask)
        output = self.decoder(output).to(self._device)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1).to(self._device)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self._device)