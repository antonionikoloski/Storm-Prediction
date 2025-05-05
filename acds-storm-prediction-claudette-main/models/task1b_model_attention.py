import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedConvLSTMCell(nn.Module):
    """
    An enhanced ConvLSTM cell with batch normalization and residual connections.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super(EnhancedConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim, 
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

        
        self.bn_i = nn.BatchNorm2d(hidden_dim)
        self.bn_f = nn.BatchNorm2d(hidden_dim)
        self.bn_g = nn.BatchNorm2d(hidden_dim)
        self.bn_o = nn.BatchNorm2d(hidden_dim)

    def forward(self, x, h_cur, c_cur):
        """
        x:      (B, input_dim, H, W)
        h_cur:  (B, hidden_dim, H, W)
        c_cur:  (B, hidden_dim, H, W)
        """
        combined = torch.cat([x, h_cur], dim=1) 
        gates = self.conv(combined)             
        i, f, g, o = torch.split(gates, self.hidden_dim, dim=1)


        i = self.bn_i(i)
        f = self.bn_f(f)
        g = self.bn_g(g)
        o = self.bn_o(o)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

       
        if self.input_dim == self.hidden_dim:
            h_next = h_next + x  

        return h_next, c_next

    def init_hidden(self, batch_size, height, width, device):
       
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        )

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.
    Computes attention weights over spatial dimensions.
    """
    def __init__(self, hidden_dim):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: (B, hidden_dim, H, W)
        """
        attn = self.conv(x)  
        attn = self.sigmoid(attn)  
        return x * attn  

class EnhancedConvLSTMForecast(nn.Module):
    """
    Enhanced Encoder-Decoder ConvLSTM model with multiple layers and attention mechanism.
    """
    def __init__(self, 
                 in_channels=4,      
                 hidden_dims=[64, 64], 
                 kernel_size=3,
                 height=384,
                 width=384,
                 in_time=12,          
                 out_frames=12,       
                 attention=True):
        super(EnhancedConvLSTMForecast, self).__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)
        self.kernel_size = kernel_size
        self.height = height
        self.width = width
        self.in_time = in_time
        self.out_frames = out_frames
        self.attention = attention


        self.encoder_convs = nn.ModuleList()
        for i in range(self.num_layers):
            input_dim = in_channels if i == 0 else hidden_dims[i-1]
            self.encoder_convs.append(
                nn.Conv2d(input_dim, hidden_dims[i], kernel_size=3, padding=1)
            )


        self.encoder_lstm = nn.ModuleList()
        for i in range(self.num_layers):
            input_dim = hidden_dims[i] if i == 0 else hidden_dims[i]
            self.encoder_lstm.append(
                EnhancedConvLSTMCell(input_dim=input_dim,
                                     hidden_dim=hidden_dims[i],
                                     kernel_size=kernel_size)
            )

      
        if self.attention:
            self.attn_layers = nn.ModuleList([
                SpatialAttention(hidden_dim=hd) for hd in hidden_dims
            ])

        
        self.decoder_lstm = nn.ModuleList()
        for i in range(self.num_layers):
            input_dim = hidden_dims[i] 
            self.decoder_lstm.append(
                EnhancedConvLSTMCell(input_dim=input_dim,
                                     hidden_dim=hidden_dims[i],
                                     kernel_size=kernel_size)
            )

     
        self.decoder_convs = nn.ModuleList()
        for i in range(self.num_layers):
            output_dim = hidden_dims[i]
            self.decoder_convs.append(
                nn.Conv2d(output_dim, hidden_dims[i], kernel_size=3, padding=1)
            )

       
        self.final_conv = nn.Conv2d(hidden_dims[-1], 1, kernel_size=1, padding=0)

    def forward(self, x):
        """
        x shape: (B, in_channels, in_time, H, W)
        Return shape: (B, out_frames, H, W)
        """
        b, c, t, h, w = x.shape
        device = x.device

        
        encoder_hidden = []
        encoder_cell = []
        for i in range(self.num_layers):
            h_enc, c_enc = self.encoder_lstm[i].init_hidden(b, h, w, device)
            encoder_hidden.append(h_enc)
            encoder_cell.append(c_enc)

       
        for time_i in range(t):
            x_frame = x[:, :, time_i] 
            for i in range(self.num_layers):
                x_conv = self.encoder_convs[i](x_frame) 
                h_enc, c_enc = self.encoder_lstm[i](x_conv, encoder_hidden[i], encoder_cell[i])
                if self.attention:
                    h_enc = self.attn_layers[i](h_enc) 
                encoder_hidden[i] = h_enc
                encoder_cell[i] = c_enc
                x_frame = h_enc  

        
        decoder_hidden = [h.clone() for h in encoder_hidden]
        decoder_cell = [c.clone() for c in encoder_cell]

       
        decoder_input = torch.zeros(b, self.hidden_dims[-1], h, w, device=device)


        outputs = []


        for _ in range(self.out_frames):
            x_dec = decoder_input
            for i in range(self.num_layers):
                h_dec, c_dec = self.decoder_lstm[i](x_dec, decoder_hidden[i], decoder_cell[i])
                if self.attention:
                    h_dec = self.attn_layers[i](h_dec)  
                decoder_hidden[i] = h_dec
                decoder_cell[i] = c_dec
                x_dec = self.decoder_convs[i](h_dec) 
           
            out_vil = self.final_conv(x_dec) 
            outputs.append(out_vil)
            decoder_input = x_dec 

        outputs = torch.stack(outputs, dim=1).squeeze(2) 
        return outputs

class Task1bEnhancedConvLSTM(nn.Module):
    """
    A wrapper for the Enhanced ConvLSTM Forecast model.
    """
    def __init__(self, 
                 in_channels=4,
                 in_time=12,
                 out_frames=12,   
                 hidden_dims=[64, 64],
                 height=384,
                 width=384,
                 kernel_size=3,
                 attention=True):
        super(Task1bEnhancedConvLSTM, self).__init__()
        self.model = EnhancedConvLSTMForecast(
            in_channels=in_channels,
            hidden_dims=hidden_dims,
            kernel_size=kernel_size,
            height=height,
            width=width,
            in_time=in_time,
            out_frames=out_frames,
            attention=attention
        )

    def forward(self, x):
        return self.model(x)
