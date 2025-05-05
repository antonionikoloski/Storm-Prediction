import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1) 
        gates = self.conv(combined) 
        i, f, g, o = torch.split(gates, self.hidden_dim, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, height, width, device):
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        )

class Task1bConvLSTM(nn.Module):
    def __init__(
        self, 
        in_channels=4, 
        in_time=12, 
        out_frames=12, 
        hidden_dim=64, 
        num_layers=2, 
        kernel_size=3
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_frames = out_frames


        self.encoder_layers = nn.ModuleList()
        self.encoder_layers.append(ConvLSTMCell(in_channels, hidden_dim, kernel_size))
        for _ in range(1, num_layers):
            self.encoder_layers.append(ConvLSTMCell(hidden_dim, hidden_dim, kernel_size))


        self.decoder_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.decoder_layers.append(ConvLSTMCell(hidden_dim, hidden_dim, kernel_size))


        self.pred_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1)  
        )

    def forward(self, x):
        B, _, T_in, H, W = x.size()
        device = x.device

        encoder_h = []
        encoder_c = []
        for layer in self.encoder_layers:
            h, c = layer.init_hidden(B, H, W, device)
            encoder_h.append(h)
            encoder_c.append(c)


        for t in range(T_in):
            x_t = x[:, :, t, :, :] 
            for layer_idx in range(self.num_layers):
                h, c = self.encoder_layers[layer_idx](
                    x_t, encoder_h[layer_idx], encoder_c[layer_idx]
                )
                encoder_h[layer_idx] = h
                encoder_c[layer_idx] = c
                x_t = h 


        decoder_h = [h.clone() for h in encoder_h]
        decoder_c = [c.clone() for c in encoder_c]


        predictions = []
        for _ in range(self.out_frames):
            x_t = torch.zeros(B, self.hidden_dim, H, W, device=device)
            for layer_idx in range(self.num_layers):
                h, c = self.decoder_layers[layer_idx](
                    x_t, decoder_h[layer_idx], decoder_c[layer_idx]
                )
                decoder_h[layer_idx] = h
                decoder_c[layer_idx] = c
                x_t = h
            pred = self.pred_conv(x_t) 
            predictions.append(pred)


        return torch.stack(predictions, dim=1).squeeze(2)