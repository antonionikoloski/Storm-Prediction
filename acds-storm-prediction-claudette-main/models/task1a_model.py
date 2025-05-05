import torch
import torch.nn as nn

class EnhancedVILPredictor(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=64, out_frames=12):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.out_frames = out_frames
        self.encoder1 = ConvLSTMCell(input_channels, hidden_dim, kernel_size=3)
        self.down1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)
        
        self.encoder2 = ConvLSTMCell(hidden_dim, hidden_dim*2, kernel_size=3)
        self.down2 = nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=3, stride=2, padding=1)
        self.bottleneck = ConvLSTMCell(hidden_dim*2, hidden_dim*4, kernel_size=3)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder1 = ConvLSTMCell(hidden_dim*4, hidden_dim*2, kernel_size=3)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder2 = ConvLSTMCell(hidden_dim*2, hidden_dim, kernel_size=3)
        self.final_conv = nn.Conv2d(hidden_dim, out_frames, kernel_size=1)

    def forward(self, x):
        b, _, t, h, w = x.size()
        
        h1, c1 = self.encoder1.init_hidden(b, (h, w))       
        h2, c2 = self.encoder2.init_hidden(b, (h//2, w//2))  
        h_b, c_b = self.bottleneck.init_hidden(b, (h//4, w//4))  


        for t_step in range(t):
            x_t = x[:, :, t_step] 
            
    
            h1, c1 = self.encoder1(x_t, (h1, c1))
            d1 = self.down1(h1)   
            
            h2, c2 = self.encoder2(d1, (h2, c2))
            d2 = self.down2(h2)   


        h_b, c_b = self.bottleneck(d2, (h_b, c_b)) 


        u1 = self.up1(h_b)      
        h_d1, _ = self.decoder1(u1, None) 
        
        u2 = self.up2(h_d1)        
        h_d2, _ = self.decoder2(u2, None)  

        return self.final_conv(h_d2).unsqueeze(2)

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, 
                     kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(4 * hidden_dim // 4, 4 * hidden_dim)
        )

    def forward(self, x, hidden_state):
        if hidden_state is None:
            h, c = self.init_hidden(x.size(0), x.shape[-2:])
        else:
            h, c = hidden_state

        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        
        c_next = (torch.sigmoid(forgetgate) * c) + (torch.sigmoid(ingate) * torch.tanh(cellgate))
        h_next = torch.sigmoid(outgate) * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        h, w = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, h, w).to(next(self.parameters()).device),
            torch.zeros(batch_size, self.hidden_dim, h, w).to(next(self.parameters()).device)
        )