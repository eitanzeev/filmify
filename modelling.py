import torch
import torch.nn as nn 

class ResidualBlock(nn.Module):
    def __init__(self, channels, nonlinearity="leaky", use_batchnorm = True):
        super().__init__()
        
        self.channels = channels
        self.nonlinearity = nonlinearity

        #We are not adjusting the channel sizes for the 
        self.res_conv2d = nn.Conv2d(in_channels=self.channels, 
                                                    out_channels = self.channels,
                                                    kernel_size = 3, 
                                                    stride = 1, 
                                                    padding = 1, 
                                                    bias = False if use_batchnorm else True)
                                    
        self.res_batchnorm2d = nn.BatchNorm2d(self.channels)
        
        if self.nonlinearity.lower() == "relu":
            self.nonlinearity = nn.ReLU(inplace=True)
        elif self.nonlinearity.lower() == "tanh":
            self.nonlinearity = nn.Tanh()
        elif self.nonlinearity.lower() == "leaky":
            self.nonlinearity = nn.LeakyReLU(inplace=True)

        
        if use_batchnorm:
            self.res_batchnorm2d = nn.BatchNorm2d(self.channels)
            self.residual_block = nn.Sequential(self.res_conv2d,
                                                self.res_batchnorm2d,
                                                self.nonlinearity,
                                                self.res_conv2d,
                                                self.res_batchnorm2d
                                                )
        else:
            self.residual_block = nn.Sequential(self.res_conv2d,
                                    self.nonlinearity,
                                    self.res_conv2d,
                                    )
            
    def forward(self, x):

        """Residual connections help the flow of gradients across the data it is defined as F(x) = H(x) - x
        Which can be re-written as H(x) = F(x) + X. Here the model just needs to learn difference represented by 
        """

        return x + self.residual_block(x) #This is the identity transformation of residual networks!
       

            

class Encoder(nn.Module):
    def __init__(self, start_channels = 256, encoder_depth = 4, nonlinearity = "leaky", 
                 
                 ):

        super().__init__()
        self.start_channels = start_channels
        self.encoder_depth = encoder_depth
        self.nonlinearity = nonlinearity
                                            
        


        #Logic for automating the in_channels and out_channels
        channels_flow = [(256 / encoder_depth * i, (256 / encoder_depth * i)/2) for i in range(1,encoder_depth+1)]
        channels_flow = channels_flow[:-1]

        if self.nonlinearity.lower() == "relu":
            self.nonlinearity = nn.ReLU(inplace=True)
        elif self.nonlinearity.lower() == "tanh":
            self.nonlinearity = nn.Tanh()
        elif self.nonlinearity.lower() == "leaky":
            self.nonlinearity = nn.LeakyReLU(inplace=True)

        

        self.encoder = nn.Sequential(nn.Conv2d(in_channels=start_channels, 
                                            out_channels = 64,
                                            kernel_size = 4,
                                            stride = 2,
                                            padding=1),
                                    self.nonlinearity,
                                    nn.Conv2d(in_channels = 64, 
                                            out_channels= 128,
                                            kernel_size=4,
                                            stride = 2,
                                            padding = 1),
                                    nn.BatchNorm2d(128),
                                    self.nonlinearity
                                    )
        
    def forward(self, x):
        
        return self.encoder(x)