import torch
import torch.nn as nn 

class ResidualBlock(nn.Module):
    def __init__(self, channels, nonlinearity="leaky", use_batchnorm = False):
        super().__init__()
        
        self.channels = channels
        self.nonlinearity = nonlinearity

        #We are not adjusting the channel sizes for the residual net as that requires
        #An additional transformation
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
        else:
            self.nonlinearity = nn.ReLU(inplace = True)

        
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
        Make sure to print out the activation function
        """


        #"x + self.residual_block(x)" is the "shortcut". We're learning the difference 

        if self.nonlinearity.lower() == "leaky":
            return nn.LeakyReLU(x + self.residual_block(x))
        elif self.nonlinearity.lower() == "relu":
            return nn.ReLU(x + self.residual_block(x))

class Encoder(nn.Module):
    def __init__(self, start_channels = 3,
                encoder_depth = 4,
                nonlinearity = "leaky", 
                ):

        super().__init__()

        self.start_channels = start_channels
        self.encoder_depth = encoder_depth
        self.nonlinearity = nonlinearity
                                            
    
        #Logic for automating the in_channels and out_channels
        channels_flow = [(256 / encoder_depth * i, (256 / encoder_depth * i)/2) for i in range(1,encoder_depth+1)]
        channels_flow = channels_flow[::-1]

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
        
    # def __repr__(self):
    #     """Prints the class representation"""

    #     return f"Encoder(encoder ={self.encoder})"

    #     print()
        
    def forward(self, x):
        
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, start_channels, 
                intermediate_nonlinearity = "leaky",
                ending_linearity = "tanh", 
                norm_type = None,
                final_layer_dims = 3):
        
        super().__init__()

        all_layers = []

        #Start channels must match the preceding channel work
        all_layers += [nn.ConvTranspose2d(in_channels = start_channels,
                                        out_channels = 64,
                                        kernel_size = 4,
                                        stride = 2,
                                        padding =1)]
        

        if norm_type == "layer":
            """Layer normalization gets messy. You need to permute the dimensions as layer norm will work on las """
            normalization_layer = nn.LayerNorm(64)
            all_layers += normalization_layer
        elif norm_type == "batch":
            normalization_layer = nn.BatchNorm2d(64)
            all_layers += normalization_layer
        elif norm_type == "instance":
            normalization_layer = nn.InstanceNorm2d(64)
            all_layers += normalization_layer
        else: 
            "Skip the normalization layer"
            pass


        if intermediate_nonlinearity.lower() == "relu":
            all_layers += [nn.ReLU(inplace=True)]
        elif intermediate_nonlinearity.lower() == "tanh":
            all_layers += [nn.Tanh()]
        elif intermediate_nonlinearity.lower() == "leaky":
            all_layers += [nn.LeakyReLU(inplace=True)]



        final_upsample = nn.ConvTranspose2d(in_channels = 64, out_channels = final_layer_dims, kernel_size=4, stride=2, padding=1)

        all_layers += [final_upsample]

        if ending_linearity == "tanh":
            all_layers += [nn.Tanh()]

        ##Create the sequential
        self.decoder = nn.Sequential(*all_layers)


    def forward(self, x):
  
        return self.decoder(x)
    

class EncoderDecoderSkipConnection(nn.Module):

    def __init__(self,first_channels, res_blocks = 6):

        super().__init__()


        #TODO adjust automatic readout of channels from encoder -> res_blocks
        encoder = Encoder(start_channels=first_channels, nonlinearity="leaky")
        decoder = Decoder(start_channels = 128, 
                            intermediate_nonlinearity="leaky",
                            ending_linearity="tanh",
                            norm_type=None)
        # Residual blocks
        residuals = nn.Sequential(*[ResidualBlock(128, nonlinearity="leaky") for _ in range(res_blocks)])

        ###Equivalent to: 
        """
        nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        """

        self.encoder_resblock_decoder = nn.Sequential(encoder, residuals, decoder)


    def forward(self, x):
        x = self.encoder_resblock_decoder(x)
        return x



class PatchGANDiscriminator(nn.Module):

    def __init__(self, in_channels):
        """Lets just implement a basic version of this"""
        super().__init__()

        self.patchgan = nn.Sequential(
                                    nn.Conv2d(in_channels, out_channels = 64, kernel_size = 4, stride = 2, padding =1),
                                     nn.LeakyReLU(0.2, inplace = True),
                                     nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 1),
                                     nn.BatchNorm2d(128),
                                     nn.LeakyReLU(0.2, inplace = True),
                                     nn.Conv2d(in_channels = 128, out_channels = 256, stride = 2, kernel_size = 4, padding = 1),
                                     nn.BatchNorm2d(256), 
                                     nn.LeakyReLU(0.2, inplace = True),
                                     nn.Conv2d(in_channels = 256, out_channels = 1, stride = 1, kernel_size = 4, padding = 1)
                                     )
        
        "patchGANs downsample the photo and "


    def forward(self,x):
        return self.patchgan(x)





        
if __name__ == "__main__":
    #Assume channels = 3 to mirror the RGB channels of a photo.
    # encoder_sample = Encoder(start_channels=3, nonlinearity="leaky")
    # print(encoder_sample)

    # decoder_sample = Decoder(start_channels = 128, 
    #                         intermediate_nonlinearity="leaky",
    #                         ending_linearity="tanh",
    #                         norm_type=None)
    
    # print(decoder_sample)
    
    # residual_block_sample = ResidualBlock(128, nonlinearity="leaky", use_batchnorm = False)

    # print(residual_block_sample)

    encoder_decoder = EncoderDecoderSkipConnection(res_blocks = 3)

    print(encoder_decoder)

    patchgan = PatchGANDiscriminator(in_channels = 64)

    print(patchgan)

    print("Complete")
    


