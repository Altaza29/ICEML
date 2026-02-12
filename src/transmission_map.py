import torch.nn as nn
import torch
#from torchsummary import summary

class DownSample(nn.Module): #For downsampling block in UNET. Class DownSample inherits from Parent class nn.Module
    def __init__(self, in_channels, out_channels): # To initialize object attributes. Called whenever object created from class
    #Self represents instance of object itself. When Class Downsample is created it expects two inputs in_channels and out_channels
        super(DownSample,self).__init__() #To call the constructor of parent class otherwise only new class constructor would be run
        self.model=nn.Sequential( # To implement a sequential module
            nn.Conv2d(in_channels, out_channels, 4,2,1,bias=False), #Kernel Size 4, stride 2, padding=1
            nn.LeakyReLU(0.2) #0.2 is the slope of LeakyRelu Function
            )
        
    def forward(self, x): #Represents forward function, with x being the input image
        down=self.model(x) #To go through the forward function
        return down #Return the output from a downward pass from forward function
    
class Upsample(nn.Module): #For upsampling block in UNet
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__() #To call constructor of parent class otherwise only new class constructor would be run
        
        self.model=nn.Sequential( #To implement the sequential module to run the block
            nn.ConvTranspose2d(in_channels, out_channels, 4,2,1,bias=False), #To upsample image
            nn.InstanceNorm2d(out_channels), #Makes distrbution of each image gaussian unlike accross the whole batch unlike batch norm
            nn.ReLU(inplace=True,) #To apply relu function. Inplace=true means input is replaced by output in the memory
                                    #Basically means faster training. Discouraged in Pytorch documentation
            )
        
    def forward(self,x,skip_input): #Skip_Input for skip connections detailed in the network image
        x=self.model(x) #To implement forward function
        x=torch.cat((x,skip_input),1) #Concatenate output with the skip_input accross the first dimension, first dimension is number of channels
        return x #Return the output from the upsampling round
        
class Generator(nn.Module): #Define the whole generator architecture
    def __init__(self, in_channels=3, out_channels=1): #Input and Output Channels are kept to 3 since we want an RGB Image
        super(Generator, self).__init__() # To call constructor of parent class otherwise only new class constructor would be run
        
        self.down1=DownSample(in_channels, 64) #The first downsample block to increase channels to 64
        self.down2=DownSample(64,128) #2nd downsample block increases channels to 128 while decreasing image size
        self.down3=DownSample(128, 256) #3rd downsample block increases channels to 256 while decreasing image size
        self.down4=DownSample(256, 512) #4th downsample block increases channels to 512 while decreasing image size
        self.down5=DownSample(512,512) #5th downsample block maintains channels to 512 while decreasing image size
        self.down6=DownSample(512,512) #6th downsample block maintains channels to 512 while decreasing image size
        self.down7=DownSample(512,512) #7th downsample block maintains channels to 512 while decreasing image size
        self.down8=DownSample(512,512) #8th downsample block maintains channels to 512 while decreasing image size
        
        
        self.up1=Upsample(512, 512) #Maintain channels to 512 and increase image size 
        self.up2=Upsample(1024, 512) #Upsample channels to 512 (in up1 due to skip connection 512 + 512=1024) and increase image size
        self.up3=Upsample(1024, 512) #Upsample channels to 512 (in up1 due to skip connection 512 + 512=1024 and increase image size
        self.up4=Upsample(1024, 512) #Downsample channels to 512 (in up1 due to skip connection 512 + 512=1024 and increase image size
        self.up5=Upsample(1024, 256) #Downsample channels to 256 and increase image size
        self.up6=Upsample(512, 128) #
        self.up7=Upsample(256, 64)
        
        self.final = nn.Sequential( #The final layer
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1), # out_channels
            
            )
        self.sigmoid=nn.Sigmoid()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1) # Global average pooling layer
        self.fc = nn.Linear(out_channels, 1) # Fully connected layer to output a single scalar
    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7) #Forward pass with skip connections
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        u8 = self.final(u7)
        
        
        u8 = self.global_avg_pool(u8) # Apply global average pooling
        u8 = u8.view(u8.size(0), -1) # Flatten the tensor
        u8 = self.fc(u8) # Apply the fully connected layer
        u8=self.sigmoid(u8)
        return u8


class Generator_tmap(nn.Module): #Define the whole generator architecture
    def __init__(self, in_channels=3, out_channels=1): #Input and Output Channels are kept to 3 since we want an RGB Image
        super(Generator_tmap, self).__init__() # To call constructor of parent class otherwise only new class constructor would be run
        
        self.down1=DownSample(in_channels, 64) #The first downsample block to increase channels to 64
        self.down2=DownSample(64,128) #2nd downsample block increases channels to 128 while decreasing image size
        self.down3=DownSample(128, 256) #3rd downsample block increases channels to 256 while decreasing image size
        self.down4=DownSample(256, 512) #4th downsample block increases channels to 512 while decreasing image size
        self.down5=DownSample(512,512) #5th downsample block maintains channels to 512 while decreasing image size
        self.down6=DownSample(512,512) #6th downsample block maintains channels to 512 while decreasing image size
        self.down7=DownSample(512,512) #7th downsample block maintains channels to 512 while decreasing image size
        self.down8=DownSample(512,512) #8th downsample block maintains channels to 512 while decreasing image size
        
        
        self.up1=Upsample(512, 512) #Maintain channels to 512 and increase image size 
        self.up2=Upsample(1024, 512) #Upsample channels to 512 (in up1 due to skip connection 512 + 512=1024) and increase image size
        self.up3=Upsample(1024, 512) #Upsample channels to 512 (in up1 due to skip connection 512 + 512=1024 and increase image size
        self.up4=Upsample(1024, 512) #Downsample channels to 512 (in up1 due to skip connection 512 + 512=1024 and increase image size
        self.up5=Upsample(1024, 256) #Downsample channels to 256 and increase image size
        self.up6=Upsample(512, 128) #
        self.up7=Upsample(256, 64)
        
        self.final = nn.Sequential( #The final layer
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1), # out_channels
            
            )
        self.sigmoid=nn.Sigmoid()
        
    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7) #Forward pass with skip connections
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        u8 = self.final(u7)
        u8=self.sigmoid(u8)
        return u8


