from torch import nn                               # neural networks library
class Generator(nn.Module):
   
    def __init__(self, z_dim=10, im_chan=3, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 8),
            self.make_gen_block(hidden_dim * 8, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh(),
            )

    def forward(self, noise):
        
        x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(x)
    
    
class Classif(nn.Module):
    def __init__(self, im_chan=3, hidden_dim=64,n_classes=40):
        super(Classif, self).__init__()
        self.classif = nn.Sequential(
            self.make_classif_block(im_chan, hidden_dim),
            self.make_classif_block(hidden_dim, hidden_dim * 2),
            self.make_classif_block(hidden_dim * 2, hidden_dim * 4, stride=3),
            self.make_classif_block(hidden_dim * 4, n_classes, final_layer=True),
        )

    def make_classif_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a block of the classifier
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            print(output_channels)
            return nn.Sequential(
                
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh()
            )

    def forward(self, image):
        '''
        Function for completing a forward pass of the classifier
        Parameters:
            image: a flattened image tensor with dimension (im_chan)
        '''
        classif_pred = self.classif(image)
        return classif_pred.view(len(classif_pred), -1)    