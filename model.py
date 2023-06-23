import torch.nn as nn
import torch.nn.functional as F
import torchinfo


#####                   Assignment-8 Model                      ######


class ConvLayer(nn.Module):
    def __init__(self, input_channel, output_channel, padding=1, bias=False, skip=False, norm_type=None, n_groups=4, dropout=0):
        super(ConvLayer, self).__init__()

        # Member Variables
        self.skip = skip
        self.norm_type = norm_type
        self.n_groups = n_groups

        self.convlayer = nn.Conv2d(input_channel, output_channel, 3, padding=padding, bias=bias, padding_mode='replicate')
        self.normlayer = None
        if self.norm_type is not None:
            self.normlayer = self.get_norm_layer(output_channel)
        self.activation = nn.ReLU()
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

    def get_norm_layer(self, c):
        if self.norm_type == 'batch':
            return nn.BatchNorm2d(c)
        elif self.norm_type == 'layer':
            return nn.GroupNorm(1, c)
        elif self.norm_type == 'group':
            return nn.GroupNorm(self.n_groups, c)
        else:
            raise Exception(f'Unknown norm_type: {self.norm_type}')

    def forward(self, x):
        x_ = x
        x = self.convlayer(x)
        if self.normlayer is not None:
            x = self.normlayer(x)
        if self.skip:
            x += x_
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class Net(nn.Module):
    def __init__(self, norm_type=None, n_groups=4, dropout=0, skip=False):
        super(Net, self).__init__()

       
        self.norm_type = norm_type
        self.n_groups = n_groups
        self.dropout = dropout

        self.conv1 = self.get_conv_block(input_channel= 3, output_channel= 16, reps= 2, padding= 0, skip= False)
        self.pool1 = self.get_trans_block(input_channel= 16, output_channel = 24)
        self.conv2 = self.get_conv_block(input_channel= 24, output_channel= 24, reps= 3, padding= 1, skip= skip)
        self.pool2 = self.get_trans_block(input_channel= 24, output_channel= 32)
        self.conv3 = self.get_conv_block(input_channel= 32, output_channel= 32, reps= 3, padding= 1, skip= skip)

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, 10, 1, bias= True),
            nn.Flatten(),
            nn.LogSoftmax(-1)
        )

    def get_conv_block(self, input_channel, output_channel, reps=1, padding=1, bias=False, skip=False):
        block = list()
        for i in range(0, reps):
            block.append(
                ConvLayer(output_channel if i > 0 else input_channel, output_channel, padding=padding, bias=bias, skip=skip,
                          norm_type=self.norm_type, n_groups=self.n_groups, dropout=self.dropout)
            )
        return nn.Sequential(*block)

    @staticmethod
    def get_trans_block(input_channel, output_channel):
        return nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 1, bias=False),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.gap(x)

        return x


    def summary(self, input_size=None):
        return torchinfo.summary(self, input_size=input_size,
                                 col_names=["input_size", "output_size", "num_params", "params_percent"])

class GroupNormModel(Net):
    def __init__(self, n_groups=4, dropout=0, skip=False):
        super(GroupNormModel, self).__init__(norm_type='group', n_groups=n_groups, dropout=dropout, skip=skip)


class LayerNormModel(Net):
    def __init__(self, dropout=0, skip=False):
        super(LayerNormModel, self).__init__(norm_type='layer', dropout=dropout, skip=skip)


class BatchNormModel(Net):
    def __init__(self, dropout=0, skip=False):
        super(BatchNormModel, self).__init__(norm_type='batch', dropout=dropout, skip=skip)






####                         Assignment-7 Models                     ########

class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()

        #r_in: 1, n_in: 28, j_in: 1, s:1, r_out:3, n_out:26, j_out: 1 
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels= 1, out_channels= 16, kernel_size = 3),
            nn.ReLU()
        )

        #r_in: 3, n_in: 26, j_in: 1, s:1, r_out:5, n_out:24, j_out: 1 
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels= 16, out_channels= 16, kernel_size=3),
            nn.ReLU()
        )

        #r_in: 5, n_in: 24, j_in: 1, s:1, r_out:7, n_out:22, j_out: 1
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels= 16, out_channels=16, kernel_size= 3),
            nn.ReLU()
        )

        self.pool1 = nn.Sequential(
            nn.MaxPool2d(2,2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels= 32, kernel_size= 3),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels= 16, kernel_size= 3),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels= 16, out_channels= 16, kernel_size= 3),
            nn.ReLU()
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels= 16, out_channels= 32, kernel_size= 3),
            nn.ReLU()
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels= 32, out_channels= 16, kernel_size= 3),
            
        )

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = x.view(-1, 16)
        return F.log_softmax(x, dim=-1)

dropout_value = 0.1
class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) 

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=30, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(30),
            nn.Dropout(dropout_value)
        ) 

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=10, kernel_size=(3, 3)),
            nn.ReLU(),            
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) 
        self.pool1 = nn.MaxPool2d(2, 2) 

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(1, 1)),
            nn.ReLU(),            
            nn.BatchNorm2d(14),
            nn.Dropout(dropout_value)
        ) 

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=16, kernel_size=(3, 3)),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) 

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3)),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) 

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) 

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1)),
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.gap(x)
        x = self.conv7(x)        

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

    

class Model_3(nn.Module):
    def __init__(self):
        super(Model_3, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels= 1, out_channels= 8, kernel_size= 3),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.1)         
        )

        self.conv2 =nn.Sequential(
            nn.Conv2d(in_channels= 8, out_channels= 16, kernel_size= 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels= 16, out_channels= 16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )

        self.pool1 = nn.MaxPool2d(2,2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels= 16, kernel_size= (1 ,1)),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels= 16 , out_channels= 8, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.1)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels= 8, out_channels= 8, kernel_size= 3),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.1)
        )


        self.conv7 =nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels= 16, kernel_size= 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )

        self.gap1 = nn.Sequential(
            nn.AvgPool2d(kernel_size= 5),
            nn.Conv2d(in_channels= 16, out_channels= 10, kernel_size= (1,1)),
        ) 
        

    
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.gap1(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    

class Model_4(nn.Module):
    def __init__(self):
        super(Model_4, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels= 1, out_channels= 8, kernel_size= 3),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.1)         
        )

        self.conv2 =nn.Sequential(
            nn.Conv2d(in_channels= 8, out_channels= 16, kernel_size= 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels= 16, out_channels= 16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )

        self.pool1 = nn.MaxPool2d(2,2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels= 16, kernel_size= (1 ,1)),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels= 16 , out_channels= 8, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.1)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels= 8, out_channels= 8, kernel_size= 3),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.1)
        )


        self.conv7 =nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels= 16, kernel_size= 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )

        self.gap1 = nn.Sequential(
            nn.AvgPool2d(kernel_size= 5),
            nn.Conv2d(in_channels= 16, out_channels= 10, kernel_size= (1,1)),
        ) 
        

    
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.gap1(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)



####                                Assignment-6 Models                                              ######

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 16, 3)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 16, 3)
        self.bn3 = nn.BatchNorm2d(16)

        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(16,16,1)
        self.bn4 = nn.BatchNorm2d(16)
        
        self.conv5 = nn.Conv2d(16, 16, 3)
        self.bn5 = nn.BatchNorm2d(16)
        self.dp1 = nn.Dropout2d(0.01)

        self.conv6 = nn.Conv2d(16, 16, 3)
        self.bn6 = nn.BatchNorm2d(16)
        self.dp2 = nn.Dropout2d(0.01)

        self.conv7 = nn.Conv2d(16, 16 , 3)
        self.bn7 = nn.BatchNorm2d(16)
        self.dp3 = nn.Dropout2d(0.01)

        self.conv8 = nn.Conv2d(16, 16, 3, padding = 1)
        self.bn8 = nn.BatchNorm2d(16)
        self.dp4 = nn.Dropout2d(0.01)

        self.conv9 = nn.Conv2d(16, 32 ,3, padding = 1)
        self.bn9 = nn.BatchNorm2d(32)
        self.dp5 = nn.Dropout2d(0.01)

        
        self.gap = nn.AvgPool2d(kernel_size=5)
        self.linear = nn.Linear(32, 10)

    def forward(self, x):
        x = (self.bn1(F.relu(self.conv1(x))))
        x = (self.bn2(F.relu(self.conv2(x))))

        x = (self.pool1(self.bn3(F.relu(self.conv3(x)))))

        x = (self.bn4(F.relu(self.conv4(x))))

        x = self.dp1((self.bn5(F.relu(self.conv5(x)))))
        x = self.dp2((self.bn6(F.relu(self.conv6(x)))))
        x = self.dp3((self.bn7(F.relu(self.conv7(x)))))
        x = self.dp4((self.bn8(F.relu(self.conv8(x)))))
        x = self.dp5((self.bn9(F.relu(self.conv9(x)))))

        x = self.gap(x)
        x = x.view(-1, 32)
        x = self.linear(x)
        return F.log_softmax(x)