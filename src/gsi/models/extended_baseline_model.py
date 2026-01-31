from torch import nn

class ExtendedBaselineModel(nn.Module):
    def __init__(self, in_channels, num_classes, height = 224, width = 224):
        super().__init__()   
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * (height // 8) * (width // 8), 512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.1)


    def forward(self, x):
        x = self.relu(self.conv1(x))  
        x = self.pool(x)           
        x = self.relu(self.conv2(x))  
        x = self.pool(x)
        x = self.relu(self.conv3(x))  
        x = self.pool(x)             
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)   
        x = self.fc2(x)         
        return x