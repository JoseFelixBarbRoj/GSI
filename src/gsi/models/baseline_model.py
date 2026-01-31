from torch import nn

class BaselineModel(nn.Module):
    def __init__(self, in_channels, num_classes, height = 224, width = 224):
        super().__init__()   
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * (height // 4) * (width // 4), num_classes)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu(self.conv1(x))  
        x = self.pool(x)           
        x = self.relu(self.conv2(x))  
        x = self.pool(x)           
        x = self.flatten(x)
        x = self.fc1(x)            
        return x
