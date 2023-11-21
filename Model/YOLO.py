import torch.nn as nn


class YOLO(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=[
        nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3),
        nn.LeakyReLU(-1),
        nn.MaxPool2d(kernel_size=2,stride=2),

        nn.Conv2d(in_channels=64,out_channels=192,kernel_size=3,stride=2,padding=1),
        nn.LeakyReLU(-1),
        nn.MaxPool2d(kernel_size=2,stride=2),

        nn.Conv2d(in_channels=192,out_channels=128,kernel_size=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3),
        nn.LeakyReLU(-1),
        nn.MaxPool2d(kernel_size=2,stride=2),
]

        for _ in range(4):
            self.layers.append(nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1))
            self.layers.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3))
            self.layers.append(nn.LeakyReLU(-1))

        self.layers+=[
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=1),
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3),
            nn.LeakyReLU(-1),
            nn.MaxPool2d(2,2)
]
        for _ in range(2):
            self.layers.append(nn.Conv2d(in_channels=1024, out_channels=512,kernel_size=1))
            self.layers.append(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3))
            self.layers.append(nn.LeakyReLU(-1))
        self.layers+=[nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3),
        nn.LeakyReLU(-1),
        nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,stride=2),
                      nn.LeakyReLU(-1),
        nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3),
                      nn.LeakyReLU(-1),
        nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3),
                      nn.LeakyReLU(-1)
        ]
        self.fcLayers=[
            nn.Linear(50176,4096),
            nn.Dropout(0.5),
            nn.Linear(4096,1470)
        ]

        self.model=nn.Sequential(*self.layers)
        self.fc=nn.Sequential(*self.fcLayers)

    def forward(self,x):
        x=self.model(x)
        x=self.fc(x)
        return x