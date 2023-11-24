import torch.optim as optim
import torch.nn as nn
from cnn import net

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)