import torch
import time

print(torch.cuda.is_available())

for i in range(100):
    print(i)
    time.sleep(0.5)
