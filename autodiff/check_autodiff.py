# Torch code to verify the ad.py examples

import torch
import torchviz

x1 = torch.tensor(2.0, requires_grad=True)
x2 = torch.tensor(3.0, requires_grad=True)
f = (x1 + x1 + x2)**2

print(f'{f=}')
f.backward()
print(f'{x1.grad=} {x2.grad=}')

# will also write out the computation graph
d = torchviz.make_dot(f, params=dict(x1=x1, x2=x2))
d.render()



