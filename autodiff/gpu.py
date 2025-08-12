import torch
import timeit

# create some vectors

N = 1024*1024*8
a = torch.randn(N,  dtype=torch.float)
b = torch.randn(N,  dtype=torch.float)
c = torch.zeros(N,  dtype=torch.float)

print(f'vectors are length {N}')

def show_timing(fn, tag, repeat):
    print(f'benchmarking {tag} performance...')
    fn()   #fresh up any jit compilation, etc
    sec = timeit.timeit(fn, number=repeat)    
    print(f'{tag} ms: {sec*1000}')    

# verify an accelerator is available
# on macbook it is 'metal performance shaders'

print(f'{torch.backends.mps.is_available()=}')

# name the accelerator device
device = torch.device('mps')

# copy to the device

a1 = a.to(device)
b1 = b.to(device)
c1 = c.to(device)

def local_fn():
    c = a + b

def mps_fn():
    c1 = a1 + b1

show_timing(local_fn, 'local', 100)
show_timing(mps_fn, 'mps', 100)

# benchmark copying to device

def copy_fn():
    a.to(device)
    b.to(device)

show_timing(copy_fn, 'copy', 1000)

# report

if device.type == 'mps':
    print('Memory Usage:')
    print('Allocated:', round(torch.mps.current_allocated_memory()/1024**3,1), 'GB')
    print('Driver Memory:   ', round(torch.mps.driver_allocated_memory()/1024**3,1), 'GB')
    print('Cached:   ', round(torch.mps.recommended_max_memory()/1024**3,1), 'GB')
else:
    print(f"Device: XXX{device}XXX")

    
