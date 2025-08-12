# Simple example of reverse-mode autodifferentiation

import collections
import math

# A Wengert list
PROGRAM1 = [('z1', 'add', ['x1', 'x1']),
           ('z2', 'add', ['z1', 'x2']),
           ('f', 'square', ['z2'])]

PROGRAM = [('z', 'square', ['x']),
           ('f', 'ln', ['z'])]

# Function library
G = {
    'add':      lambda x1,x2: x1 + x2,
    'square':   lambda x: x * x,
    'ln':       lambda x: math.log(x)
}

# Partial derivatives of each function wrt each input
DG = {
    'add': [lambda x1,x2: 1, lambda x1,x2: 1],
    'square': [lambda x: 2.0 * x],
    'ln':  [lambda x: 1/x],
}


def forward(opseq, val):
    for output, fn_name, inputs in opseq:
        input_vals = [val[inp] for inp in inputs]
        fn = G[fn_name]
        val[output] = fn(*input_vals)
        print(f'val[{output}]={val[output]}')

def backward(opseq, val, delta):
    for output, fn_name, inputs in reversed(opseq):
        input_vals = [val[inp] for inp in inputs]
        print(f'{output=} {fn_name=} {inputs=}')
        for i in range(len(inputs)):
            xi = inputs[i]
            dxi_dinputi_fn = DG[fn_name][i]
            dy_dxi = delta[output] * dxi_dinputi_fn(*input_vals)
            delta[xi] += dy_dxi
            print(f'delta[{xi}] += {dy_dxi}')

if __name__ == '__main__':
    #val = dict(x1=2, x2=3)
    val = dict(x=4)
    forward(PROGRAM, val) 
    print(f'{val=}')
    delta = collections.defaultdict(float)
    delta['f'] = 1.0
    backward(PROGRAM, val, delta)
    print(f'{delta=}')

    
