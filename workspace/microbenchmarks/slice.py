# This file is used for memory profiling of the MXNet `slice` layer.

import mxnet as mx

G, B, H = 4, 80, 1000

x = mx.sym.var('x')
y = mx.sym.split(x, num_outputs=4, axis=0)
z = y[0] + y[1] + y[2] + y[3]
z = mx.sym.reshape(z, shape=(B, H))

texec = z.bind(ctx=mx.gpu(), 
               args     ={'x': mx.nd.ones(shape=(G, B, H), ctx=mx.gpu(), dtype='float32')},
               args_grad={'x': mx.nd.ones(shape=(G, B, H), ctx=mx.gpu(), dtype='float32')})

texec. forward( is_train=True)
texec.backward(out_grads=mx.nd.ones(shape=(B, H)))
mx.nd.waitall()