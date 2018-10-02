# This file is used for memory profiling of the MXNet `LSTMCell` layer.

import mxnet as mx

B, H, prefix = 128, 1000, 'lstm_'

inputs   = mx.sym.var('inputs')
states_h = mx.sym.var('states_h')
states_c = mx.sym.var('states_c')

lstm = mx.rnn.LSTMCell(num_hidden=H, prefix=prefix)

outputs, states = lstm(inputs=inputs, states=[states_h, states_c])

output = outputs
# output = mx.sym.Group(states)

args, args_grad, grad_req = {}, {}, {}

for placeholder in ['inputs', 'states_h', 'states_c']:
    args     [placeholder] = mx.nd.zeros(shape=(B, H), ctx=mx.gpu(), dtype='float32')
    args_grad[placeholder] = mx.nd.zeros(shape=(B, H), ctx=mx.gpu(), dtype='float32')
    grad_req [placeholder] = 'write'

for weight in ['%si2h_weight' % prefix,
               '%sh2h_weight' % prefix]:
    args     [weight] = mx.nd.zeros(shape=(4 * H, H), ctx=mx.gpu(), dtype='float32')
    args_grad[weight] = mx.nd.zeros(shape=(4 * H, H), ctx=mx.gpu(), dtype='float32')
    grad_req [weight] = 'write'

for bias in ['%si2h_bias' % prefix,
             '%sh2h_bias' % prefix]:
    args     [bias] = mx.nd.zeros(shape=(4 * H,), ctx=mx.gpu(), dtype='float32')
    args_grad[bias] = mx.nd.zeros(shape=(4 * H,), ctx=mx.gpu(), dtype='float32')
    grad_req [bias] = 'write'

texec = output.bind(ctx=mx.gpu(), args=args, args_grad=args_grad, grad_req=grad_req)

texec. forward( is_train=True)
# texec.backward(out_grads=[mx.nd.zeros(shape=(B, H), ctx=mx.gpu(), dtype='float32'),
#                           mx.nd.zeros(shape=(B, H), ctx=mx.gpu(), dtype='float32')])
texec.backward(out_grads=[mx.nd.zeros(shape=(B, H), ctx=mx.gpu(), dtype='float32')])
mx.nd.waitall()
