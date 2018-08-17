"""Directly calling this module; apparently from another node.
Calls should come as

python -m amp.model id hostname:port

This session will then start a zmq session with that socket, labeling
itself with id. Instructions on what to do will come from the socket.
"""
import sys
import tempfile
import zmq

from ..utilities import MessageDictionary, string2dict, Logger
from .. import importhelper


hostsocket = sys.argv[-1]
proc_id = sys.argv[-2]
msg = MessageDictionary(proc_id)

# Send standard lines to stdout signaling process started and where
# error is directed.
print('<amp-connect>')  # Signal that program started.
sys.stderr = tempfile.NamedTemporaryFile(mode='w', delete=False,
                                         suffix='.stderr')
print('Log and stderr written to %s<stderr>' % sys.stderr.name)

# Also send logger output to stderr to aid in debugging.
log = Logger(file=sys.stderr)

# Establish client session via zmq; find purpose.
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect('tcp://%s' % hostsocket)
socket.send_pyobj(msg('<purpose>'))
purpose = socket.recv_string()

if purpose == 'calculate_loss_function':
    # Request variables.
    socket.send_pyobj(msg('<request>', 'fortran'))
    fortran = socket.recv_pyobj()
    socket.send_pyobj(msg('<request>', 'modelstring'))
    modelstring = socket.recv_pyobj()
    dictionary = string2dict(modelstring)
    Model = importhelper(dictionary.pop('importname'))
    log('Model received:')
    log(str(dictionary))
    model = Model(fortran=fortran, **dictionary)
    model.log = log
    log('Model set up.')

    socket.send_pyobj(msg('<request>', 'args'))
    args = socket.recv_pyobj()
    d = args['d']
    socket.send_pyobj(msg('<request>', 'lossfunctionstring'))
    lossfunctionstring = socket.recv_pyobj()
    dictionary = string2dict(lossfunctionstring)
    log(str(dictionary))
    LossFunction = importhelper(dictionary.pop('importname'))
    lossfunction = LossFunction(parallel={'cores': 1},
                                raise_ConvergenceOccurred=False,
                                d=d, **dictionary)
    log('Loss function set up.')

    images = None
    socket.send_pyobj(msg('<request>', 'images'))
    images = socket.recv_pyobj()
    log('Images received.')

    fingerprints = None
    socket.send_pyobj(msg('<request>', 'fingerprints'))
    fingerprints = socket.recv_pyobj()
    log('Fingerprints received.')

    fingerprintprimes = None
    socket.send_pyobj(msg('<request>', 'fingerprintprimes'))
    fingerprintprimes = socket.recv_pyobj()
    log('Fingerprintprimes received.')

    # Set up local loss function.
    lossfunction.attach_model(model,
                              fingerprints=fingerprints,
                              fingerprintprimes=fingerprintprimes,
                              images=images)
    log('Images, fingerprints, and fingerprintprimes '
        'attached to the loss function.')

    if model.fortran:
        log('fmodules will be used to evaluate loss function.')
    else:
        log('Fortran will not be used to evaluate loss function.')
    # Now wait for parameters, and send the component of the loss function.
    while True:
        socket.send_pyobj(msg('<request>', 'parameters'))
        parameters = socket.recv_pyobj()
        if parameters == '<stop>':
            # FIXME/ap: I removed an fmodules.deallocate_variables() call
            # here. Do we need to add this to LossFunction?
            break
        elif parameters == '<continue>':
            # Master is waiting for other workers to finish.
            # Any more elegant way
            # to do this without opening another comm channel?
            # or having a thread for each process?
            pass
        else:
            # FIXME/ap: Why do we need to request this every time?
            # Couldn't it be part of earlier request?
            socket.send_pyobj(msg('<request>', 'args'))
            args = socket.recv_pyobj()
            lossprime = args['lossprime']
            output = lossfunction.get_loss(parameters,
                                           lossprime=lossprime)

            socket.send_pyobj(msg('<result>', output))
            socket.recv_string()

else:
    raise NotImplementedError('Purpose "%s" unknown.' % purpose)
