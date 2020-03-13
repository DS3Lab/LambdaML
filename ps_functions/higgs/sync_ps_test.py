import time
import numpy as np

from thrift_ps.ps_service import ParameterServer
from thrift_ps.client import ps_client

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

from thrift_ps import constants

# algorithm setting
NUM_EPOCHS = 10
NUM_BATCHES = 10
MODEL_NAME = "w.b"
MODEL_LENGTH = 472360
LEARNING_RATE = 0.01
RANDOM_SEED = 42


def handler(event, context):
    start_time = time.time()
    bucket = event['bucket_name']
    worker_index = event['rank']
    num_workers = event['num_workers']
    key = event['file']

    print('bucket = {}'.format(bucket))
    print('number of workers = {}'.format(num_workers))
    print('worker index = {}'.format(worker_index))
    print("file = {}".format(key))

    # Set thrift connection
    # Make socket
    transport = TSocket.TSocket(constants.HOST, constants.PORT)
    # Buffering is critical. Raw sockets are very slow
    transport = TTransport.TBufferedTransport(transport)
    # Wrap in a protocol
    protocol = TBinaryProtocol.TBinaryProtocol(transport)
    # Create a client to use the protocol encoder
    t_client = ParameterServer.Client(protocol)
    # Connect!
    transport.open()

    # test thrift connection
    ps_client.ping(t_client)
    print("create and ping thrift server >>> HOST = {}, PORT = {}"
          .format(constants.HOST, constants.PORT))

    # register model
    ps_client.register_model(t_client, worker_index, MODEL_NAME, MODEL_LENGTH, num_workers)
    ps_client.exist_model(t_client, MODEL_NAME)
    print("register and check model >>> name = {}, length = {}".format(MODEL_NAME, MODEL_LENGTH))

    # Training the Model
    train_start = time.time()
    iter_counter = 0
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        for batch_index in range(NUM_BATCHES):
            print("------worker {} epoch {} batch {}------"
                  .format(worker_index, epoch, batch_index))
            batch_start = time.time()

            loss = 0.0

            # pull latest model
            ps_client.can_pull(t_client, MODEL_NAME, iter_counter, worker_index)
            latest_model = ps_client.pull_model(t_client, MODEL_NAME, iter_counter, worker_index)

            w_b_grad = np.random.rand(1, MODEL_LENGTH).astype(np.double).flatten()
            cal_time = time.time() - batch_start

            # push gradient to PS
            sync_start = time.time()
            ps_client.can_push(t_client, MODEL_NAME, iter_counter, worker_index)
            ps_client.push_grad(t_client, MODEL_NAME, w_b_grad, LEARNING_RATE, iter_counter, worker_index)
            ps_client.can_pull(t_client, MODEL_NAME, iter_counter + 1, worker_index)  # sync all workers
            sync_time = time.time() - sync_start

            print('Epoch: [%d/%d], Step: [%d/%d] >>> Time: %.4f, Loss: %.4f, epoch cost %.4f, '
                  'batch cost %.4f s: cal cost %.4f s and communication cost %.4f s'
                  % (epoch + 1, NUM_EPOCHS, batch_index, NUM_BATCHES,
                     time.time() - train_start, loss, time.time() - epoch_start,
                     time.time() - batch_start, cal_time, sync_time))
            iter_counter += 1

    end_time = time.time()
    print("Elapsed time = {} s".format(end_time - start_time))
