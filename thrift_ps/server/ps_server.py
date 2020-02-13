import sys
import threading
import time

import numpy as np

from thrift_ps.ps_service import ParameterServer
from thrift_ps.ps_service.ttypes import Model, Grad, Update, Operation, InvalidOperation


class PSHandler:

    def __init__(self):
        self.models = {}
        self.model_ts = {}
        self.model_parallelism = {}
        self.model_pull_count = {}
        self.model_push_count = {}
        self.w_lock = threading.Lock()

    def model_ids(self):
        return self.model_ts.keys()

    def delete_expired(self, lower):
        print("current model on PS >>> {}".format(self.models.keys()))
        print("current model parallelism on PS >>> {}".format(self.model_parallelism))
        print("current model pull count on PS >>> {}".format(self.model_pull_count))
        print("current model push count on PS >>> {}".format(self.model_push_count))
        for k, v in self.model_ts.items():
            if v <= lower:
                self.delete(k)

    def delete(self, mid):
        print("delete model {}".format(mid))
        self.models.pop(mid)
        self.model_ts.pop(mid)
        self.model_parallelism.pop(mid)
        self.model_pull_count.pop(mid)
        self.model_push_count.pop(mid)

    def ping(self):
        print('ping()')

    def register_model(self, mid, length, parallelism):
        self.models[mid] = np.random.rand(length)
        self.model_ts[mid] = time.time()
        self.model_parallelism[mid] = parallelism
        self.model_pull_count[mid] = 0
        self.model_push_count[mid] = 0
        print("register model >>> id = {}, length = {}, parallelism = {}"
              .format(mid, length, parallelism))

    def exist_model(self, mid):
        return self.models.__contains__(mid)

    def can_pull(self, mid, n_iter, worker_id):
        #print("worker {} ask can_pull >>> id = {}, iter = {}".format(worker_id, mid, n_iter))
        if self.model_push_count.__contains__(mid):
            return self.model_push_count[mid] == self.model_parallelism[mid] * n_iter
        else:
            x = InvalidOperation()
            x.whatOp = Operation.CAN_PULL
            x.why = 'No model {} in model_push_count on PS'.format(mid)
            raise x

    def can_push(self, mid, n_iter, worker_id):
        #print("worker {} ask can_push >>> id = {}, iter = {}".format(worker_id, mid, n_iter))
        if self.model_pull_count.__contains__(mid):
            return self.model_pull_count[mid] == self.model_parallelism[mid] * (n_iter+1)
        else:
            x = InvalidOperation()
            x.whatOp = Operation.CAN_PUSH
            x.why = 'No model {} in model_pull_count on PS'.format(mid)
            raise x

    def pull_model(self, mid, n_iter, worker_id):
        print("worker {} pull model >>> id = {}, iter = {}".format(worker_id, mid, n_iter))
        if self.models.__contains__(mid):
            model = Model()
            model.id = mid
            model.data = self.models[mid].tolist()
            model.length = len(model.data)
            self.model_pull_count[mid] = self.model_pull_count[mid] + 1
            return model
        else:
            x = InvalidOperation()
            x.whatOp = Operation.PULL_MODEL
            x.why = 'No model {} on PS'.format(mid)
            raise x

    def push_grad(self, grad):
        print("worker {} push grad >>> id = {}, lr = {}, n_iter = {}"
              .format(grad.worker_id, grad.id, grad.learning_rate, grad.n_iter))
        if self.models.__contains__(grad.id):
            self.w_lock.acquire()
            self.models[grad.id] = np.add(self.models.get(grad.id),
                                          np.multiply(grad.data, grad.learning_rate))
            self.model_push_count[grad.id] = self.model_push_count[grad.id] + 1
            self.w_lock.release()
        else:
            x = InvalidOperation()
            x.whatOp = Operation.PUSH_GRAD
            x.why = 'No model {} on PS'.format(grad.id)
            raise x

    def push_update(self, update):
        print("worker {} push update >>> id = {}, n_iter = {}"
              .format(update.worker_id, update.id, update.n_iter))
        if self.models.__contains__(update.id):
            self.w_lock.acquire()
            self.models[update.id] = np.add(self.models.get(update.id), update.data)
            self.model_push_count[update.id] = self.model_push_count[update.id] + 1
            self.w_lock.release()
        else:
            x = InvalidOperation()
            x.whatOp = Operation.PUSH_UPDATE
            x.why = 'No model {} on PS'.format(update.id)
            raise x
