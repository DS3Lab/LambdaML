

class Communicator(object):

    def __init__(self, __name):
        self.name = __name

    def reduce_batch(self, vector, **kwargs):
        """

        :param vector: merged tensor
        :param kwargs: other params
        :return:
        """
        return

    def reduce_epoch(self, vector, **kwargs):
        """

        :param vector: merged tensor
        :param kwargs: other params
        :return:
        """
        return


class S3Communicator(Communicator):

    def __init__(self, __name):
        super(S3Communicator, self).__init__()
        self.name = __name

    def reduce_batch(self, vector, **kwargs):
        return

    def reduce_epoch(self, vector, **kwargs):
        return