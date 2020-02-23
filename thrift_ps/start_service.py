import sys
sys.path.append("../")

from thrift_ps.ps_service import ttypes
from thrift_ps.ps_service import ParameterServer

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

from thrift_ps.server.ps_server import PSHandler
from thrift_ps.model.model_moniter import ModelMonitor
from thrift_ps import constants


if __name__ == "__main__":
    # PS handler
    handler = PSHandler()
    # Monitor thread of PS handler
    mm = ModelMonitor(handler, constants.INTERVAL, constants.EXPIRED_TIME)
    mm.start()
    # Start PS process
    processor = ParameterServer.Processor(handler)
    transport = TSocket.TServerSocket(host=constants.HOST, port=constants.PORT)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()
    # server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
    # You could do one of these for a multithreaded server
    server = TServer.TThreadedServer(processor, transport, tfactory, pfactory)

    print('Starting the server >>> IP = {}, PORT = {}'
          .format(constants.HOST, constants.PORT))
    server.serve()
    print('done.')
