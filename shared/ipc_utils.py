import zmq

def setup_publisher(context, port):
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    return socket

def setup_subscriber(context, host, port, topic_filter=""):
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://{host}:{port}")
    socket.setsockopt_string(zmq.SUBSCRIBE, topic_filter)
    return socket

def setup_requester(context, host, port):
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{host}:{port}")
    return socket

def setup_responder(context, port):
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{port}")
    return socket