import pickle
import zmq
import multiprocessing as mp


class WorkerProcess:
    def __init__(self, worker_class, port):
        self.worker_class = worker_class
        self.port = port

    def start(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://*:{self.port}")
        worker = pickle.loads(self.worker_class)()
        while True:
            message = socket.recv_pyobj()
            print('recv', self.port, message)
            if message == "stop":
                break
            result = worker.step(*message)
            socket.send_pyobj(result)
        socket.close()
        context.term()


class WorkerPool:
    def __init__(self, worker_class, num_workers):
        self.worker_class = pickle.dumps(worker_class)
        self.num_workers = num_workers
        self.workers = []
        self.running = False
        self.worker_ports = []
        self.worker_sockets = []
        self.start_port = 6000

    def start(self):
        self.running = True
        for i in range(self.num_workers):
            port = self.start_port + i
            worker = WorkerProcess(self.worker_class, port)
            worker_process = mp.Process(target=worker.start)
            worker_process.start()
            self.workers.append(worker_process)
            self.worker_ports.append(port)

            context = zmq.Context()
            socket = context.socket(zmq.DEALER)
            socket.connect(f"tcp://localhost:{port}")
            self.worker_sockets.append(socket)

    def stop(self):
        self.running = False
        for socket in self.worker_sockets:
            socket.send(b"", zmq.SNDMORE)
            socket.send_pyobj("stop")
        for worker in self.workers:
            worker.join()

    def submit(self, num_reqs: int, *args):
        for i in range(num_reqs):
            socket = self.worker_sockets[i % self.num_workers]
            socket.send(b"", zmq.SNDMORE)
            socket.send_pyobj(args)

    def get_results(self, num_reqs: int):
        results = []
        for i in range(num_reqs):
            socket = self.worker_sockets[i % self.num_workers]
            socket.recv()
            result = socket.recv_pyobj()
            results.append(result)
        return results


class MyWorker:
    def __init__(self):
        pass

    def step(self, x):
        return x ** 2


def main():
    pool = WorkerPool(MyWorker, num_workers=5)
    pool.start()
    results = []
    pool.submit(10, 2)
    results = pool.get_results(10)
    print(results)
    pool.stop()


if __name__ == "__main__":
    main()
