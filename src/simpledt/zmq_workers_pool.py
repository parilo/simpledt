import pickle
import zmq
import multiprocessing as mp


class WorkerProcess:
    def __init__(self, worker_class, worker_args, port):
        self.worker_class = worker_class
        self.worker_args = worker_args
        self.port = port

    def start(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://*:{self.port}")
        worker = pickle.loads(self.worker_class)(**self.worker_args)
        while True:
            step_args = socket.recv_pyobj()
            if step_args == "stop":
                break
            result = worker.step(*step_args)
            socket.send_pyobj(result)
        socket.close()
        context.term()


class WorkerPool:
    def __init__(self, worker_class, worker_args, num_workers):
        self.worker_class = pickle.dumps(worker_class)
        self.worker_args = worker_args
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
            worker = WorkerProcess(self.worker_class, self.worker_args, port)
            mpctx = mp.get_context('spawn')
            worker_process = mpctx.Process(target=worker.start)
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

    def request(self, num_reqs: int, *args):
        self.submit(num_reqs, *args)
        return self.get_results(num_reqs)


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
