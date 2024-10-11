import torch.multiprocessing as mp
from multiprocessing import Manager
import threading
from typing import Dict, Any
import random
import time


def _run_process(registration_queue, manager_dict):
    from network_factory import setup_network_architecture

    organism_components = {}
    threads = {}

    while True:
        # Process registration messages
        if not registration_queue.empty():
            message = registration_queue.get()
            if message[0] == 'register':
                _, organism_id, network_params = message
                # Set up architecture for the organism
                architecture = setup_network_architecture(network_params)
                learn_input_queue = manager_dict[f'{organism_id}_input_queue']
                learn_output_queue = manager_dict[f'{organism_id}_output_queue']

                organism_components[organism_id] = {
                    'learn_input_queue': learn_input_queue,
                    'learn_output_queue': learn_output_queue,
                    'architecture': architecture,
                    'network_params': network_params
                }

                # Start a thread for this organism
                thread = threading.Thread(
                    target=_organism_thread,
                    args=(organism_id, organism_components[organism_id], manager_dict)
                )
                thread.start()
                threads[organism_id] = thread

            elif message[0] == 'terminate':
                # Terminate all organism threads
                for org_id, thread in threads.items():
                    organism_components[org_id]['learn_input_queue'].put(None)
                    thread.join()
                break
        else:
            time.sleep(0.01)

def _organism_thread(organism_id: str, components: Dict[str, Any], manager_dict: Dict[str, Any]):
    import torch
    import torch.nn.functional as F  # used in learn function

    learn_input_queue = components['learn_input_queue']
    learn_output_queue = components['learn_output_queue']
    architecture = components['architecture']
    learn = architecture['learn']

    try:
        while True:
            if not learn_input_queue.empty():
                data = learn_input_queue.get()
                if data is None:
                    break

                organism_state, experiences, training_steps = data

                metrics, weights, td_errors = learn(
                    organism_state=organism_state,
                    experiences=experiences,
                    training_steps=training_steps,
                    architecture=architecture
                )

                learn_output_queue.put((metrics, weights, td_errors))
            else:
                time.sleep(0.01)
    finally:
        # Clean up resources
        del architecture
        del components
        
        # Remove organism queues from manager_dict
        manager_dict.pop(f'{organism_id}_input_queue', None)
        manager_dict.pop(f'{organism_id}_output_queue', None)

class ProcessPool:
    def __init__(self, num_processes: int):
        self.num_processes = num_processes

        self.manager = Manager()
        self.manager_dict = self.manager.dict()

        self.processes = {}
        for idx in range(self.num_processes):
            self.processes[idx] = {
                'process': None,
                'registration_queue': None,
                'organisms': set()
            }

        self.organism_learn_input_queues = {}
        self.organism_learn_output_queues = {}

    def start(self):
        for idx in range(self.num_processes):
            registration_queue = self.manager.Queue()
            process = mp.Process(target=_run_process, args=(registration_queue, self.manager_dict))
            self.processes[idx]['process'] = process
            self.processes[idx]['registration_queue'] = registration_queue

            process.start()

    def register_organism(self, organism_id: str, network_params: Dict[str, Any]):
        min_count = min(len(p['organisms']) for p in self.processes.values())
        candidates = [idx for idx, p in self.processes.items() if len(p['organisms']) == min_count]
        process_index = random.choice(candidates)

        self.processes[process_index]['organisms'].add(organism_id)

        learn_input_queue = self.manager.Queue()
        learn_output_queue = self.manager.Queue()
        self.organism_learn_input_queues[organism_id] = learn_input_queue
        self.organism_learn_output_queues[organism_id] = learn_output_queue

        self.manager_dict[f'{organism_id}_input_queue'] = learn_input_queue
        self.manager_dict[f'{organism_id}_output_queue'] = learn_output_queue

        self.processes[process_index]['registration_queue'].put((
            'register',
            organism_id,
            network_params
        ))

    def get_organism_queues(self, organism_id: str):
        learn_input_queue = self.organism_learn_input_queues.get(organism_id)
        learn_output_queue = self.organism_learn_output_queues.get(organism_id)
        return learn_input_queue, learn_output_queue

    def cleanup_organism(self, organism_id: str):
        for process_data in self.processes.values():
            if organism_id in process_data['organisms']:
                process_data['organisms'].remove(organism_id)
                learn_input_queue = self.organism_learn_input_queues.pop(organism_id, None)
                self.organism_learn_output_queues.pop(organism_id, None)
                if learn_input_queue is not None:
                    learn_input_queue.put(None)
                break

    def stop(self):
        # Send terminate signal to all processes
        for process_data in self.processes.values():
            registration_queue = process_data['registration_queue']
            try:
                registration_queue.put(('terminate',))
            except BrokenPipeError:
                print(f"Warning: BrokenPipeError when sending terminate signal to process {process_data['process'].pid}")

        # Join all processes
        for process_data in self.processes.values():
            process = process_data['process']
            process.join(timeout=5)  # Add a timeout to avoid hanging indefinitely
            if process.is_alive():
                print(f"Warning: Process {process.pid} did not terminate within the timeout period")
                process.terminate()

        # Shutdown the manager
        self.manager.shutdown()
    
    def get_load(self):
        return [len(process['organisms']) for process in self.processes.values()]
