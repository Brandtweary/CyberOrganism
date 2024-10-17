import torch
import torch.multiprocessing as mp
import threading
from typing import Dict, Any, Callable, List
import time
import copy
import traceback
from shared.network_factory import create_neural_network
import queue
from collections import defaultdict

def batch_logs_defaultdict():
    return defaultdict(lambda: {'CRITICAL': [], 'ERROR': [], 'WARNING': [], 'INFO': [], 'DEBUG': [], 'METRICS': defaultdict(list)})


def _run_inference_process(registration_queue, inference_input_queue, inference_output_queue):
    organism_components = {}
    threads = {}
    results = {}
    organisms_processed = 0
    batch_size = 0
    lock = threading.Lock()
    batch_start = threading.Event()
    processing_complete = threading.Event()
    batch_complete = threading.Event()
    current_batch = {}
    threads_ready = {}  # New dict to track thread readiness
    batch_logs = batch_logs_defaultdict()  # New dict for logging

    def _organism_thread(organism_id: str, components: Dict[str, Any]):
        nonlocal results, organisms_processed, batch_size, current_batch, threads_ready, batch_logs

        # Import required modules
        dependencies = components['network_params'].get('select_action_dependencies', [])
        for module_name in dependencies:
            if '.' in module_name:
                module, attribute = module_name.rsplit('.', 1)
                globals()[attribute] = getattr(__import__(module, fromlist=[attribute]), attribute)
            else:
                globals()[module_name] = __import__(module_name)

        inference_buffer = components['inference_buffer']
        current_inference_buffer = components['current_inference_buffer']
        inference_buffer_lock = components['inference_buffer_lock']
        select_action = components['select_action']

        threads_ready[organism_id] = False
        threading.Timer(0.001, lambda: threads_ready.update({organism_id: True})).start()

        try:
            while True:
                batch_start.wait()
                
                if organism_id not in current_batch:
                    batch_logs[organism_id]['DEBUG'].append(f"Organism not in current batch")
                    batch_complete.wait()
                    continue

                start_time = time.perf_counter()  # Start timing
                
                state = current_batch[organism_id]
                if state is None:
                    break
                
                if not inference_buffer_lock.acquire(timeout=0.001):
                    batch_logs[organism_id]['WARNING'].append(f"Timeout while trying to acquire inference_buffer_lock for organism.")
                else:
                    try:
                        current_buffer = current_inference_buffer
                    finally:
                        inference_buffer_lock.release()
                
                try:
                    organism_results = select_action(inference_buffer[current_buffer], state, components)
                except Exception as e:
                    batch_logs[organism_id]['ERROR'].append(f"Error selecting action for organism: {str(e)}")
                    raise

                batch_logs[organism_id]['DEBUG'].append(f"Action selected: {organism_results['action_index']}")

                with lock:
                    results[organism_id] = organism_results
                    organisms_processed += 1
                    batch_logs[organism_id]['DEBUG'].append(f"Organisms processed: {organisms_processed}/{batch_size}")
                    end_time = time.perf_counter()
                    total_time_ms = (end_time - start_time) * 1000
                    batch_logs[organism_id]['METRICS']['inference_time_ms'].append(total_time_ms)
                    if organisms_processed == batch_size:
                        processing_complete.set()

        except Exception as e:
            batch_logs[organism_id]['ERROR'].append(f"Error in organism thread: {str(e)}")
            traceback.print_exc()
            raise  # Re-raise the exception to stop execution

    def _batch_processing_thread():
        nonlocal results, organisms_processed, batch_size, current_batch, threads_ready, batch_logs

        try:
            while True:
                batch = inference_input_queue.get()
                start_time = time.perf_counter()
                if batch is None:
                    break

                with lock:
                    results.clear()
                    organisms_processed = 0
                    batch_size = len(batch)
                    current_batch = batch

                processing_complete.clear()
                batch_complete.clear()
                
                batch_start.set()

                # Wait for threads to be ready, with a timeout
                wait_start = time.perf_counter()
                unready_organisms = set(batch.keys())
                while unready_organisms and (time.perf_counter() - wait_start) < 0.003:  # 3ms timeout
                    unready_organisms = {org_id for org_id in unready_organisms if not threads_ready.get(org_id, False)}
                    if unready_organisms:
                        time.sleep(0.0001)  # Short sleep to prevent busy-waiting

                # Remove unready organisms from the batch and log warnings
                with lock:
                    for org_id in unready_organisms:
                        batch_logs[org_id]['WARNING'].append(f"Thread for organism not ready in time. Removed from batch.")
                        del current_batch[org_id]
                    batch_size = len(current_batch)

                batch_start.clear()  # Clear after all possible threads have started

                if not current_batch:
                    inference_output_queue.put(({}, dict(batch_logs)))
                    batch_logs.clear()
                    batch_complete.set()
                    continue

                if not processing_complete.wait(timeout=0.03):
                    batch_logs['batch']['WARNING'].append("Inference batch processing took longer than 30 milliseconds. Stopped prematurely.")
                    with lock:
                        inference_output_queue.put((results, dict(batch_logs)))
                    batch_logs.clear()
                    batch_complete.set()
                    continue

                if not results:
                    raise ValueError("No results processed for batch")
                
                end_time = time.perf_counter()
                batch_processing_time_ms = (end_time - start_time)
                batch_logs['batch']['METRICS']['inference_batch_processing'].append(batch_processing_time_ms)

                inference_output_queue.put((results, dict(batch_logs)))
 
                batch_logs.clear()
                batch_complete.set()

        except Exception as e:
            batch_logs['batch']['ERROR'].append(f"Error in batch processing thread: {str(e)}")
            traceback.print_exc()
            raise  # Re-raise the exception to stop execution

    batch_thread = threading.Thread(target=_batch_processing_thread)
    batch_thread.start()

    while True:
        try:
            if not registration_queue.empty():
                message = registration_queue.get()
                if message[0] == 'register':
                    _, organism_id, network_params = message
                    inference_network = _setup_inference_network(network_params)
                    inference_buffer = [inference_network, copy.deepcopy(inference_network)]
                    current_inference_buffer = 0
                    inference_buffer_lock = threading.Lock()

                    organism_components[organism_id] = {
                        'inference_network': inference_network,
                        'inference_buffer': inference_buffer,
                        'current_inference_buffer': current_inference_buffer,
                        'inference_buffer_lock': inference_buffer_lock,
                        'select_action': network_params['select_action'],
                        'network_params': network_params
                    }

                    thread = threading.Thread(
                        target=_organism_thread,
                        args=(organism_id, organism_components[organism_id])
                    )
                    thread.start()
                    threads[organism_id] = thread

                elif message[0] == 'update_weights':
                    _, organism_id, weights = message
                    if organism_id in organism_components:
                        _update_inference_network(organism_components[organism_id], weights)

                elif message[0] == 'terminate':
                    with lock:
                        for org_id, thread in threads.items():
                            current_batch[org_id] = None
                    batch_start.set()
                    batch_complete.set()
                    for thread in threads.values():
                        thread.join()
                    inference_input_queue.put(None)
                    batch_thread.join()
                    batch_start.clear()
                    batch_complete.clear()
                    break
            else:
                time.sleep(0.01)
        except Exception as e:
            print(f"Error in main inference process loop: {str(e)}")
            traceback.print_exc()
            # Terminate all threads
            with lock:
                for org_id in threads:
                    current_batch[org_id] = None
            batch_start.set()
            batch_complete.set()
            for thread in threads.values():
                thread.join()
            inference_input_queue.put(None)
            batch_thread.join()
            batch_start.clear()
            batch_complete.clear()
            raise  # Re-raise the exception to stop the process

def _setup_inference_network(network_params):
    inference_network = create_neural_network(network_params)
    inference_network.eval()
    return inference_network

def _update_inference_network(components, weights):
    new_inference_network = components['inference_buffer'][1 - components['current_inference_buffer']]
    new_inference_network.load_state_dict(weights)
    new_inference_network.eval()
    with components['inference_buffer_lock']:
        components['current_inference_buffer'] = 1 - components['current_inference_buffer']

class InferenceProcess:
    def __init__(self):
        self.registration_queue = mp.Queue()
        self.inference_input_queue = mp.Queue()
        self.inference_output_queue = mp.Queue()
        self.process = None

    def start(self):
        self.process = mp.Process(
            target=_run_inference_process,
            args=(self.registration_queue, self.inference_input_queue, self.inference_output_queue)
        )
        self.process.start()

    def register_organism(self, organism_id: str, network_params: Dict[str, Any]):
        if not isinstance(organism_id, str):
            raise ValueError(f"Error: organism_id must be a string, got {type(organism_id).__name__}")
        self.registration_queue.put(('register', organism_id, network_params))

    def update_weights(self, organism_id: str, weights: Dict[str, torch.Tensor]):
        if not isinstance(organism_id, str):
            raise ValueError(f"Error: organism_id must be a string, got {type(organism_id).__name__}")
        self.registration_queue.put(('update_weights', organism_id, weights))

    def add_to_inference_queue(self, state_dict: Dict[str, torch.Tensor]):
        self.inference_input_queue.put(state_dict)

    def get_inference_results(self):
        try:
            results, batch_logs = self.inference_output_queue.get(timeout=6)
            return results, batch_logs
        except queue.Empty:
            print("Warning: Inference results retrieval timed out.")
            return {}, {}

    def stop(self):
        self.registration_queue.put(('terminate',))
        self.process.join(timeout=5)
        if self.process.is_alive():
            print(f"Warning: Inference process did not terminate within the timeout period")
            self.process.terminate()
