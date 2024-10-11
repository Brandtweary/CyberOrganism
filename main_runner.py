#!/usr/bin/env python3

import subprocess
import sys
import time
import zmq
import os

def main():
    # Number of learner processes
    num_learners = 4

    # Paths to the interpreters
    simulation_python = os.path.abspath('environments/simulation_env/Scripts/pypy3.exe')
    gui_python = os.path.abspath('environments/gui_env/Scripts/python.exe')
    learner_python = os.path.abspath('environments/learner_env/Scripts/python.exe')

    # Start the PyPy simulation process
    simulation_process = subprocess.Popen(
        [simulation_python, 'simulation/run_simulation.py']
    )

    # Start the CPython GUI process
    gui_process = subprocess.Popen(
        [gui_python, 'gui/run_gui.py']
    )

    # Start learner processes
    learner_processes = []
    for i in range(num_learners):
        p = subprocess.Popen(
            [learner_python, 'learner/run_learner.py', '--learner-id', str(i)]
        )
        learner_processes.append(p)

    try:
        # Main loop can handle monitoring or other tasks
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")

    # Terminate all processes
    simulation_process.terminate()
    gui_process.terminate()
    for p in learner_processes:
        p.terminate()

if __name__ == '__main__':
    main()