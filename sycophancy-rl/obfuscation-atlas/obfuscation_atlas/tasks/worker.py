import contextlib
import os
import queue
import resource
import signal

import psutil


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException()


# Result id a worker sends once it is up (spawned workers first re-import the parent's main
# module, which can take tens of seconds).
READY_TASK_ID = -1


def _persistent_worker(task_queue, result_queue):
    """Worker that limits execution time and memory usage and stays alive across all batches."""
    signal.signal(signal.SIGALRM, timeout_handler)
    result_queue.put((READY_TASK_ID, True))

    process = psutil.Process(os.getpid())
    baseline = process.memory_info().vms / (1024**3)
    print(f"[Worker Debug] Baseline Virtual Memory (VMS): {baseline:.2f} GB")

    try:
        task_allowance = 4 * 1024 * 1024 * 1024  # 4GB
        # 28GB (from wandb when running RL) + 4GB = 32GB limit
        total_limit = int(process.memory_info().vms + task_allowance)

        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (total_limit, hard))
    except (ValueError, OSError):  # setting memory limit does not work on macos
        print("WARNING: Failed to set memory limit. Memory limit will not be enforced.")

    while True:
        try:
            task_id, code_string, timeout_seconds = task_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        success = False
        try:
            signal.alarm(timeout_seconds)
            try:
                with open(os.devnull, "w") as devnull:
                    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                        exec(code_string, {})
                success = True
            except BaseException:
                # Any failure of the executed program: syntax errors, the alarm's TimeoutException,
                # and SystemExit from exit()/sys.exit(), which is not an Exception and would
                # otherwise end this worker.
                success = False
            finally:
                signal.alarm(0)
        except TimeoutException:
            success = False
        except Exception as e:
            print(f"[CodeExecutor Worker] Exception happened in worker harness: {e}")
            success = False

        result_queue.put((task_id, success))
