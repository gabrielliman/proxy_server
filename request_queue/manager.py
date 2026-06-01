from threading import Lock

queue_lock = Lock()
queue_state = {}

def init_queue(backends):
    global queue_state
    queue_state = {url: 0 for url in backends}


def acquire_backend(url):
    with queue_lock:
        queue_state[url] = queue_state.get(url, 0) + 1
    return url


def release_backend(url):
    with queue_lock:
        if queue_state[url] > 0:
            queue_state[url] -= 1


def get_queue_state():
    with queue_lock:
        return queue_state.copy()
