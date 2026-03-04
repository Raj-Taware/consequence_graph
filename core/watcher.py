"""
Live file watcher.
Uses watchdog to monitor Python file changes and incrementally re-index.
Patches only the changed file's nodes/edges — no full rebuild.
"""
import time
import os
import threading
from typing import Optional

from .graph import KnowledgeGraph
from .indexer import Indexer
from .enricher import Enricher

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False


class _CodeChangeHandler:
    """Handles file system events and triggers incremental re-index."""

    # Debounce window in seconds — batch rapid saves (e.g. editor auto-save)
    DEBOUNCE = 0.8

    def __init__(self, graph: KnowledgeGraph, root_path: str, verbose: bool = True):
        self.graph = graph
        self.root_path = root_path
        self.verbose = verbose
        self._pending: dict[str, float] = {}  # path → timestamp
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None

    def on_modified(self, path: str):
        self._schedule(path)

    def on_created(self, path: str):
        self._schedule(path)

    def on_deleted(self, path: str):
        if path.endswith(".py"):
            self.graph.remove_nodes_for_file(path)
            self.graph.save()
            if self.verbose:
                print(f"[codegraph] 🗑  Removed nodes for deleted file: {os.path.relpath(path, self.root_path)}")

    def _schedule(self, path: str):
        if not path.endswith(".py"):
            return
        with self._lock:
            self._pending[path] = time.time()
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(self.DEBOUNCE, self._flush)
            self._timer.daemon = True
            self._timer.start()

    def _flush(self):
        with self._lock:
            paths = list(self._pending.keys())
            self._pending.clear()

        if not paths:
            return

        indexer = Indexer(self.graph, self.root_path)
        for path in paths:
            if os.path.exists(path):
                indexer.reindex_file(path)
                if self.verbose:
                    rel = os.path.relpath(path, self.root_path)
                    print(f"[codegraph] ♻  Re-indexed: {rel} "
                          f"({self.graph.node_count()} nodes, {self.graph.edge_count()} edges)")

        # Re-run enricher after patch
        Enricher(self.graph).run()
        self.graph.save()


if WATCHDOG_AVAILABLE:
    from watchdog.events import FileSystemEventHandler

    class _WatchdogBridge(FileSystemEventHandler):
        def __init__(self, handler: _CodeChangeHandler):
            self.handler = handler

        def on_modified(self, event):
            if not event.is_directory:
                self.handler.on_modified(event.src_path)

        def on_created(self, event):
            if not event.is_directory:
                self.handler.on_created(event.src_path)

        def on_deleted(self, event):
            if not event.is_directory:
                self.handler.on_deleted(event.src_path)


class Watcher:
    """
    Start a background watchdog observer on a directory.
    Incrementally patches the graph on file changes.
    """

    def __init__(self, graph: KnowledgeGraph, watch_path: str, verbose: bool = True):
        if not WATCHDOG_AVAILABLE:
            raise RuntimeError(
                "watchdog is not installed. Run: pip install watchdog"
            )
        self.graph = graph
        self.watch_path = os.path.abspath(watch_path)
        self.verbose = verbose
        self._handler = _CodeChangeHandler(graph, watch_path, verbose)
        self._observer = Observer()

    def start(self):
        bridge = _WatchdogBridge(self._handler)
        self._observer.schedule(bridge, self.watch_path, recursive=True)
        self._observer.start()
        if self.verbose:
            print(f"[codegraph] 👁  Watching: {self.watch_path}")
            print("[codegraph]    Press Ctrl+C to stop.")

    def stop(self):
        self._observer.stop()
        self._observer.join()
        if self.verbose:
            print("[codegraph] 👁  Watcher stopped.")

    def run_forever(self):
        """Block until Ctrl+C."""
        self.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
