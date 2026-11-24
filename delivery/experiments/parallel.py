"""Serial-first parallel task executor (§6.1, M2.6).

Prefers deterministic serial execution for development and testing (workers=1),
with optional multiprocessing pool for heavy experimental sweeps when requested.
"""

from __future__ import annotations

import concurrent.futures
from typing import Any, Callable, Iterable, List, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def map_tasks(
    fn: Callable[[T], R],
    items: Iterable[T],
    workers: int = 1,
    desc: Optional[str] = None,
) -> List[R]:
    """Executes fn over items with serial-first semantics.

    Parameters:
    - fn: Callable taking one argument and returning a result.
    - items: Iterable of work items.
    - workers: Worker count. Defaults to 1 (purely serial, zero IPC/spawn overhead).
    - desc: Optional description for progress logging.

    Returns:
    - List of results in input order.
    """
    item_list = list(items)
    if not item_list:
        return []

    # Pure serial path: completely deterministic and debuggable
    if workers <= 1 or len(item_list) <= 1:
        return [fn(item) for item in item_list]

    # Multiprocessing pool path when explicitly requested
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(fn, item_list))

    return results
