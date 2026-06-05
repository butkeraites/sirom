"""Fetch, parse, and subsample VRP-REP datasets.

`vrp-rep.org <http://www.vrp-rep.org/datasets.html>`_ publishes vehicle routing
benchmark instances as **one ZIP per dataset**, each holding several
*VRP-REP unified XML* instance files. This module knows how to:

* list a small curated catalog of (small) CVRP datasets,
* download and cache a dataset ZIP,
* parse a unified-XML instance into a plain :class:`Instance`, and
* subsample a large instance down to a size SIROM can actually solve
  (depot + the ``k`` nearest customers).

It deliberately depends only on the standard library — the VRP backend image
stays light and never imports SIROM (it talks to the SIROM HTTP API instead).
"""

from __future__ import annotations

import io
import math
import os
import re
import tempfile
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

VRP_REP_BASE = os.getenv("VRP_REP_BASE", "http://www.vrp-rep.org")
FETCH_TIMEOUT = float(os.getenv("VRP_FETCH_TIMEOUT", "60"))
_CACHE_DIR = os.getenv("VRP_CACHE_DIR", os.path.join(tempfile.gettempdir(), "vrp-rep-cache"))


@dataclass(frozen=True)
class Customer:
    id: int          # the VRP-REP node id (as found in the file)
    cx: float
    cy: float
    demand: float


@dataclass(frozen=True)
class Instance:
    dataset: str
    name: str
    depot_id: int
    depot_cx: float
    depot_cy: float
    customers: List[Customer]
    capacity: float
    total_customers: int   # before any subsampling

    @property
    def n(self) -> int:
        return len(self.customers)


@dataclass(frozen=True)
class DatasetInfo:
    slug: str            # vrp-rep download slug, e.g. "augerat-1995-set-a"
    title: str
    variant: str         # always "CVRP" here — the only variant this demo solves


# The full set of **CVRP** datasets published on VRP-REP (discovered from the
# repository's dataset pages). The slug is the download slug
# (``/datasets/download/<slug>.zip``); instance file names are listed *lazily*
# from the zip on demand (see :func:`list_instances`), so adding a dataset is a
# single line here. Other VRP-REP variants (VRPTW, PDPTW, MDVRP, …) are
# intentionally excluded: this demo's model is capacitated routing only.
# Coordinate-based CVRP datasets (smallest-first). Set E (Christofides–Eilon)
# and the Belgium set are excluded: they define distances with an explicit
# matrix rather than node coordinates, which the demo's map can't display.
CATALOG: List[DatasetInfo] = [
    DatasetInfo("augerat-1995-set-p", "Augerat 1995 — Set P (CVRP, small)", "CVRP"),
    DatasetInfo("augerat-1995-set-a", "Augerat 1995 — Set A (CVRP, random)", "CVRP"),
    DatasetInfo("augerat-1995-set-b", "Augerat 1995 — Set B (CVRP, clustered)", "CVRP"),
    DatasetInfo("fisher-1994-set-f", "Fisher 1994 — Set F (CVRP)", "CVRP"),
    DatasetInfo("christofides-et-al-1979-cmt", "Christofides et al. 1979 — CMT (CVRP)", "CVRP"),
    DatasetInfo("christofides-et-al-1979-set-m", "Christofides et al. 1979 — Set M (CVRP)", "CVRP"),
    DatasetInfo("uchoa-et-al-2014", "Uchoa et al. 2014 — X instances (CVRP)", "CVRP"),
    DatasetInfo("golden-et-al-1998-set-1", "Golden et al. 1998 — Set 1 (large CVRP)", "CVRP"),
    DatasetInfo("li-et-al-2005", "Li et al. 2005 (very large CVRP)", "CVRP"),
    DatasetInfo("verolog-members-vrp", "VeRoLog Members VRP (very large CVRP)", "CVRP"),
    DatasetInfo("verolog-members-vrp-2016", "VeRoLog Members VRP 2016 (very large CVRP)", "CVRP"),
]

_CATALOG_BY_SLUG: Dict[str, DatasetInfo] = {d.slug: d for d in CATALOG}

# In-memory zip cache: slug -> raw zip bytes. Guarded by a lock so concurrent
# requests for the same dataset don't trigger duplicate downloads.
_zip_cache: Dict[str, bytes] = {}
_zip_lock = threading.Lock()


class DatasetError(Exception):
    """Raised when a dataset/instance cannot be fetched or parsed."""


def list_datasets() -> List[Dict]:
    """Return the catalog as plain dicts (no instances — those load lazily)."""
    return [{"slug": d.slug, "title": d.title, "variant": d.variant} for d in CATALOG]


def _instance_sort_key(name: str) -> Tuple[int, str]:
    """Order instances by embedded node count (e.g. A-n32-k05 -> 32), then name."""
    m = re.search(r"-n0*(\d+)", name) or re.search(r"n0*(\d+)", name)
    return (int(m.group(1)) if m else 9999, name)


def list_instances(slug: str) -> List[str]:
    """List the instance file names (without ``.xml``) inside a dataset zip.

    Smallest-first so the most demo-friendly instances surface at the top.
    Solution/report files bundled alongside instances are filtered out.
    """
    if slug not in _CATALOG_BY_SLUG:
        raise DatasetError(f"Unknown dataset slug: {slug!r}.")
    data = fetch_dataset(slug)
    names = set()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for entry in zf.namelist():
            base = entry.split("/")[-1]
            low = base.lower()
            if low.endswith(".xml") and "solution" not in low and "report" not in low:
                names.add(base[:-4])
    return sorted(names, key=_instance_sort_key)


def _zip_path(slug: str) -> str:
    return os.path.join(_CACHE_DIR, f"{slug}.zip")


def fetch_dataset(slug: str) -> bytes:
    """Download (and cache) a dataset ZIP, returning its raw bytes.

    Cache layers, fastest first: process memory, then an on-disk copy under
    ``VRP_CACHE_DIR``, then the network. vrp-rep.org is slow/flaky, so a single
    successful download is reused for the life of the container.
    """
    if slug not in _CATALOG_BY_SLUG:
        raise DatasetError(f"Unknown dataset slug: {slug!r}.")

    cached = _zip_cache.get(slug)
    if cached is not None:
        return cached

    with _zip_lock:
        cached = _zip_cache.get(slug)
        if cached is not None:
            return cached

        disk = _zip_path(slug)
        if os.path.exists(disk):
            with open(disk, "rb") as fh:
                data = fh.read()
            _zip_cache[slug] = data
            return data

        url = f"{VRP_REP_BASE}/datasets/download/{slug}.zip"
        try:
            # vrp-rep.org returns 406 unless the request looks browser-like
            # (it does content negotiation on the Accept header).
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; sirom-vrp-demo)",
                    "Accept": "*/*",
                    "Accept-Language": "en",
                },
            )
            with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
                data = resp.read()
        except Exception as exc:  # noqa: BLE001 - surface a clean message
            raise DatasetError(
                f"Could not download dataset {slug!r} from vrp-rep.org "
                f"({type(exc).__name__}). The site is occasionally slow; "
                "try again in a moment."
            ) from exc

        if not data or not data[:2] == b"PK":
            raise DatasetError(
                f"Downloaded data for {slug!r} is not a valid ZIP archive."
            )

        os.makedirs(_CACHE_DIR, exist_ok=True)
        try:
            with open(disk, "wb") as fh:
                fh.write(data)
        except OSError:
            pass  # disk cache is best-effort; memory cache still applies

        _zip_cache[slug] = data
        return data


def _read_instance_xml(slug: str, name: str) -> bytes:
    """Extract one instance XML (by name, with or without .xml) from a dataset."""
    data = fetch_dataset(slug)
    wanted = name if name.endswith(".xml") else f"{name}.xml"
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = {n.split("/")[-1]: n for n in zf.namelist()}
        if wanted not in names:
            raise DatasetError(
                f"Instance {name!r} not found in dataset {slug!r}. "
                f"Available: {sorted(n for n in names if n.endswith('.xml'))}."
            )
        return zf.read(names[wanted])


def _text(node: Optional[ET.Element], default: Optional[str] = None) -> str:
    if node is None or node.text is None:
        if default is None:
            raise DatasetError("Missing expected XML element.")
        return default
    return node.text.strip()


def parse_instance(xml_bytes: bytes) -> Instance:
    """Parse a VRP-REP unified-XML CVRP instance.

    Reads node coordinates, the single vehicle profile's ``capacity``, and
    per-request ``quantity`` demands. The depot is whichever node the fleet
    departs from (falling back to a ``type 0`` node, then the node with no
    request). Time windows and other variant fields are ignored (CVRP base).
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise DatasetError(f"Malformed instance XML: {exc}") from exc

    dataset = _text(root.find("./info/dataset"), default="")
    name = _text(root.find("./info/name"), default="")

    coords: Dict[int, Tuple[float, float]] = {}
    depot_by_type: Optional[int] = None
    for node in root.findall("./network/nodes/node"):
        cx_el, cy_el = node.find("cx"), node.find("cy")
        if cx_el is None or cy_el is None:
            # Some instances (e.g. Christofides–Eilon Set E) give an explicit
            # distance matrix via <links> instead of coordinates. This demo's
            # map and Euclidean objective need coordinates.
            raise DatasetError(
                "This instance has no node coordinates (it defines distances "
                "with an explicit matrix). Pick a coordinate-based instance."
            )
        try:
            node_id = int(node.attrib["id"])
            node_type = int(node.attrib.get("type", "1"))
            coords[node_id] = (float(_text(cx_el)), float(_text(cy_el)))
        except (KeyError, ValueError) as exc:
            raise DatasetError(f"Bad node element: {exc}") from exc
        if node_type == 0 and depot_by_type is None:
            depot_by_type = node_id

    capacity_node = root.find("./fleet/vehicle_profile/capacity")
    if capacity_node is None:
        raise DatasetError("Instance has no vehicle capacity.")
    capacity = float(_text(capacity_node))

    # Customers = the requested nodes (with their demands).
    customers: List[Customer] = []
    requested: set = set()
    for req in root.findall("./requests/request"):
        try:
            node_id = int(req.attrib["node"])
            demand = float(_text(req.find("quantity")))
        except (KeyError, ValueError) as exc:
            raise DatasetError(f"Bad request element: {exc}") from exc
        if node_id not in coords:
            raise DatasetError(f"Request references unknown node {node_id}.")
        requested.add(node_id)
        cx, cy = coords[node_id]
        customers.append(Customer(id=node_id, cx=cx, cy=cy, demand=demand))

    if not customers:
        raise DatasetError("Instance has no customer requests.")

    # Depot: prefer the fleet's departure node, then a type-0 node, then the
    # single non-requested node (Uchoa-style instances mark it only this way).
    depot_id: Optional[int] = None
    dep_el = root.find("./fleet/vehicle_profile/departure_node")
    if dep_el is not None and dep_el.text and int(dep_el.text) in coords:
        depot_id = int(dep_el.text)
    elif depot_by_type is not None:
        depot_id = depot_by_type
    else:
        leftover = [nid for nid in coords if nid not in requested]
        if len(leftover) == 1:
            depot_id = leftover[0]
    if depot_id is None:
        raise DatasetError("Could not identify the depot node.")
    # A depot that also carries a request isn't a customer here.
    customers = [c for c in customers if c.id != depot_id]
    depot = (depot_id, coords[depot_id][0], coords[depot_id][1])

    return Instance(
        dataset=dataset,
        name=name,
        depot_id=depot[0],
        depot_cx=depot[1],
        depot_cy=depot[2],
        customers=customers,
        capacity=capacity,
        total_customers=len(customers),
    )


def load_instance(slug: str, name: str) -> Instance:
    """Fetch + parse one instance by dataset slug and instance name."""
    return parse_instance(_read_instance_xml(slug, name))


def subsample(instance: Instance, k: int) -> Instance:
    """Return a copy keeping the depot + the ``k`` customers nearest the depot.

    Subsampling is the price of solving a full arc-based MILP across many
    scenarios with SIROM. The number dropped is preserved via
    ``total_customers`` so the UI can be honest about what was left out.
    """
    if k <= 0:
        raise DatasetError("k must be positive.")
    if k >= instance.n:
        return instance  # already small enough; nothing dropped

    def dist(c: Customer) -> float:
        return math.hypot(c.cx - instance.depot_cx, c.cy - instance.depot_cy)

    nearest = sorted(instance.customers, key=dist)[:k]
    # Keep them in original id order for stable variable indexing downstream.
    nearest = sorted(nearest, key=lambda c: c.id)
    return Instance(
        dataset=instance.dataset,
        name=instance.name,
        depot_id=instance.depot_id,
        depot_cx=instance.depot_cx,
        depot_cy=instance.depot_cy,
        customers=nearest,
        capacity=instance.capacity,
        total_customers=instance.total_customers,
    )
