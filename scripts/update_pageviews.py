#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv or pip install needed):
#      uv run scripts/update_pageviews.py
# 3. Or run with Python 3.12+:
#      python3 scripts/update_pageviews.py
# ──────────────────

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, TypeAlias
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

API_ROOT: Final = "https://api.umami.is/v1"
PAGE_SIZE: Final = 500
SNAPSHOT_PATH: Final = Path("docs/assets/data/pageviews.json")
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
QueryValue: TypeAlias = str | int
JsonRequester: TypeAlias = Callable[[str, dict[str, QueryValue], str], JsonValue]


@dataclass(frozen=True, slots=True)
class UmamiDataError(Exception):
    reason: str

    def __str__(self) -> str:
        return self.reason


def normalize_path(value: str) -> str:
    path = urlsplit(value).path or "/"
    if not path.startswith("/"):
        path = f"/{path}"
    if path != "/" and not path.endswith("/"):
        path = f"{path}/"
    return path


def request_json(
    path: str,
    params: dict[str, QueryValue],
    api_key: str,
) -> JsonValue:
    query = urlencode(params)
    suffix = f"?{query}" if query else ""
    request = Request(
        f"{API_ROOT}{path}{suffix}",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urlopen(request, timeout=20) as response:
        return json.load(response)


def _millisecond_timestamp(value: JsonValue, field: str) -> int:
    if not isinstance(value, str):
        raise UmamiDataError(f"Umami date range field {field!r} must be a string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise UmamiDataError(f"Umami date range field {field!r} is invalid") from error
    return int(parsed.timestamp() * 1000)


def fetch_pageviews(
    api_key: str,
    website_id: str,
    request_json: JsonRequester = request_json,
    page_size: int = PAGE_SIZE,
) -> dict[str, int]:
    if page_size <= 0:
        raise UmamiDataError("Umami page size must be positive")

    date_range = request_json(f"/websites/{website_id}/daterange", {}, api_key)
    if not isinstance(date_range, dict):
        raise UmamiDataError("Umami date range must be an object")
    start_at = _millisecond_timestamp(date_range.get("startDate"), "startDate")
    end_at = _millisecond_timestamp(date_range.get("endDate"), "endDate")

    pages: dict[str, int] = {}
    offset = 0
    while True:
        metrics = request_json(
            f"/websites/{website_id}/metrics/expanded",
            {
                "startAt": start_at,
                "endAt": end_at,
                "type": "path",
                "limit": page_size,
                "offset": offset,
            },
            api_key,
        )
        if not isinstance(metrics, list):
            raise UmamiDataError("Umami metrics must be a list")
        for metric in metrics:
            if not isinstance(metric, dict):
                raise UmamiDataError("Umami metric must be an object")
            name = metric.get("name")
            count = metric.get("pageviews")
            if not isinstance(name, str) or type(count) is not int or count < 0:
                raise UmamiDataError("Umami path metric is invalid")
            path = normalize_path(name)
            pages[path] = pages.get(path, 0) + count
        if len(metrics) < page_size:
            return dict(sorted(pages.items()))
        offset += page_size


def write_snapshot(
    target: Path,
    pages: dict[str, int],
    now: datetime | None = None,
) -> bool:
    current = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
    if isinstance(current, dict) and current.get("pages") == pages:
        return False

    timestamp = now or datetime.now(ZoneInfo("Asia/Taipei"))
    payload = {
        "schema_version": 1,
        "updated_at": timestamp.isoformat(timespec="seconds"),
        "pages": dict(sorted(pages.items())),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=target.parent,
            delete=False,
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return True


def main() -> int:
    api_key = os.environ.get("UMAMI_API_KEY")
    website_id = os.environ.get("UMAMI_WEBSITE_ID")
    if not api_key or not website_id:
        raise SystemExit("UMAMI_API_KEY and UMAMI_WEBSITE_ID are required")
    changed = write_snapshot(SNAPSHOT_PATH, fetch_pageviews(api_key, website_id))
    print("Pageview snapshot updated" if changed else "Pageview snapshot unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
