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
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, TypeAlias
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

API_ROOT: Final = "https://{site_code}.goatcounter.com/api/v0"
PAGE_SIZE: Final = 100
ALL_TIME_START: Final = "1970-01-01T00:00:00Z"
SNAPSHOT_PATH: Final = Path("docs/assets/data/pageviews.json")
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
QueryValue: TypeAlias = str | int | tuple[int, ...]
JsonRequester: TypeAlias = Callable[[str, dict[str, QueryValue], str], JsonValue]


@dataclass(frozen=True, slots=True)
class GoatCounterDataError(Exception):
    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class GoatCounterHttpError(Exception):
    status_code: int
    detail: str

    def __str__(self) -> str:
        return f"GoatCounter API returned HTTP {self.status_code}: {self.detail}"


def normalize_path(value: str) -> str:
    path = urlsplit(value).path or "/"
    if not path.startswith("/"):
        path = f"/{path}"
    if path != "/" and not path.endswith("/"):
        path = f"{path}/"
    return path


def request_json(
    url: str,
    params: dict[str, QueryValue],
    api_key: str,
) -> JsonValue:
    query = urlencode(params, doseq=True)
    suffix = f"?{query}" if query else ""
    request = Request(
        f"{url}{suffix}",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urlopen(request, timeout=20) as response:
            return json.load(response)
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace").strip()
        detail = body or str(error.reason)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("error"), str):
            detail = payload["error"]
        raise GoatCounterHttpError(error.code, detail[:500]) from error


def _utc_hour(value: datetime) -> str:
    rounded = value.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    return rounded.isoformat().replace("+00:00", "Z")


def fetch_pageviews(
    api_key: str,
    site_code: str,
    request_json: JsonRequester = request_json,
    page_size: int = PAGE_SIZE,
    now: datetime | None = None,
) -> dict[str, int]:
    if page_size <= 0 or page_size > 100:
        raise GoatCounterDataError("GoatCounter page size must be between 1 and 100")

    pages: dict[str, int] = {}
    excluded_path_ids: list[int] = []
    endpoint = f"{API_ROOT.format(site_code=site_code)}/stats/hits"
    end_at = _utc_hour(now or datetime.now(UTC))
    while True:
        params: dict[str, QueryValue] = {
            "start": ALL_TIME_START,
            "end": end_at,
            "limit": page_size,
        }
        if excluded_path_ids:
            params["exclude_paths"] = tuple(excluded_path_ids)
        response = request_json(endpoint, params, api_key)
        if not isinstance(response, dict):
            raise GoatCounterDataError("GoatCounter hits response must be an object")
        hits = response.get("hits")
        more = response.get("more")
        if not isinstance(hits, list) or type(more) is not bool:
            raise GoatCounterDataError("GoatCounter hits response is invalid")

        new_path_ids = 0
        for hit in hits:
            if not isinstance(hit, dict):
                raise GoatCounterDataError("GoatCounter hit must be an object")
            path_id = hit.get("path_id")
            path_name = hit.get("path")
            count = hit.get("count")
            event = hit.get("event")
            if (
                type(path_id) is not int
                or path_id < 0
                or not isinstance(path_name, str)
                or type(count) is not int
                or count < 0
                or type(event) is not bool
                or path_id in excluded_path_ids
            ):
                raise GoatCounterDataError("GoatCounter hit is invalid")
            excluded_path_ids.append(path_id)
            new_path_ids += 1
            if event:
                continue
            path = normalize_path(path_name)
            pages[path] = pages.get(path, 0) + count

        if not more:
            return dict(sorted(pages.items()))
        if new_path_ids == 0:
            raise GoatCounterDataError(
                "GoatCounter pagination returned no new path IDs",
            )


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


def sync_snapshot(
    target: Path,
    api_key: str,
    site_code: str,
    request_json: JsonRequester = request_json,
) -> bool:
    pages = fetch_pageviews(
        api_key,
        site_code,
        request_json=request_json,
    )
    return write_snapshot(target, pages)


def main() -> int:
    api_key = os.environ.get("GOATCOUNTER_API_KEY")
    site_code = os.environ.get("GOATCOUNTER_SITE_CODE")
    if not api_key or not site_code:
        raise SystemExit(
            "GOATCOUNTER_API_KEY and GOATCOUNTER_SITE_CODE are required",
        )
    changed = sync_snapshot(SNAPSHOT_PATH, api_key, site_code)
    print("Pageview snapshot updated" if changed else "Pageview snapshot unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
