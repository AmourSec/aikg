# /// script
# requires-python = ">=3.12"
# dependencies = ["PyYAML>=6,<7"]
# ///
"""Generate the daily contribution page from the shared page-view snapshot.

Run from the repository root: python -m scripts.update_leaderboard
"""

from __future__ import annotations

import argparse
import html
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final
from urllib.parse import quote
from zoneinfo import ZoneInfo

from scripts.leaderboard_data import (
    Article,
    LeaderboardDataError,
    Snapshot,
    collect_articles,
    load_snapshot,
)

PAGE_PATH: Final = Path("docs/13-contribution-leaderboard/index.md")
TIMEZONE: Final = ZoneInfo("Asia/Taipei")
TOP_LIMIT: Final = 10


@dataclass(frozen=True, slots=True)
class Contributor:
    name: str
    articles: int
    views: int


def rank_contributors(articles: tuple[Article, ...]) -> tuple[Contributor, ...]:
    counts: Counter[str] = Counter()
    views: Counter[str] = Counter()
    for article in articles:
        for author in dict.fromkeys(article.authors):
            counts[author] += 1
            views[author] += article.views
    contributors = (
        Contributor(name, count, views[name]) for name, count in counts.items()
    )
    return tuple(sorted(contributors, key=lambda item: (-item.views, item.name)))[
        :TOP_LIMIT
    ]


def rank_articles(articles: tuple[Article, ...]) -> tuple[Article, ...]:
    return tuple(sorted(articles, key=lambda item: (-item.views, item.source)))[
        :TOP_LIMIT
    ]


def cell(value: str) -> str:
    """Keep metadata literal inside Markdown table cells and link labels."""
    escaped = html.escape(" ".join(value.split()), quote=False).replace("|", "&#124;")
    return re.sub(r"([\\`*{}_\[\]()!])", r"\\\1", escaped)


def render_page(
    articles: tuple[Article, ...], snapshot: Snapshot, now: datetime
) -> str:
    day = now.astimezone(TIMEZONE).date().isoformat()
    snapshot_time = (
        datetime.fromisoformat(snapshot.updated_at)
        .astimezone(TIMEZONE)
        .strftime("%Y-%m-%d %H:%M:%S")
        if snapshot.updated_at
        else "待首次同步"
    )
    contributor_count = len({name for article in articles for name in article.authors})
    lines = [
        "---",
        "title: 贡献榜单",
        "description: 每日更新的贡献者和文章累计浏览量 Top 10",
        "status: reviewed",
        "owner: maintainers",
        "license: CC-BY-4.0",
        f"updated: {day}",
        "---",
        "",
        "# 贡献榜单",
        "",
        '<div class="contribution-leaderboard" markdown="1">',
        "",
        "看看哪些贡献者的文章被读得最多，以及知识库中最受关注的文章。",
        "",
        f"榜单更新：**{day}**（台北时间，UTC+8）。每天 23:17 自动更新，任务执行可能略有延迟。",
        "",
        (
            f"统计范围：**{len(articles):,} 篇文章** · **{contributor_count:,} 位贡献者** · "
            f"**{sum(article.views for article in articles):,} 次累计浏览**。"
        ),
        "",
        f"浏览量快照最近变更：{snapshot_time}（台北时间）；浏览量未变化时保留该时间。",
        "",
        "## 贡献者排行榜 Top 10",
        "",
        "按每位贡献者名下**全部文章**的累计浏览量之和排序。",
        "",
    ]
    contributors = rank_contributors(articles)
    if contributors:
        lines.extend(
            ["| 排名 | 贡献者 | 文章数 | 累计浏览量 |", "| ---: | --- | ---: | ---: |"]
        )
        lines.extend(
            f"| {rank} | {cell(item.name)} | {item.articles:,} | {item.views:,} |"
            for rank, item in enumerate(contributors, 1)
        )
        if len(contributors) < TOP_LIMIT:
            lines.extend(
                ["", f"目前共有 {contributor_count} 位贡献者，按实际人数展示。"]
            )
    else:
        lines.append(
            "暂无可归属的贡献者；文章补充作者署名或可追溯的 Git 创建记录后自动入榜。"
        )
    lines.extend(
        [
            "",
            "## 文章排行榜 Top 10",
            "",
            "按单篇文章的累计浏览量排序。点击文章标题阅读全文。",
            "",
        ]
    )
    ranked = rank_articles(articles)
    if ranked:
        lines.extend(
            ["| 排名 | 文章 | 作者 | 累计浏览量 |", "| ---: | --- | --- | ---: |"]
        )
        for rank, article in enumerate(ranked, 1):
            url = quote(f"../{article.source}", safe="/.-_")
            authors = "、".join(cell(name) for name in article.authors) or "作者未标注"
            lines.append(
                f"| {rank} | [{cell(article.title)}]({url}) | {authors} | {article.views:,} |"
            )
    else:
        lines.append("暂无符合统计范围的文章。")
    lines.extend(
        [
            "",
            "## 统计口径",
            "",
            "- 浏览量来自 GoatCounter，与文章页面使用同一份每日快照；为接入统计以来的累计浏览次数，并非独立访客数或当日增量。",
            "- 只统计站点导航中的知识文章；首页、知识地图、主题概览、模板和榜单自身不参与排名。未有访问记录的文章按 0 计入。",
            "- 作者优先使用文章元数据中的 `authors`（兼容单作者 `author`）；历史未署名文章按 Git 首次提交者归属，文件重命名会追溯原记录。该归属代表仓库贡献记录。",
            "- 多作者文章的完整浏览量分别计入各位作者，因此贡献者浏览量相加可能超过文章总浏览量。同一作者在同篇文章中只计一次。",
            "- 同分时按贡献者名称或文章路径稳定排序，各榜最多展示 10 项。无法确认作者的文章显示“作者未标注”，不计入个人榜。",
            "",
            "作者补充、姓名合并与本地更新方法见[榜单维护说明](maintenance.md)。",
            "",
            "</div>",
            "",
        ]
    )
    return "\n".join(lines)


def update_leaderboard(root: Path, now: datetime | None = None) -> bool:
    snapshot = load_snapshot(root / "docs/assets/data/pageviews.json")
    articles = collect_articles(root, snapshot)
    content = render_page(articles, snapshot, now or datetime.now(TIMEZONE))
    target = root / PAGE_PATH
    if target.exists() and target.read_text(encoding="utf-8") == content:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=target.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            _ = handle.write(content)
        _ = temporary.replace(target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.parse_args()
    try:
        changed = update_leaderboard(Path(__file__).resolve().parents[1])
    except (LeaderboardDataError, OSError) as error:
        parser.exit(1, f"Leaderboard update failed: {error}\n")
    print(
        "Contribution leaderboard updated"
        if changed
        else "Contribution leaderboard unchanged"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
