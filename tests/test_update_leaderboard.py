from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import markdown
from bs4 import BeautifulSoup

from scripts.leaderboard_data import Article, LeaderboardDataError, Snapshot
from scripts.update_leaderboard import (
    PAGE_PATH,
    rank_articles,
    rank_contributors,
    render_page,
    update_leaderboard,
)


class LeaderboardTests(unittest.TestCase):
    def test_contributor_totals_include_articles_outside_article_top_ten(self) -> None:
        articles = tuple(
            Article(f"a/{i:02}.md", str(i), ("alice",), 4) for i in range(12)
        )
        articles += (Article("b.md", "Bob", ("bob",), 30),)
        ranked = rank_contributors(articles)
        self.assertEqual(
            [(row.name, row.articles, row.views) for row in ranked],
            [("alice", 12, 48), ("bob", 1, 30)],
        )
        self.assertEqual(len(rank_articles(articles)), 10)
        self.assertEqual(rank_articles(articles)[0].title, "Bob")

    def test_both_rankings_limit_ten_and_break_ties_stably(self) -> None:
        articles = tuple(
            Article(f"a/{i:02}.md", str(i), (f"author{i:02}",), 0)
            for i in reversed(range(15))
        )
        self.assertEqual(
            [a.source for a in rank_articles(articles)],
            [f"a/{i:02}.md" for i in range(10)],
        )
        self.assertEqual(
            [a.name for a in rank_contributors(articles)],
            [f"author{i:02}" for i in range(10)],
        )

    def test_shared_article_counts_once_for_each_author_and_unknown_is_excluded(
        self,
    ) -> None:
        articles = (
            Article("a.md", "Shared", ("alice", "bob", "alice"), 20),
            Article("b.md", "Unknown", (), 99),
        )
        self.assertEqual(
            [(a.name, a.articles, a.views) for a in rank_contributors(articles)],
            [("alice", 1, 20), ("bob", 1, 20)],
        )
        self.assertEqual(rank_articles(articles)[0].views, 99)

    def test_rendered_tables_keep_titles_authors_counts_and_relative_links(
        self,
    ) -> None:
        title = "长标题 | [示例] <script>alert(1)</script>"
        author = "A_*[人] | <b>"
        source = "02-ai-workloads/中文 (测试).md"
        article = Article(source, title, (author,), 12345)
        rendered = render_page(
            (article,),
            Snapshot("2026-09-07T15:17:00+00:00", {}),
            datetime.fromisoformat("2026-09-08T23:17:00+08:00"),
        )
        soup = BeautifulSoup(
            markdown.markdown(rendered, extensions=["tables", "md_in_html"]),
            "html.parser",
        )
        tables = soup.find_all("table")
        self.assertEqual(len(tables), 2)
        cells = tables[1].select("tbody td")
        self.assertEqual(
            [td.get_text() for td in cells], ["1", title, author, "12,345"]
        )
        link = cells[1].find("a")
        assert link is not None
        self.assertEqual(
            link["href"],
            "../02-ai-workloads/%E4%B8%AD%E6%96%87%20%28%E6%B5%8B%E8%AF%95%29.md",
        )
        self.assertIsNone(soup.find("script"))

    def test_unknown_authors_are_visible_in_article_ranking(self) -> None:
        rendered = render_page(
            (Article("a.md", "Unknown", (), 0),),
            Snapshot(None, {}),
            datetime.fromisoformat("2026-09-08T00:00:00+08:00"),
        )
        soup = BeautifulSoup(
            markdown.markdown(rendered, extensions=["tables", "md_in_html"]),
            "html.parser",
        )
        self.assertEqual(len(soup.find_all("table")), 1)
        self.assertEqual(soup.select("tbody td")[2].get_text(), "作者未标注")

    def test_empty_snapshot_still_ranks_known_zero_view_articles(self) -> None:
        articles = (Article("a.md", "A", ("alice",), 0),)
        result = rank_contributors(articles)
        self.assertEqual((result[0].articles, result[0].views), (1, 0))

    def test_daily_regeneration_is_stable_and_invalid_data_preserves_last_page(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _ = subprocess.run(["git", "init", "-q", str(root)], check=True)
            _ = (root / "mkdocs.yml").write_text(
                "site_url: https://example.test/\nnav: []\n", encoding="utf-8"
            )
            snapshot = root / "docs/assets/data/pageviews.json"
            snapshot.parent.mkdir(parents=True)
            _ = snapshot.write_text(
                json.dumps({"schema_version": 1, "updated_at": None, "pages": {}}),
                encoding="utf-8",
            )
            first_day = datetime.fromisoformat("2026-09-08T23:17:00+08:00")
            self.assertTrue(update_leaderboard(root, first_day))
            self.assertFalse(update_leaderboard(root, first_day))
            yesterday = (root / PAGE_PATH).read_bytes()
            self.assertTrue(
                update_leaderboard(
                    root, datetime.fromisoformat("2026-09-09T23:17:00+08:00")
                )
            )
            today = (root / PAGE_PATH).read_bytes()
            self.assertNotEqual(today, yesterday)
            _ = snapshot.write_text(
                '{"schema_version": 1, "pages": {"/a/": -1}}', encoding="utf-8"
            )
            with self.assertRaises(LeaderboardDataError):
                update_leaderboard(root, first_day)
            self.assertEqual((root / PAGE_PATH).read_bytes(), today)


if __name__ == "__main__":
    _ = unittest.main()
