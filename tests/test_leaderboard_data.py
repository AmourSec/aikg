from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import leaderboard_data as data


class LeaderboardDataTests(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.snapshot = data.Snapshot(None, {})

    def setUp(self) -> None:
        self.addCleanup(self.directory.cleanup)
        self.git("init", "-q")
        self.git("config", "user.name", "Fixture Committer")
        self.git("config", "user.email", "fixture@example.test")
        self.write(
            "mkdocs.yml",
            "site_url: https://example.test/custom/base/\nnav:\n  - Article: topic/article.md\n",
        )

    def write(self, source: str, text: str) -> Path:
        target = self.root / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return target

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(self.root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def commit(self, author: str) -> None:
        self.git("add", ".")
        self.git(
            "commit", "-qm", "fixture", "--author", f"{author} <author@example.test>"
        )

    def test_creation_author_wins_over_latest_editor(self) -> None:
        # Given an article created and later edited by different people
        self.write("docs/topic/article.md", "# First title\n")
        self.commit("Original Author")
        self.write("docs/topic/article.md", "# Updated title\n\nMore information.\n")
        self.commit("Latest Editor")
        # When collecting the catalog
        (article,) = data.collect_articles(self.root, self.snapshot)
        # Then the creator is credited with the current title
        self.assertEqual(
            article,
            data.Article("topic/article.md", "Updated title", ("Original Author",), 0),
        )

    def test_renamed_article_follows_creation_author(self) -> None:
        # Given an article renamed by a different contributor
        self.write("docs/topic/old.md", "# Original article\n\nStable content.\n")
        self.commit("Creator")
        self.git("mv", "docs/topic/old.md", "docs/topic/article.md")
        self.commit("Renamer")
        # When collecting after the rename
        (article,) = data.collect_articles(self.root, self.snapshot)
        # Then ownership follows the original file
        self.assertEqual(article.authors, ("Creator",))

    def test_explicit_authors_override_git_and_legacy_author(self) -> None:
        # Given real YAML front matter with repeated, padded authors
        self.write(
            "docs/topic/article.md",
            '---\ntitle: "Chosen: title"\nauthors:\n  - " Alice "\n  - Bob\n  - Alice\nauthor: Legacy\n---\n# Heading\n',
        )
        self.commit("Git Author")
        # When collecting the catalog
        (article,) = data.collect_articles(self.root, self.snapshot)
        # Then authors take precedence, preserving distinct author order
        self.assertEqual(
            (article.title, article.authors), ("Chosen: title", ("Alice", "Bob"))
        )

    def test_legacy_author_is_supported(self) -> None:
        # Given a legacy singular author field
        self.write("docs/topic/article.md", "---\nauthor: ' Legacy '\n---\n# Heading\n")
        # When collecting the catalog
        (article,) = data.collect_articles(self.root, self.snapshot)
        # Then the explicit author is used
        self.assertEqual(article.authors, ("Legacy",))

    def test_body_authors_are_not_metadata_for_untracked_article(self) -> None:
        # Given an untracked article containing an authors example in its body
        self.write("docs/topic/article.md", "# Heading\n\nauthors: [Example]\n")
        self.git("add", "mkdocs.yml")
        self.git("commit", "-qm", "initialize")
        # When collecting the catalog
        (article,) = data.collect_articles(self.root, self.snapshot)
        # Then no unsupported author is inferred
        self.assertEqual(article.authors, ())

    def test_untracked_article_in_new_repository_has_no_author(self) -> None:
        # Given a new repository without any Git commits
        self.write("docs/topic/article.md", "# Heading\n")
        # When collecting the untracked article
        (article,) = data.collect_articles(self.root, self.snapshot)
        # Then its missing provenance remains explicit
        self.assertEqual(article.authors, ())

    def test_mailmap_canonicalizes_git_author_name(self) -> None:
        # Given a creator whose public name is mapped by the repository
        self.write("docs/topic/article.md", "# Heading\n")
        self.write(
            ".mailmap",
            "Canonical Name <author@example.test> Alias <author@example.test>\n",
        )
        self.commit("Alias")
        # When collecting the catalog
        (article,) = data.collect_articles(self.root, self.snapshot)
        # Then only the canonical name is exposed
        self.assertEqual(article.authors, ("Canonical Name",))

    def test_nav_limits_catalog_and_excludes_non_articles(self) -> None:
        # Given nav entries for knowledge articles, overviews, and utility pages
        paths = (
            "index.md",
            "knowledge-map.md",
            "contribution-leaderboard.md",
            "topic/index.md",
            "99-templates/note.md",
            "superpowers/plan.md",
            "13-contribution-leaderboard/index.md",
            "13-contribution-leaderboard/detail.md",
            "topic/article.md",
        )
        for source in (*paths, "topic/unlisted.md"):
            self.write(f"docs/{source}", "---\nauthors: [Writer]\n---\n# Heading\n")
        nav = "".join(
            f"  - Page: {source}\n"
            for source in (*paths, "topic/missing.md", "topic/article.md")
        )
        self.write(
            "mkdocs.yml",
            "site_url: https://example.test/\nplugin: !!python/name:module.function\nnav:\n"
            + nav,
        )
        # When collecting the configured navigation
        articles = data.collect_articles(self.root, self.snapshot)
        # Then only the listed, existing knowledge article is counted once
        self.assertEqual(
            tuple(article.source for article in articles), ("topic/article.md",)
        )

    def test_snapshot_paths_merge_only_actual_article_addresses(self) -> None:
        # Given aliases and unrelated addresses for an article under a custom base
        self.write("docs/topic/article.md", "---\nauthors: [Writer]\n---\n# Heading\n")
        pages = {
            "/topic/article": 2,
            "/topic/article/?q=1#part": 3,
            "/custom/base/topic/article/": 5,
            "https://example.test/custom/base/topic/article?x=1": 7,
            "https://foreign.test/topic/article/": 100,
            "/unknown/": 200,
            "/custom/base/topic/article.md": 400,
            "javascript:/topic/article/": 800,
            "//foreign.test/topic/article/": 1600,
            "https:/topic/article/": 3200,
        }
        target = self.write(
            "snapshot.json",
            json.dumps({"schema_version": 1, "updated_at": None, "pages": pages}),
        )
        # When loading and collecting the snapshot
        (article,) = data.collect_articles(self.root, data.load_snapshot(target))
        # Then only genuine URLs for the same article are summed
        self.assertEqual(article.views, 17)

    def test_invalid_explicit_authors_raise_instead_of_using_git(self) -> None:
        # Given malformed explicit metadata, with a known Git fallback
        self.write("docs/topic/article.md", "# Heading\n")
        self.commit("Creator")
        for metadata in (
            "authors: Alice",
            "authors: []",
            "authors: [Alice, 42]",
            "authors: [' ']",
            "authors: null",
            "author: [Alice]",
            "author: ''",
        ):
            with self.subTest(metadata=metadata):
                self.write(
                    "docs/topic/article.md", f"---\n{metadata}\n---\n# Heading\n"
                )
                # When collecting malformed metadata
                # Then explicit errors remain visible
                with self.assertRaises(data.LeaderboardDataError):
                    data.collect_articles(self.root, self.snapshot)

    def test_invalid_snapshot_counts_are_rejected(self) -> None:
        # Given values which are not nonnegative integer pageviews
        for count in (-1, True, False, 1.5, "3", None):
            with self.subTest(count=count):
                target = self.write(
                    "snapshot.json",
                    json.dumps(
                        {
                            "schema_version": 1,
                            "updated_at": None,
                            "pages": {"/topic/article/": count},
                        }
                    ),
                )
                # When loading the snapshot
                # Then malformed statistics fail explicitly
                with self.assertRaises(data.LeaderboardDataError):
                    data.load_snapshot(target)

    def test_invalid_snapshot_schema_is_rejected(self) -> None:
        # Given invalid top-level fields in otherwise valid snapshots
        for payload in (
            {"schema_version": True, "updated_at": None, "pages": {}},
            {"schema_version": 2, "updated_at": None, "pages": {}},
            {"schema_version": 1, "updated_at": 42, "pages": {}},
            {"schema_version": 1, "updated_at": None, "pages": []},
        ):
            with self.subTest(payload=payload):
                target = self.write("snapshot.json", json.dumps(payload))
                # When loading an incompatible snapshot
                # Then it is rejected before article calculations
                with self.assertRaises(data.LeaderboardDataError):
                    data.load_snapshot(target)

    def test_shallow_history_fails_with_full_history_guidance(self) -> None:
        # Given a real shallow clone containing an unsigned article
        self.write("docs/topic/article.md", "# Heading\n")
        self.commit("Creator")
        clone = self.root / "shallow-clone"
        self.git("clone", "-q", "--depth=1", self.root.as_uri(), str(clone))
        # When collecting without complete provenance
        # Then execution requests the full-history checkout configuration
        with self.assertRaisesRegex(data.LeaderboardDataError, "fetch-depth: 0"):
            data.collect_articles(clone, self.snapshot)

    def test_invalid_or_naive_snapshot_time_is_rejected(self) -> None:
        # Given a timestamp that cannot specify an unambiguous update time
        for timestamp in ("yesterday", "", "2026-09-08", "2026-09-08T03:00:00"):
            with self.subTest(timestamp=timestamp):
                target = self.write(
                    "snapshot.json",
                    json.dumps(
                        {"schema_version": 1, "updated_at": timestamp, "pages": {}}
                    ),
                )
                # When loading the snapshot
                # Then invalid or timezone-free values fail at the boundary
                with self.assertRaises(data.LeaderboardDataError):
                    data.load_snapshot(target)

    def test_snapshot_preserves_timezone_timestamp(self) -> None:
        # Given the snapshot's existing timezone-qualified schema
        target = self.write(
            "snapshot.json",
            '{"schema_version":1,"updated_at":"2026-09-08T03:11:35+08:00","pages":{}}',
        )
        # When loading the snapshot
        snapshot = data.load_snapshot(target)
        # Then the source timestamp is preserved for display
        self.assertEqual(snapshot.updated_at, "2026-09-08T03:11:35+08:00")


if __name__ == "__main__":
    unittest.main()
