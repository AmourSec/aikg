# Automatic Skill Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a platform-independent system that discovers Skills from configured sources, retrieves and validates the best candidates for a user task, discloses the chosen Skill names before activation, and lazily loads the selected workflows without hardcoded Skill names.

**Architecture:** A Python package builds a normalized, versioned Skill catalog from local and Git sources, then exposes hybrid retrieval, model-facing selection contracts, disclosure state, and lazy loading through stable data models and CLI commands. Runtime selection uses three disclosure levels: catalog recall, `SKILL.md` inspection, and post-disclosure activation; generated artifacts are published transactionally and guarded by deterministic tests and routing evaluations.

**Tech Stack:** Python 3.11+, standard library, PyYAML 6.x, jsonschema 4.x, `unittest`, Git CLI, Markdown, MkDocs Material.

**Authoritative Design:** `docs/superpowers/specs/2026-08-03-automatic-skill-routing-design.md`. If implementation evidence requires changing a contract, update the design, this plan, Schemas, tests, Runbook, and generated AI indexes in the same reviewed change; do not silently diverge.

## Global Constraints

- Never hardcode a business Skill name in routing logic; Skill-specific examples belong only in fixtures, configuration, documentation, and evaluations.
- Adding a valid `SKILL.md` under an enabled source must not require Router code changes.
- Adding a source must require only `skill-system/sources.yaml`, an updated lock, and normal validation artifacts.
- Default activation policy is `notify`; `confirm` Skills cannot activate until explicit user approval is recorded.
- Every selected Skill must be disclosed by public name and one-sentence purpose before activation.
- Candidate inspection may read only `SKILL.md`; referenced files load only after the Disclosure Gate passes.
- Generated files and caches are never edited manually.
- Source builds are transactional; a failed build leaves the active catalog and index unchanged.
- Enabled Skill parse rate, selected Skill load rate, and Disclosure Gate completion rate must each be 100%.
- Routing Gold Set Recall@10 must be at least 95% for catalog publication. A production-qualified model adapter additionally requires final Top-1 accuracy at least 90% and no-match false selection rate at most 5%; missing adapter evidence is `not_run`, never pass.
- External sources, including initialized Git submodules, resolve to immutable commits in `skill-system/sources.lock.json`; credentials never enter repository configuration.
- `skill-system/sources.lock.json` is the published active lock snapshot. `sync` writes only a candidate lock under ignored staging state; failed build or publication must not change the active lock snapshot.
- Preserve the untracked `sanitize-for-intranet.patch`; do not stage, edit, or delete it.
- Run tests before every task commit and run the complete verification suite before claiming implementation completion.

---

## File and Responsibility Map

### Python package

- `skill_router/models.py`: enums and immutable data contracts shared by every component.
- `skill_router/errors.py`: stable error codes and structured exceptions.
- `skill_router/schema_validation.py`: YAML/JSON loading and Draft 2020-12 Schema validation.
- `skill_router/source_registry.py`: source configuration, immutable revision and submodule resolution, local/Git materialization, and lock writing.
- `skill_router/discovery.py`: `SKILL.md` discovery, Front Matter parsing, and initial validation.
- `skill_router/references.py`: explicit reference parsing, boundary checks, deterministic content and bundle hashes.
- `skill_router/overrides.py`: reviewed metadata overrides and Quarantine acknowledgements.
- `skill_router/catalog.py`: normalized entries, canonicalization, exact duplicate folding, same-name preservation, and reports.
- `skill_router/publishing.py`: staged builds, active-version pointer, deterministic export, and rollback.
- `skill_router/index.py`: lexical BM25-like retrieval, semantic-provider protocol, and reciprocal-rank fusion.
- `skill_router/selector.py`: model-facing selection request, prompt, response validation, and selection rules.
- `skill_router/disclosure.py`: `notify`/`confirm` user messages and activation state machine.
- `skill_router/loader.py`: `inspect` and `activate` modes plus Handoff Bundle creation.
- `skill_router/router.py`: end-to-end routing state, rejected-Skill exclusion, and bounded subtask rerouting.
- `skill_router/evaluation.py`: routing metrics, regression comparison, and release thresholds.
- `skill_router/cli.py`, `skill_router/__main__.py`: maintenance and model-facing command-line interface.

### Configuration and generated artifacts

- `skill-system/sources.yaml`: editable source declarations.
- `skill-system/sources.lock.json`: generated immutable source resolutions.
- `skill-system/router-instructions.md`: platform-neutral instructions that make the model use the routing protocol.
- `skill-system/schemas/*.json`: machine-readable contracts.
- `skill-system/overrides/<source-id>/*.yaml`: reviewed local metadata and Quarantine acknowledgements.
- `skill-system/evals/routing-cases.yaml`: gold routing and no-match cases.
- `skill-system/generated/`: deterministic public catalog, summary, report, and quarantine export.
- `skill-system/state/`: ignored runtime version snapshots and active pointer.
- `skill-system/cache/`: ignored source mirrors and staging directories.

### Tests and documentation

- `tests/skill_router/`: one focused `unittest` module per component.
- `tests/fixtures/skill_router/`: local and Git fixtures, valid and invalid Skills, overrides, and gold cases.
- `docs/11-knowledge-index/skill-routing-system.md`: user-facing architecture and use protocol.
- `docs/11-knowledge-index/skill-maintenance-runbook.md`: precise newcomer/AI maintenance procedures.
- `README.md`, `skills/README.md`, `AI_CONTINUATION_GUIDE.md`, `docs/knowledge-map.md`, `mkdocs.yml`: entry-point integration.
- `scripts/generate_llms_files.py`: dynamic catalog and routing-protocol integration.

---

### Task 1: Core Models, Errors, and Schema Validation

**Files:**
- Modify: `requirements.txt`
- Create: `skill_router/__init__.py`
- Create: `skill_router/models.py`
- Create: `skill_router/errors.py`
- Create: `skill_router/schema_validation.py`
- Create: `skill-system/schemas/sources.schema.json`
- Create: `skill-system/schemas/source-lock.schema.json`
- Create: `skill-system/schemas/override.schema.json`
- Create: `skill-system/schemas/quarantine-ack.schema.json`
- Create: `skill-system/schemas/catalog-entry.schema.json`
- Create: `skill-system/schemas/selection-request.schema.json`
- Create: `skill-system/schemas/selection-response.schema.json`
- Create: `skill-system/schemas/disclosure-record.schema.json`
- Create: `skill-system/schemas/handoff.schema.json`
- Create: `skill-system/schemas/routing-session.schema.json`
- Create: `skill-system/schemas/routing-case.schema.json`
- Create: `skill-system/examples/sources.valid.yaml`
- Create: `skill-system/examples/sources.invalid.yaml`
- Create: `skill-system/examples/override.valid.yaml`
- Create: `skill-system/examples/quarantine-ack.valid.yaml`
- Create: `tests/__init__.py`
- Create: `tests/skill_router/__init__.py`
- Create: `tests/skill_router/test_schema_validation.py`
- Create: `tests/skill_router/test_models.py`

**Interfaces:**
- Produces enums: `SourceType`, `SubmoduleMode`, `SkillStatus`, `ActivationPolicy`, `RoutingDecision`, `DisclosureStatus`, `SelectionRole`.
- Produces dataclasses: `SourceSpec`, `SubmoduleLockEntry`, `SourceLockEntry`, `ResolvedSource`, `ReferenceItem`, `CatalogEntry`, `QuarantineItem`, `CandidateScore`, `SearchResult`, `SelectionItem`, `SelectionResult`, `DisclosureRecord`, `LoadedResource`, `LoadedSkill`, `HandoffBundle`, `RoutingSession`.
- Produces `load_yaml(path: Path) -> dict[str, object]`.
- Produces `load_json(path: Path) -> dict[str, object]`.
- Produces `validate_document(document: Mapping[str, object], schema_name: str, schema_dir: Path) -> None`.
- Produces `SkillRouterError(code: str, message: str, path: Path | None)` and subclasses `SchemaValidationError`, `SourceResolutionError`, `SkillValidationError`, `SelectionValidationError`, `DisclosureError`, `PublishError`.

- [ ] **Step 1: Write failing model and Schema tests**

```python
from pathlib import Path
from unittest import TestCase

from skill_router.models import ActivationPolicy, SourceSpec, SourceType, SubmoduleMode
from skill_router.schema_validation import load_yaml, validate_document


class SchemaValidationTests(TestCase):
    def test_valid_source_example_passes_schema(self) -> None:
        root = Path(__file__).resolve().parents[2]
        document = load_yaml(root / "skill-system/examples/sources.valid.yaml")
        validate_document(document, "sources.schema.json", root / "skill-system/schemas")

    def test_invalid_source_example_reports_field_path(self) -> None:
        root = Path(__file__).resolve().parents[2]
        document = load_yaml(root / "skill-system/examples/sources.invalid.yaml")
        with self.assertRaisesRegex(Exception, "sources.0.source_id"):
            validate_document(document, "sources.schema.json", root / "skill-system/schemas")


class ModelTests(TestCase):
    def test_source_spec_uses_immutable_collections(self) -> None:
        source = SourceSpec(
            source_id="local-aikg",
            type=SourceType.LOCAL,
            location="skills",
            revision=None,
            enabled=True,
            priority=100,
            include=("**/SKILL.md",),
            exclude=(),
            submodules=SubmoduleMode.NONE,
        )
        self.assertEqual(source.type, SourceType.LOCAL)
        self.assertEqual(ActivationPolicy.NOTIFY.value, "notify")
```

- [ ] **Step 2: Run the tests and verify the package is missing**

Run: `.venv/bin/python -m unittest tests.skill_router.test_models tests.skill_router.test_schema_validation -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'skill_router'`.

- [ ] **Step 3: Add explicit runtime dependencies**

Append to `requirements.txt`:

```text
PyYAML>=6.0,<7
jsonschema>=4.23,<5
```

Run: `.venv/bin/python -m pip install -r requirements.txt`

Expected: installation succeeds and `.venv/bin/python -c "import yaml, jsonschema"` exits 0.

- [ ] **Step 4: Implement stable errors, enums, and dataclasses**

Use frozen, slotted dataclasses. Define `CatalogEntry` with exactly these routing fields:

```python
@dataclass(frozen=True, slots=True)
class CatalogEntry:
    schema_version: int
    skill_id: str
    name: str
    description: str
    scope_summary: str
    source_id: str
    source_revision: str
    submodule_path: str | None
    submodule_revision: str | None
    relative_path: str
    content_hash: str
    bundle_hash: str
    status: SkillStatus
    activation_policy: ActivationPolicy
    repeatable: bool
    aliases: tuple[str, ...]
    tags: tuple[str, ...]
    dependencies: tuple[str, ...]
    conflicts: tuple[str, ...]
    references: tuple[ReferenceItem, ...]
    canonical_skill_id: str
```

Define `ReferenceItem` with `source_relative_path`, `bundle_relative_path`, `sha256`, `size_bytes`, and `media_type`. The source-relative path is used to load from the locked source; the bundle-relative path is used for source-independent duplicate hashing. Define `CandidateScore` with `skill_id`, lexical/semantic/fused scores, rank, and evidence terms. Define `SearchResult` with ordered candidates, active retrieval modes, degraded flag, and stable diagnostics.

`DisclosureRecord` must bind `session_id`, the ordered selected Skill IDs, `selection_hash`, `message_hash`, effective policy, status, and actor. `LoadedResource` must carry paths, hash, media type, encoding (`utf-8` or `base64`), and content. `LoadedSkill` must carry the verified full `SKILL.md` text and ordered loaded resources so a Handoff is self-contained rather than dependent on an unstated filesystem lookup.

`SourceSpec` contains source ID/type/location/requested revision, enabled flag, priority, include/exclude tuples, and submodule mode. `SourceLockEntry` contains requested and resolved parent revisions, a credential-free canonical location, source content fingerprint, and sorted `SubmoduleLockEntry` values. `ResolvedSource` contains the locked materialized root plus the same revision data. `QuarantineItem` contains source ID, parent revision, optional owning submodule path/commit, stable Skill path, error code, sanitized message, and acknowledgement status. `RoutingSession` contains a random non-secret session ID, original task, current subtask, status, ordered candidate/selected/completed/rejected IDs, disclosure record when present, visited state hashes, depth, and maximum depth. Persisted session and disclosure records must validate against their Schemas before use.

Implement `as_dict()` and `from_dict()` methods for each persisted dataclass. Reject unknown enum values with a `SchemaValidationError` carrying a stable code such as `schema.invalid_enum`.

- [ ] **Step 5: Implement Schema loading and deterministic validation errors**

```python
def validate_document(
    document: Mapping[str, object], schema_name: str, schema_dir: Path
) -> None:
    schema = load_json(schema_dir / schema_name)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(document), key=lambda item: list(item.absolute_path))
    if errors:
        first = errors[0]
        field = ".".join(str(part) for part in first.absolute_path) or "$"
        raise SchemaValidationError("schema.invalid", f"{field}: {first.message}")
```

Write every Schema with `additionalProperties: false`, explicit `schema_version: 1`, required keys from the design, and enum values copied from the Python enums.

- [ ] **Step 6: Run focused and complete tests**

Run: `.venv/bin/python -m unittest tests.skill_router.test_models tests.skill_router.test_schema_validation -v`

Expected: all Task 1 tests PASS.

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: the complete suite PASSes with zero failures.

- [ ] **Step 7: Commit Task 1**

```bash
git add requirements.txt skill_router skill-system/schemas skill-system/examples tests
git commit -m "feat: add skill routing data contracts"
```

---

### Task 2: Source Registry, Git Resolution, and Lock File

**Files:**
- Create: `skill_router/source_registry.py`
- Create: `skill-system/sources.yaml`
- Create: `tests/skill_router/test_source_registry.py`
- Create: `tests/fixtures/skill_router/local-source/skills/sample/SKILL.md`

**Interfaces:**
- Consumes: `SourceSpec`, `SubmoduleLockEntry`, `SourceLockEntry`, `ResolvedSource`, `SourceResolutionError`, Schema utilities.
- Produces `load_source_specs(config_path: Path, schema_dir: Path) -> tuple[SourceSpec, ...]`.
- Produces `resolve_sources(specs: Sequence[SourceSpec], workspace_root: Path, cache_root: Path, staging_root: Path, runner: CommandRunner = run_command) -> tuple[ResolvedSource, ...]`.
- Produces `write_source_lock(entries: Sequence[SourceLockEntry], path: Path) -> None`.
- `CommandRunner` signature: `Callable[[Sequence[str], Path | None], subprocess.CompletedProcess[str]]`.

- [ ] **Step 1: Write failing local-source and local-Git tests**

```python
class SourceRegistryTests(TestCase):
    def test_local_source_resolves_inside_workspace(self) -> None:
        resolved = resolve_sources(
            specs=(self.local_spec,),
            workspace_root=self.root,
            cache_root=self.temp / "cache",
            staging_root=self.temp / "staging",
        )
        self.assertRegex(resolved[0].resolved_revision, r"^local-sha256:[0-9a-f]{64}$")
        self.assertTrue(resolved[0].root.is_dir())

    def test_git_source_resolves_requested_ref_to_commit(self) -> None:
        resolved = resolve_sources(
            specs=(self.git_fixture_spec,),
            workspace_root=self.root,
            cache_root=self.temp / "cache",
            staging_root=self.temp / "staging",
        )
        self.assertRegex(resolved[0].resolved_revision, r"^[0-9a-f]{40,64}$")
        self.assertTrue((resolved[0].root / "skills/sample/SKILL.md").is_file())

    def test_recursive_submodules_are_materialized_and_locked(self) -> None:
        resolved = resolve_sources(
            specs=(self.git_fixture_with_submodule_spec,),
            workspace_root=self.root,
            cache_root=self.temp / "cache",
            staging_root=self.temp / "staging",
        )
        self.assertTrue((resolved[0].root / "official/child/skills/sample/SKILL.md").is_file())
        self.assertEqual(resolved[0].submodules[0].path, "official/child")
        self.assertRegex(resolved[0].submodules[0].commit, r"^[0-9a-f]{40,64}$")
```

Create the parent Git repository and a child submodule during `setUp()` using only local `git init`, `git add`, `git -c user.name=Fixture -c user.email=fixture@example.invalid commit`, and `git -c protocol.file.allow=always submodule add`; do not depend on the maintainer's global Git identity or network access.

- [ ] **Step 2: Run tests and verify missing source registry failure**

Run: `.venv/bin/python -m unittest tests.skill_router.test_source_registry -v`

Expected: FAIL because `skill_router.source_registry` does not exist.

- [ ] **Step 3: Implement source parsing and path safety**

Reject duplicate `source_id`, local paths outside `workspace_root`, disabled sources during resolution, empty locations, Git sources without a revision, and remote URLs containing userinfo or embedded credentials. Resolve local paths with `Path.resolve()` and compare with `is_relative_to(workspace_root.resolve())`. Default `include` to `("**/SKILL.md",)`, `exclude` to empty, and `submodules` to `none`. For a local source, compute `resolved_revision` from every sorted regular-file relative path and bytes under the source root after exclude rules (including Skill references, scripts, and assets), rejecting escaping symlinks, so any bundle change invalidates the candidate lock and produces a different deterministic build ID.

- [ ] **Step 4: Implement deterministic Git materialization**

Use argument lists, never `shell=True`:

```python
def run_command(args: Sequence[str], cwd: Path | None) -> CompletedProcess[str]:
    return subprocess.run(
        list(args), cwd=cwd, text=True, capture_output=True, check=True
    )
```

For each Git source, maintain a mirror at `cache/git/<source-id>.git`, resolve `<revision>^{commit}`, clone the mirror into the task staging directory with `--shared --no-checkout`, and check out the resolved commit in detached mode. When `submodules: recursive` is configured, run `git submodule sync --recursive` and `git submodule update --init --recursive` in the isolated checkout, then record every initialized submodule's POSIX path, URL, and checked-out commit in sorted order. The local-only test runner may inject `-c protocol.file.allow=always` for its fixture; production resolution must retain Git's default protocol restrictions. Store the parent resolved commit, upstream URL, and submodule locks in `SourceLockEntry`.

- [ ] **Step 5: Implement canonical lock serialization**

Write UTF-8 JSON with sorted source entries and submodule entries, sorted keys, two-space indentation, and a final newline. Validate the result against `source-lock.schema.json` before replacing the destination with `os.replace()`. Treat the destination as caller-selected: Task 11 writes candidate locks under staging; only publication copies a validated version lock to the active public snapshot.

- [ ] **Step 6: Run source tests and full suite**

Run: `.venv/bin/python -m unittest tests.skill_router.test_source_registry -v`

Expected: all source tests PASS, including deterministic local fingerprints, local Git commit resolution, recursive submodule materialization, and submodule lock capture.

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: zero failures.

- [ ] **Step 7: Commit Task 2**

```bash
git add skill_router/source_registry.py skill-system/sources.yaml tests/skill_router/test_source_registry.py tests/fixtures/skill_router/local-source
git commit -m "feat: resolve versioned skill sources"
```

---

### Task 3: Skill Discovery, Reference Safety, and Hashing

**Files:**
- Create: `skill_router/discovery.py`
- Create: `skill_router/references.py`
- Create: `tests/skill_router/test_discovery.py`
- Create: `tests/skill_router/test_references.py`
- Create: `tests/fixtures/skill_router/discovery-source/valid/SKILL.md`
- Create: `tests/fixtures/skill_router/discovery-source/valid/references/checklist.md`
- Create: `tests/fixtures/skill_router/discovery-source/malformed/SKILL.md`
- Create: `tests/fixtures/skill_router/discovery-source/broken-reference/SKILL.md`
- Create: `tests/fixtures/skill_router/discovery-source/escape-reference/SKILL.md`
- Create: `tests/fixtures/skill_router/discovery-source/cyclic/SKILL.md`
- Create: `tests/fixtures/skill_router/discovery-source/cyclic/references/a.md`
- Create: `tests/fixtures/skill_router/discovery-source/cyclic/references/b.md`
- Create: `tests/fixtures/skill_router/discovery-source/oversized/SKILL.md`
- Create: `tests/fixtures/skill_router/discovery-source/oversized/assets/payload.bin`

**Interfaces:**
- Consumes: `ResolvedSource`, `ReferenceItem`, `QuarantineItem`, `SkillValidationError`.
- Produces `ParsedSkill` with parent source revision, optional owning submodule path/revision, path, name, description, deterministic `scope_summary`, body, metadata, content hash, bundle hash, and references.
- Produces `DiscoveryBatch` with `skills: tuple[ParsedSkill, ...]` and `quarantine: tuple[QuarantineItem, ...]`.
- Produces `discover_skills(source: ResolvedSource) -> DiscoveryBatch`.
- Produces `resolve_references(skill_file: Path, source_root: Path, max_files: int = 64, max_bytes: int = 2_000_000) -> tuple[ReferenceItem, ...]`.
- Produces `content_hash(text: str) -> str` and `bundle_hash(skill_file: Path, references: Sequence[ReferenceItem]) -> str`.

- [ ] **Step 1: Write failing discovery and boundary tests**

```python
class DiscoveryTests(TestCase):
    def test_discovers_only_exact_skill_filename(self) -> None:
        batch = discover_skills(self.resolved_source)
        self.assertEqual([skill.name for skill in batch.skills], ["valid-skill"])
        self.assertIn("skill.front_matter_invalid", {item.error_code for item in batch.quarantine})

    def test_normalizes_crlf_before_content_hash(self) -> None:
        self.assertEqual(content_hash("a\r\nb\r\n"), content_hash("a\nb\n"))


class ReferenceTests(TestCase):
    def test_resolves_explicit_relative_reference(self) -> None:
        items = resolve_references(self.skill_file, self.source_root)
        self.assertEqual(items[0].source_relative_path, "valid/references/checklist.md")
        self.assertEqual(items[0].bundle_relative_path, "references/checklist.md")

    def test_rejects_reference_outside_source(self) -> None:
        with self.assertRaisesRegex(SkillValidationError, "reference.path_escape"):
            resolve_references(self.escape_skill, self.source_root)
```

- [ ] **Step 2: Run tests and verify missing modules**

Run: `.venv/bin/python -m unittest tests.skill_router.test_discovery tests.skill_router.test_references -v`

Expected: FAIL because discovery and reference modules do not exist.

- [ ] **Step 3: Implement strict Front Matter parsing**

Require a leading `---` block, UTF-8 decoding, scalar `name`, scalar `description`, exact `SKILL.md` filename, and a directory name compatible with the declared name. Preserve all extra Front Matter fields as read-only metadata. Derive `scope_summary` from a scalar Front Matter `scope` when present, otherwise from a `Scope`, `When to use`, or `适用范围` section, otherwise use an empty string; never ask a model to invent it during catalog build. Capture per-Skill validation failures as `QuarantineItem` records so one malformed upstream Skill does not erase valid siblings.

- [ ] **Step 4: Implement include/exclude discovery**

Sort candidate paths by POSIX relative path before parsing. Apply Source exclude rules before include rules and never follow symlinks outside the resolved source root. For each path, assign the longest matching initialized submodule path from `ResolvedSource`; retain both parent and effective submodule revisions in `ParsedSkill` and quarantine evidence.

- [ ] **Step 5: Implement explicit reference parsing and limits**

Recognize a Front Matter `resources` list, relative Markdown link/image targets, and backtick-delimited path-like tokens that resolve to an existing source-local file or directory. Strip URL fragments, reject absolute paths and URL schemes, do not infer paths from prose, expand explicitly referenced directories in sorted order, enforce `max_files` and `max_bytes`, and detect cycles by resolved path. Resolve symlinks before boundary checks and reject any target outside the locked source root.

- [ ] **Step 6: Implement deterministic content and bundle hashes**

Use `sha256:<lowercase-hex>` with normalized LF text for `content_hash`. Compute `bundle_hash` over the ordered sequence `bundle_relative_path + NUL + normalized_or_raw_file_bytes`; include normalized `SKILL.md` under bundle path `SKILL.md` as the first item. Do not use the source-relative prefix in the bundle hash, so an identical bundle nested under an aggregate repository and in its direct canonical repository can be recognized as an exact duplicate. Keep source-relative paths separately for locked loading.

- [ ] **Step 7: Run focused and full tests**

Run: `.venv/bin/python -m unittest tests.skill_router.test_discovery tests.skill_router.test_references -v`

Expected: valid Skill passes; malformed, broken, cyclic, oversized, and escaping references fail with stable error codes.

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: zero failures.

- [ ] **Step 8: Commit Task 3**

```bash
git add skill_router/discovery.py skill_router/references.py tests/skill_router tests/fixtures/skill_router/discovery-source
git commit -m "feat: discover and validate skill bundles"
```

---

### Task 4: Overrides, Quarantine Acknowledgements, and Catalog Deduplication

**Files:**
- Create: `skill_router/overrides.py`
- Create: `skill_router/catalog.py`
- Create: `tests/skill_router/test_overrides.py`
- Create: `tests/skill_router/test_catalog.py`
- Create: `tests/fixtures/skill_router/overrides/cannbot/metadata.yaml`
- Create: `tests/fixtures/skill_router/overrides/cannbot/quarantine-ack.yaml`

**Interfaces:**
- Consumes: `ParsedSkill`, `CatalogEntry`, `QuarantineItem`, Schema utilities.
- Produces `OverrideRecord`, `QuarantineAcknowledgement`, `CatalogBuild`; `CatalogBuild` contains sorted enabled/disabled entries, exact-duplicate provenance, acknowledged and unacknowledged quarantine items, and deterministic count/diff inputs for the build report.
- Produces `load_overrides(root: Path, schema_dir: Path) -> tuple[OverrideRecord, ...]`.
- Produces `load_quarantine_acknowledgements(root: Path, schema_dir: Path) -> tuple[QuarantineAcknowledgement, ...]`.
- Produces `build_catalog(skills: Sequence[ParsedSkill], quarantine: Sequence[QuarantineItem], overrides: Sequence[OverrideRecord], acknowledgements: Sequence[QuarantineAcknowledgement], source_priorities: Mapping[str, int]) -> CatalogBuild`.

- [ ] **Step 1: Write failing override and deduplication tests**

```python
class CatalogTests(TestCase):
    def test_exact_duplicate_uses_higher_priority_canonical_entry(self) -> None:
        result = build_catalog(
            skills=(self.aggregate_copy, self.direct_copy),
            quarantine=(),
            overrides=(),
            acknowledgements=(),
            source_priorities={"aggregate": 50, "direct": 80},
        )
        self.assertEqual(len(result.entries), 1)
        self.assertEqual(result.entries[0].source_id, "direct")
        self.assertEqual(result.duplicate_sources[result.entries[0].skill_id], ("aggregate", "direct"))

    def test_same_name_different_bundle_is_preserved(self) -> None:
        result = build_catalog(
            skills=(self.first_version, self.second_version),
            quarantine=(),
            overrides=(),
            acknowledgements=(),
            source_priorities={"first": 50, "second": 50},
        )
        self.assertEqual(len(result.entries), 2)
```

- [ ] **Step 2: Run tests and verify missing modules**

Run: `.venv/bin/python -m unittest tests.skill_router.test_overrides tests.skill_router.test_catalog -v`

Expected: FAIL because override and catalog modules do not exist.

- [ ] **Step 3: Implement reviewed Override application**

Allow only `aliases`, `tags`, `dependencies`, `conflicts`, `status`, `activation_policy`, `repeatable`, `canonical_skill_id`, and `corrections`. A correction may target only `name`, `description`, or `scope_summary` and must contain the replacement value, evidence path/URL, reason, and review date. Default `repeatable` to false; setting it true requires evidence that repeating the workflow in one route chain is safe and meaningful. Reject unknown Skill IDs, duplicate override fields, unqualified dependency IDs, corrections without evidence, and auto-generated changes without a committed Override file.

- [ ] **Step 4: Implement Quarantine acknowledgement matching**

For Schema v1, match only `source_id`, stable relative Skill path, exact error code, exact immutable parent source revision, and—when the path belongs to a submodule—exact submodule path and commit. Do not support wildcards, revision prefixes, open-ended ranges, or branch names in v1. An acknowledgement for a different path, code, parent revision, or submodule revision must not suppress a new error; broader revision semantics require a future Schema version and migration.

- [ ] **Step 5: Implement catalog construction and canonicalization**

Generate `skill_id` as `<source_id>:<POSIX-directory-containing-SKILL.md>` (`<source_id>:.` for a source-root Skill); never derive identity from the public `name`. Create one `CatalogEntry` per valid Skill, retain disabled entries for reporting but exclude them from retrieval, fold equal `bundle_hash` enabled entries, select explicit canonical entries first and then highest source priority, and preserve all same-name/different-bundle entries. Reject missing or cyclic canonical targets and any explicit canonical relation whose bundle hashes differ. Sort all outputs by stable ID/path. Unacknowledged quarantine items make the candidate build ineligible for publication; exact acknowledgements retain the quarantine evidence and change only gate status.

- [ ] **Step 6: Run focused and full tests**

Run: `.venv/bin/python -m unittest tests.skill_router.test_overrides tests.skill_router.test_catalog -v`

Expected: Override, acknowledgement, duplicate, same-name, disabled, and canonical tests PASS.

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: zero failures.

- [ ] **Step 7: Commit Task 4**

```bash
git add skill_router/overrides.py skill_router/catalog.py tests/skill_router tests/fixtures/skill_router/overrides
git commit -m "feat: normalize and deduplicate skill catalog"
```

---

### Task 5: Transactional Build, Public Export, and Rollback

**Files:**
- Create: `skill_router/publishing.py`
- Create: `skill-system/schemas/active-pointer.schema.json`
- Create: `skill-system/schemas/build-report.schema.json`
- Create: `tests/skill_router/test_publishing.py`
- Modify: `.gitignore`
- Create: `skill-system/generated/.gitkeep`
- Create: `skill-system/state/.gitkeep`
- Create: `skill-system/cache/.gitkeep`

**Interfaces:**
- Consumes: `CatalogBuild`, a complete candidate source lock, deterministic generator/Schema versions, and optional already-built extra artifacts. High-level source/discovery/index orchestration is intentionally deferred to Task 11.
- Produces `BuildReport`, `StagedVersion`, and `PublishedVersion`.
- Produces `stage_version(catalog: CatalogBuild, source_lock: Sequence[SourceLockEntry], system_root: Path, extra_artifacts: Mapping[str, bytes] | None = None, required_artifacts: Sequence[str] = ()) -> StagedVersion`.
- Produces `publish_version(version: StagedVersion, system_root: Path) -> PublishedVersion`.
- Produces `rollback(version_id: str, system_root: Path) -> None`.
- Active pointer format: `skill-system/state/active.json` with `build_id`, `version_path`, `catalog_hash`, and non-reproducible `activated_at`; readers treat the pointed version directory as the only coherent active lock/catalog/index set.

- [ ] **Step 1: Write failing transactional publishing tests**

```python
class PublishingTests(TestCase):
    def test_failed_candidate_does_not_replace_active_pointer(self) -> None:
        publish_version(self.good_version, self.system_root)
        before = (self.system_root / "state/active.json").read_bytes()
        with self.assertRaises(PublishError):
            publish_version(self.invalid_version, self.system_root)
        self.assertEqual((self.system_root / "state/active.json").read_bytes(), before)

    def test_rollback_restores_complete_previous_version(self) -> None:
        publish_version(self.first_version, self.system_root)
        publish_version(self.second_version, self.system_root)
        rollback(self.first_version.build_id, self.system_root)
        active = json.loads((self.system_root / "state/active.json").read_text())
        self.assertEqual(active["build_id"], self.first_version.build_id)
```

- [ ] **Step 2: Run test and verify missing publisher**

Run: `.venv/bin/python -m unittest tests.skill_router.test_publishing -v`

Expected: FAIL because `skill_router.publishing` does not exist.

- [ ] **Step 3: Implement deterministic version artifacts**

Build first under a unique staging directory, then move a validated candidate to `skill-system/state/versions/<build-id>/`. Write `sources.lock.json`, `catalog.jsonl`, `catalog-summary.md`, `quarantine.json`, `build-report.json`, a manifest of relative artifact paths/hashes/required flags, and supplied extra artifacts with canonical ordering and final newlines. Reject absolute or escaping extra-artifact paths. Validate the machine-readable report against `build-report.schema.json`. Derive `build-id` from source lock hash, catalog hash, quarantine/Override inputs, manifest/artifact hashes, generator version, and Schema version rather than wall-clock time. No version artifact may contain a current timestamp; identical inputs must produce byte-identical version artifacts. Task 6 supplies serialized index artifacts and Task 11 requires them when composing the complete version.

- [ ] **Step 4: Implement active pointer and rollback**

Validate every candidate artifact, required-artifact presence, and cross-file hash before writing an active pointer that passes `active-pointer.schema.json` to `active.json.tmp`, flush and `fsync` the file, use `os.replace()` for the active pointer, and `fsync` its parent directory where supported. Rollback must point only to an existing validated complete version and then regenerate every public export, including `sources.lock.json`, from that version. A failure before pointer replacement leaves the previous active version untouched; a post-pointer export interruption is reported as `public_export_stale` and repaired deterministically from the active version.

- [ ] **Step 5: Implement deterministic public export**

Export current human- and AI-readable artifacts to `skill-system/generated/` and copy the active version lock to `skill-system/sources.lock.json` using sibling temp files and `os.replace()`. Add a header to text exports containing generator version, build ID, source-lock hash, and `DO NOT EDIT: generated file`. Public exports are mirrors; runtime readers always resolve `state/active.json` and verify the pointed version.

- [ ] **Step 6: Update ignore rules**

Add:

```gitignore
skill-system/cache/**
!skill-system/cache/.gitkeep
skill-system/state/**
!skill-system/state/.gitkeep
skill-system/generated/index/**
```

Keep public JSON/Markdown exports and `sources.lock.json` versioned.

- [ ] **Step 7: Run focused and full tests**

Run: `.venv/bin/python -m unittest tests.skill_router.test_publishing -v`

Expected: failure, success, byte-identical repeat-build, active-pointer, stale-public-export repair, complete lock/catalog rollback, and rollback tests PASS.

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: zero failures.

- [ ] **Step 8: Commit Task 5**

```bash
git add .gitignore skill_router/publishing.py skill-system/schemas/active-pointer.schema.json skill-system/schemas/build-report.schema.json skill-system/generated/.gitkeep skill-system/state/.gitkeep skill-system/cache/.gitkeep tests/skill_router/test_publishing.py
git commit -m "feat: publish skill catalogs transactionally"
```

---

### Task 6: Lexical, Semantic, and Hybrid Retrieval

**Files:**
- Create: `skill_router/index.py`
- Create: `skill-system/schemas/index.schema.json`
- Create: `tests/skill_router/test_index.py`

**Interfaces:**
- Consumes: enabled canonical `CatalogEntry` records, `CandidateScore`, and `SearchResult`.
- Produces `tokenize(text: str) -> tuple[str, ...]`.
- Produces `LexicalIndex.build(entries: Sequence[CatalogEntry]) -> LexicalIndex` and `.search(query: str, limit: int) -> tuple[CandidateScore, ...]`.
- Produces `SemanticProvider` protocol with immutable `provider_id`, `model_revision`, `dimension`, and `embed(texts: Sequence[str]) -> Sequence[Sequence[float]]`.
- Produces `SemanticIndex.build(entries: Sequence[CatalogEntry], provider: SemanticProvider) -> SemanticIndex`.
- Produces `HybridIndex(lexical: LexicalIndex, semantic: SemanticIndex | None)` and `.search(query: str, limit: int = 12) -> SearchResult`.
- Produces `serialize_index(index: HybridIndex) -> Mapping[str, object]` and `load_index(document: Mapping[str, object], provider: SemanticProvider | None = None) -> HybridIndex` with deterministic JSON-compatible data validated against `index.schema.json`.

- [ ] **Step 1: Write failing retrieval tests**

```python
class IndexTests(TestCase):
    def test_lexical_search_matches_exact_technical_terms(self) -> None:
        index = LexicalIndex.build(self.entries)
        results = index.search("SocVersion DAV_3510", limit=3)
        self.assertEqual(results[0].skill_id, "local:npu-arch-capability-check")

    def test_hybrid_search_uses_lexical_when_semantic_is_unavailable(self) -> None:
        index = HybridIndex(LexicalIndex.build(self.entries), semantic=None)
        result = index.search("检查 NPU 架构能力", limit=3)
        self.assertEqual(result.candidates[0].skill_id, "local:npu-arch-capability-check")
        self.assertEqual(result.modes, ("lexical",))
```

- [ ] **Step 2: Run tests and verify missing index**

Run: `.venv/bin/python -m unittest tests.skill_router.test_index -v`

Expected: FAIL because `skill_router.index` does not exist.

- [ ] **Step 3: Implement deterministic multilingual tokenization**

Normalize Unicode with NFKC, lowercase Latin text, retain technical tokens containing `_`, `-`, `+`, and `.`, and emit overlapping CJK bigrams in addition to complete contiguous CJK strings. Sort no tokens; preserve query and document occurrence order.

- [ ] **Step 4: Implement lexical scoring**

Build a BM25-style index over weighted fields: name 4.0, aliases 3.0, description 2.0, tags 2.0, `scope_summary` 2.0, and remaining normalized metadata 1.0. Exclude disabled, quarantined, and non-canonical duplicate entries before indexing. Resolve equal scores by `skill_id` for deterministic output.

- [ ] **Step 5: Implement optional semantic index and reciprocal-rank fusion**

Persist normalized vectors only when a provider is configured. Validate provider identity, model revision, vector dimension, finite numeric values, and deterministic cache keys based on provider revision plus catalog content hashes. Fuse lexical and semantic ranks with configurable weights and deterministic reciprocal-rank fusion. If provider construction or query embedding fails and semantic mode is optional, return lexical candidates in a `SearchResult` marked `degraded` with a stable non-secret diagnostic code. Serialize field weights, tokenization version, entry hashes, provider metadata, vectors, and fusion parameters so runtime loading never silently rebuilds a different index. A provider revision change invalidates the semantic cache and changes the build ID.

- [ ] **Step 6: Run focused and full tests**

Run: `.venv/bin/python -m unittest tests.skill_router.test_index -v`

Expected: exact-term, Chinese alias, deterministic tie, semantic fusion, invalid-vector, and fallback tests PASS.

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: zero failures.

- [ ] **Step 7: Commit Task 6**

```bash
git add skill_router/index.py skill-system/schemas/index.schema.json tests/skill_router/test_index.py
git commit -m "feat: add hybrid skill retrieval"
```

---

### Task 7: Candidate Inspection and Model Selection Contract

**Files:**
- Create: `skill_router/selector.py`
- Create: `skill-system/router-instructions.md`
- Create: `tests/skill_router/test_selector.py`

**Interfaces:**
- Consumes: `CatalogEntry`, `CandidateScore`, `SelectionItem`, `SelectionResult`, `SkillLoader.inspect()` from Task 8 through a `SkillInspector` protocol.
- Produces `CandidateInspection` and `SelectionRequest`.
- Produces `prepare_selection_request(task: str, candidates: Sequence[CandidateScore], catalog: Mapping[str, CatalogEntry], inspector: SkillInspector, inspect_limit: int = 5, dependency_limit: int = 12) -> SelectionRequest`.
- Produces `render_selector_prompt(request: SelectionRequest) -> str`; serialized requests validate against `selection-request.schema.json`.
- Produces `validate_selection_response(document: Mapping[str, object], request: SelectionRequest, schema_dir: Path) -> SelectionResult`.

- [ ] **Step 1: Write failing selection-contract tests with a fake inspector**

```python
class SelectorTests(TestCase):
    def test_response_cannot_select_skill_outside_candidates(self) -> None:
        request = self.make_request(candidate_ids=("local:one",))
        response = {
            "decision": "selected",
            "skills": [{"skill_id": "local:other", "role": "primary", "order": 1, "reason": "wrong"}],
            "confidence": "high",
            "clarification": None,
        }
        with self.assertRaisesRegex(SelectionValidationError, "selection.unknown_candidate"):
            validate_selection_response(response, request, self.schema_dir)

    def test_no_match_requires_empty_skills(self) -> None:
        request = self.make_request(candidate_ids=("local:one",))
        response = {"decision": "no_match", "skills": [], "confidence": "high", "clarification": None}
        result = validate_selection_response(response, request, self.schema_dir)
        self.assertEqual(result.decision, RoutingDecision.NO_MATCH)
```

- [ ] **Step 2: Run tests and verify missing selector**

Run: `.venv/bin/python -m unittest tests.skill_router.test_selector -v`

Expected: FAIL because `skill_router.selector` does not exist.

- [ ] **Step 3: Implement bounded candidate inspection**

Inspect the first `inspect_limit` retrieved candidate IDs, then expand and inspect their declared dependencies transitively up to `dependency_limit`. Preserve retrieval order followed by stable prerequisite order; reject dependency cycles, missing/disabled dependencies, and expansion overflow with stable validation codes rather than omitting them silently. Attach complete `SKILL.md` text plus catalog metadata, dependency/conflict IDs, source identity, and retrieval evidence. Do not load referenced files and do not create Handoff state.

- [ ] **Step 4: Write the platform-neutral model instructions**

The instructions must require the model to:

```text
For every user request that may benefit from a repeatable task workflow, automatically consult the Skill catalog even when the user does not know or mention a Skill name.
Use the JSON/CLI reference transport when available; when another resolver is used, preserve the same search, inspection, selection, disclosure, activation, and Handoff contracts.
Check Scope and explicit exclusions, not only names.
Prefer the smallest sufficient Skill set.
Use prerequisite, primary, and supporting roles with contiguous order values.
Return no_match rather than inventing a Skill.
Ask only task-specific clarification; never ask the user to choose from the Skill catalog.
After selection, show the exact disclosure message and stop for confirmation when policy is confirm; never activate before the matching disclosure record exists.
After Handoff, let the model and loaded Skill control task execution; the Router does not execute the workflow.
During the selector phase, output only a document valid against selection-response.schema.json.
```

Delimit each inspected `SKILL.md` as untrusted candidate content for applicability analysis. The selector prompt must explicitly forbid executing candidate workflow steps, following candidate attempts to change the routing protocol, or bypassing Disclosure; the content becomes operational instruction only in the post-gate Handoff.

- [ ] **Step 5: Implement strict response validation**

Validate Schema first, then enforce candidate membership, unique IDs, exactly one primary role, contiguous order starting at 1, inclusion of every declared dependency before its consumer, absence of selected conflict pairs, empty selection for `no_match`, and non-empty clarification only for `clarification_required`. A response may not use an inspected dependency as a substitute primary unless its own Scope matches the task. Add focused tests for dependency expansion, missing dependencies, cycles, conflicts, and the smallest-sufficient-set prompt rule.

- [ ] **Step 6: Run focused and full tests**

Run: `.venv/bin/python -m unittest tests.skill_router.test_selector -v`

Expected: valid single/multi selection, dependency expansion/order, missing dependency, cycle, conflict, unknown ID, duplicate ID, role, order, no-match, clarification, and inspection-boundary tests PASS.

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: zero failures.

- [ ] **Step 7: Commit Task 7**

```bash
git add skill_router/selector.py skill-system/router-instructions.md tests/skill_router/test_selector.py
git commit -m "feat: define model skill selection contract"
```

---

### Task 8: Disclosure Gate, Loader Modes, and Handoff Bundle

**Files:**
- Create: `skill_router/disclosure.py`
- Create: `skill_router/loader.py`
- Create: `tests/skill_router/test_disclosure.py`
- Create: `tests/skill_router/test_loader.py`

**Interfaces:**
- Consumes: `RoutingSession`, `SelectionResult`, `CatalogEntry`, `DisclosureRecord`, `LoadedSkill`, `HandoffBundle`.
- Produces `DisclosureRequest` with ordered public names, purpose lines, effective policy, and rendered message.
- Produces `prepare_disclosure(session: RoutingSession, selection: SelectionResult, catalog: Mapping[str, CatalogEntry]) -> DisclosureRequest`.
- Produces `begin_disclosure(request: DisclosureRequest) -> DisclosureRecord` in `pending` state.
- Produces `record_disclosure(record: DisclosureRecord, status: DisclosureStatus, actor: str, presented_message_hash: str) -> DisclosureRecord`.
- Produces `SkillLoader.inspect(entry: CatalogEntry) -> CandidateInspection`.
- Produces `SkillLoader.activate(entry: CatalogEntry, disclosure: DisclosureRecord) -> LoadedSkill`.
- Produces `build_handoff(task: str, selection: SelectionResult, disclosure: DisclosureRecord, loaded: Sequence[LoadedSkill]) -> HandoffBundle`.

- [ ] **Step 1: Write failing disclosure and activation tests**

```python
class DisclosureTests(TestCase):
    def test_notify_message_names_skills_in_execution_order(self) -> None:
        request = prepare_disclosure(self.session, self.selection, self.catalog)
        self.assertIn("first-skill", request.message)
        self.assertLess(request.message.index("first-skill"), request.message.index("second-skill"))

    def test_confirm_skill_cannot_activate_while_pending(self) -> None:
        record = begin_disclosure(self.confirm_request)
        with self.assertRaisesRegex(DisclosureError, "disclosure.confirmation_required"):
            self.loader.activate(self.confirm_entry, record)
```

- [ ] **Step 2: Run tests and verify missing modules**

Run: `.venv/bin/python -m unittest tests.skill_router.test_disclosure tests.skill_router.test_loader -v`

Expected: FAIL because disclosure and loader modules do not exist.

- [ ] **Step 3: Implement disclosure policy aggregation and messages**

Use `confirm` if any selected Skill requires confirmation; otherwise use `notify`. Render every public Skill name, one-sentence selection reason, execution order, and whether approval is required. Same-name Skills in one selection must append `[source_id]`. Compute a canonical `selection_hash` from session ID, task hash, ordered IDs/roles/reasons, and policy, plus a `message_hash` from the exact rendered disclosure text.

- [ ] **Step 4: Implement disclosure state transitions**

Allow:

```text
notify: pending -> notified
confirm: pending -> confirmed
confirm: pending -> rejected
```

Reject `notify -> confirmed`, `confirm -> notified`, transitions after `rejected`, records that omit or reorder a selected Skill ID, mismatched selection/message hashes, and disclosure records reused across sessions. For `notified`, require actor `assistant` after the exact message was presented; for `confirmed` or `rejected`, require actor `user` and an explicit user decision supplied by the caller. This is an auditable platform-neutral contract, not a claim that the core can inspect a UI.

- [ ] **Step 5: Implement Loader `inspect` and `activate` modes**

`inspect` reads and verifies only the `SKILL.md` `content_hash`. `activate` requires a disclosure record bound to the exact current selection, verifies source and submodule revisions, `content_hash`, and `bundle_hash`, loads ordered explicit references within limits, and returns immutable `LoadedSkill` content. UTF-8 resources remain text; other resources use deterministic base64 plus media type. Neither mode executes scripts, imports code, follows network URLs, or mutates the source checkout.

- [ ] **Step 6: Implement Handoff validation**

Build ordered Handoff JSON only when all selected Skills loaded, hashes match, and disclosure is `notified` or `confirmed` according to policy. Include the verified `SKILL.md` text and loaded resource payloads for each selected Skill, together with IDs, revisions, roles, order, and hashes. Validate the serialized bundle against `handoff.schema.json`; never return a partial bundle.

- [ ] **Step 7: Run focused and full tests**

Run: `.venv/bin/python -m unittest tests.skill_router.test_disclosure tests.skill_router.test_loader -v`

Expected: notify, confirm, reject, actor, selection/message binding, cross-session replay rejection, order, same-name, tampered revision/hash, partial load, binary encoding, no-execution, and self-contained Handoff tests PASS.

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: zero failures.

- [ ] **Step 8: Commit Task 8**

```bash
git add skill_router/disclosure.py skill_router/loader.py tests/skill_router/test_disclosure.py tests/skill_router/test_loader.py
git commit -m "feat: gate and load selected skills"
```

---

### Task 9: Routing Session, Rejection, Bounded Rerouting, and Model-Facing CLI

**Files:**
- Create: `skill_router/router.py`
- Create: `skill_router/cli.py`
- Create: `skill_router/__main__.py`
- Create: `skill-system/schemas/route-outcome.schema.json`
- Create: `tests/skill_router/test_router.py`
- Create: `tests/skill_router/test_cli.py`

**Interfaces:**
- Consumes: catalog, index, selector contract, disclosure, loader, `RoutingSession`.
- Produces `RouteStatus` and `RouteOutcome`; every outcome contains the updated `session` plus only the candidates, selection request, disclosure request, or Handoff appropriate for its status.
- Produces `SkillRouter.start(task: str) -> RouteOutcome`.
- Produces `SkillRouter.apply_selection(session: RoutingSession, response: Mapping[str, object]) -> RouteOutcome`.
- Produces `SkillRouter.record_disclosure(session: RoutingSession, status: DisclosureStatus, actor: str, presented_message_hash: str) -> RouteOutcome`.
- Produces `SkillRouter.activate(session: RoutingSession) -> RouteOutcome`.
- Produces `SkillRouter.reroute(subtask: str, session: RoutingSession, completed_skill_ids: Sequence[str] = ()) -> RouteOutcome`.
- CLI commands: `search`, `prepare`, `select`, `disclose`, `activate`, `reroute`, each returning JSON on stdout and errors on stderr.

- [ ] **Step 1: Write failing route-state and rejection tests**

```python
class RouterTests(TestCase):
    def test_rejected_skill_is_excluded_from_reroute(self) -> None:
        started = self.router.start("inspect environment")
        selected = self.router.apply_selection(started.session, self.select_first)
        rejected = self.router.record_disclosure(
            selected.session,
            DisclosureStatus.REJECTED,
            actor="user",
            presented_message_hash=selected.disclosure.message_hash,
        )
        rerouted = self.router.reroute("inspect environment", rejected.session)
        self.assertNotIn("local:first", [item.skill_id for item in rerouted.candidates])

    def test_repeated_state_stops_before_maximum_depth(self) -> None:
        session = self.make_repeated_session(depth=2)
        outcome = self.router.reroute("same subtask", session)
        self.assertEqual(outcome.status, RouteStatus.LOOP_STOPPED)
```

- [ ] **Step 2: Run tests and verify missing router**

Run: `.venv/bin/python -m unittest tests.skill_router.test_router tests.skill_router.test_cli -v`

Expected: FAIL because Router and CLI modules do not exist.

- [ ] **Step 3: Implement explicit route state transitions**

Support only:

```text
new -> candidates_ready
candidates_ready -> selection_ready | no_match | clarification_required
selection_ready -> disclosure_pending
disclosure_pending -> disclosure_complete | rejected
disclosure_complete -> handoff_ready
rejected -> candidates_ready | no_match
handoff_ready -> candidates_ready for a new bounded subtask
```

Persist original task, current subtask, selected/completed/rejected IDs, visited state hashes, and depth. Default maximum reroute depth is 4 and configurable. Because execution happens outside the Router, accept completed IDs only as explicit caller input to `reroute`; validate they were in the latest `handoff_ready` selection before adding them to session state.

`apply_selection` must stop at `disclosure_pending` and return the exact disclosure message plus `display_to_user` or `ask_user_confirmation` as the next action. `activate` must reject every session that has not recorded the matching notification or confirmation; there is no combined select-and-activate shortcut.

- [ ] **Step 4: Implement route filtering and bounded rerouting**

Exclude disabled, quarantined, canonical duplicates, rejected IDs, and already completed Skills that are not repeatable. Hash `(subtask, selected, completed, rejected)` to detect loops before incrementing depth.

- [ ] **Step 5: Implement model-facing JSON CLI commands**

Examples:

```bash
python -m skill_router search --task "检查昇腾环境" --limit 12
python -m skill_router prepare --task "检查昇腾环境" --session-out /tmp/skill-session.json
python -m skill_router select --session /tmp/skill-session.json --response /tmp/selection.json
python -m skill_router disclose --session /tmp/skill-session.json --status notified --actor assistant --message-hash <hash-from-select-output>
python -m skill_router activate --session /tmp/skill-session.json
python -m skill_router reroute --session /tmp/skill-session.json --subtask "验证新的独立问题" --completed <skill-id>
```

For a confirmation decision, use `--status confirmed|rejected --actor user` only after the caller has captured explicit user input. Every command must emit a document valid against `route-outcome.schema.json` with `schema_version`, stable `status`, and next required action. Validate session JSON against `routing-session.schema.json` before every transition. It must never print credentials or full external repository paths in user-facing disclosure messages.

- [ ] **Step 6: Run focused and full tests**

Run: `.venv/bin/python -m unittest tests.skill_router.test_router tests.skill_router.test_cli -v`

Expected: state transition, rejection, depth, loop, CLI JSON, stderr, and exit-code tests PASS.

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: zero failures.

- [ ] **Step 7: Commit Task 9**

```bash
git add skill_router/router.py skill_router/cli.py skill_router/__main__.py skill-system/schemas/route-outcome.schema.json tests/skill_router/test_router.py tests/skill_router/test_cli.py
git commit -m "feat: orchestrate automatic skill routing"
```

---

### Task 10: Routing Evaluations, Regression Baseline, and Release Gates

**Files:**
- Create: `skill_router/evaluation.py`
- Create: `skill-system/schemas/evaluation-report.schema.json`
- Create: `skill-system/evals/routing-cases.yaml`
- Create: `skill-system/evals/regression-baseline.json`
- Create: `tests/skill_router/test_evaluation.py`
- Create: `tests/fixtures/skill_router/evals/routing-cases.yaml`

**Interfaces:**
- Consumes: `RoutingCase`, `HybridIndex`, validated `SelectionResult` records.
- Produces `RetrievalMetrics`, `SelectionMetrics`, `EvaluationReport`.
- Produces `evaluate_retrieval(index: HybridIndex, cases: Sequence[RoutingCase], k: int = 10) -> RetrievalMetrics`.
- Produces `evaluate_selections(cases: Sequence[RoutingCase], results: Mapping[str, SelectionResult]) -> SelectionMetrics`.
- Produces `enforce_quality_gates(report: EvaluationReport, baseline: Mapping[str, object], profile: Literal["catalog", "production"]) -> None`.

- [ ] **Step 1: Write failing metric and threshold tests**

```python
class EvaluationTests(TestCase):
    def test_no_match_false_positive_rate(self) -> None:
        metrics = evaluate_selections(self.cases, self.results_with_one_false_positive)
        self.assertEqual(metrics.no_match_false_selection_rate, 0.5)

    def test_release_gate_rejects_recall_below_threshold(self) -> None:
        report = self.report(recall_at_10=0.94, top1=0.95, false_selection=0.0)
        with self.assertRaisesRegex(PublishError, "quality.recall_at_10"):
            enforce_quality_gates(report, self.baseline, profile="catalog")
```

- [ ] **Step 2: Run tests and verify missing evaluator**

Run: `.venv/bin/python -m unittest tests.skill_router.test_evaluation -v`

Expected: FAIL because `skill_router.evaluation` does not exist.

- [ ] **Step 3: Implement deterministic routing-case parsing**

Require case ID, task, expected Skill IDs or allowed set, forbidden IDs, no-match flag, allowed order, and evidence terms. Validate against `routing-case.schema.json` and reject duplicate case IDs.

- [ ] **Step 4: Implement metrics**

Calculate Recall@10, Top-1, no-match false selection, forbidden selection, multi-Skill order, Disclosure completion, and selected-bundle load success. Use exact fractions and serialize rounded values only in a document validated against `evaluation-report.schema.json`; represent unavailable metrics with explicit status and `value: null`, never zero.

- [ ] **Step 5: Implement baseline comparison and hard gates**

Fail when absolute spec thresholds are missed or when any existing case changes from pass to fail without a reviewed baseline update. The `catalog` profile gates deterministic parse/load/disclosure contracts and Recall@10, allowing the catalog/index to become active while recording model-dependent metrics as `not_run`. The `production` profile additionally requires captured results from every selector adapter declared for deployment and enforces final Top-1, no-match false selection, forbidden selection, and multi-Skill order thresholds. Never interpret `not_run` as passed, and never describe a catalog-only publication as production-qualified.

- [ ] **Step 6: Add fixture gold cases**

Include positive local NPU architecture queries, ambiguous task clarification, a multi-Skill ordered case, and unrelated no-match queries. Include at least one Chinese paraphrase that uses no Skill name.

- [ ] **Step 7: Run focused and full tests**

Run: `.venv/bin/python -m unittest tests.skill_router.test_evaluation -v`

Expected: metric, threshold, baseline regression, missing adapter, and ordering tests PASS.

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: zero failures.

- [ ] **Step 8: Commit Task 10**

```bash
git add skill_router/evaluation.py skill-system/schemas/evaluation-report.schema.json skill-system/evals tests/skill_router/test_evaluation.py tests/fixtures/skill_router/evals
git commit -m "test: add skill routing quality gates"
```

---

### Task 11: Maintenance CLI, Real Source Configuration, and End-to-End Build

**Files:**
- Modify: `skill_router/cli.py`
- Modify: `skill_router/source_registry.py`
- Modify: `skill_router/publishing.py`
- Modify: `skill-system/sources.yaml`
- Create: `skill-system/schemas/candidate-pointer.schema.json`
- Create: `skill-system/overrides/ascend-agent-skills/metadata.yaml`
- Create: `skill-system/overrides/cannbot/metadata.yaml`
- Create: `tests/skill_router/test_end_to_end.py`
- Create: `tests/fixtures/skill_router/aggregate-source/official/CANNBot/ops/duplicate/SKILL.md`
- Create: `tests/fixtures/skill_router/direct-cannbot/ops/duplicate/SKILL.md`

**Interfaces:**
- Adds CLI commands: `validate`, `sync`, `build`, `status`, `eval`, `rollback`.
- `validate` performs Schema and local-path checks without network.
- `sync` resolves configured sources into `skill-system/state/candidates/<candidate-id>/`, where `candidate-id` is the candidate lock content hash, and writes its `sources.lock.json` only after all sources and configured submodules resolve. It then validates and atomically updates the small `skill-system/state/candidate.json` pointer against `candidate-pointer.schema.json`; it never edits the public active lock, and a partial candidate directory is never pointed to.
- Produces high-level `stage_build(config_path: Path, candidate_lock_path: Path, workspace_root: Path, system_root: Path, semantic_provider: SemanticProvider | None = None) -> StagedVersion` by composing Tasks 2-6 and placing serialized indexes in the version. Candidate Git checkouts live at sibling `sources/<source-id>/` paths derived from the candidate lock location; the lock itself contains no machine-specific absolute paths. The build requires at least `index/manifest.json` and `index/lexical.json`; semantic artifacts are required only when the configured semantic mode is mandatory. It materializes exactly the candidate lock commits and never re-resolves floating revisions during build; a changed local-source fingerprint invalidates the candidate lock and requires another `sync`.
- `build --check` creates and verifies a staged version without publishing.
- `build --publish` publishes only after the `catalog` quality profile passes; it records production qualification separately.
- `status` reports active, previous, stale, degraded, quarantine, public-export consistency, and `production_qualification: passed|failed|not_run` state.

- [ ] **Step 1: Write failing end-to-end fixture test**

```python
class EndToEndTests(TestCase):
    def test_additional_skill_is_discovered_without_router_change(self) -> None:
        first = self.build_fixture_catalog()
        self.add_skill("new-skill", "Use when checking a new subsystem.")
        second = self.build_fixture_catalog()
        self.assertNotIn("fixture:new-skill", {entry.skill_id for entry in first.entries})
        self.assertIn("fixture:new-skill", {entry.skill_id for entry in second.entries})

    def test_aggregate_cannbot_copy_is_excluded(self) -> None:
        result = self.build_aggregate_and_direct_sources()
        matches = [entry for entry in result.entries if entry.name == "duplicate"]
        self.assertEqual([entry.source_id for entry in matches], ["cannbot"])
```

- [ ] **Step 2: Run test and verify CLI gaps**

Run: `.venv/bin/python -m unittest tests.skill_router.test_end_to_end -v`

Expected: FAIL because maintenance commands and complete build orchestration are missing.

- [ ] **Step 3: Implement maintenance commands with stable exit codes**

Use exit code 0 for success, 2 for configuration/validation failure, 3 for source sync failure, 4 for quality gate failure, and 5 for publish/rollback failure. JSON reports go to stdout; diagnostic text goes to stderr.

- [ ] **Step 4: Configure the three initial sources**

Configure:

```yaml
sources:
  - source_id: local-aikg
    type: local
    location: skills
    enabled: true
    priority: 100
  - source_id: ascend-agent-skills
    type: git
    location: https://gitcode.com/Ascend/agent-skills.git
    revision: master
    submodules: recursive
    enabled: true
    priority: 50
    exclude:
      - official/CANNBot/**
  - source_id: cannbot
    type: git
    location: https://gitcode.com/cann/cannbot-skills.git
    revision: master
    enabled: true
    priority: 80
```

The generated lock must replace both `master` references with immutable commits and record the aggregate repository's initialized submodule commits. The `official/CANNBot/**` exclude prevents its content from entering discovery while the direct `cannbot` source remains canonical; the direct source's lock is independent of the aggregate gitlink.

- [ ] **Step 5: Add reviewed aliases and activation policies only where needed**

Keep upstream names and descriptions unchanged. Add Chinese aliases only for high-value initial routing cases that otherwise miss lexical retrieval. Do not invent dependencies or confirmation policies without evidence from the Skill or a documented local policy.

- [ ] **Step 6: Run fixture end-to-end tests**

Run: `.venv/bin/python -m unittest tests.skill_router.test_end_to_end -v`

Expected: new-Skill discovery, aggregate exclusion, same-name preservation, staged failure, publish, and rollback tests PASS without network.

- [ ] **Step 7: Resolve real source locks and build the real catalog**

Run: `.venv/bin/python -m skill_router validate`

Expected: source configuration and local files validate.

Run: `.venv/bin/python -m skill_router sync`

Expected: both GitCode sources and aggregate submodules resolve to immutable commits, a complete versioned candidate directory is written, and `skill-system/state/candidate.json` changes only after all succeed; `skill-system/sources.lock.json` remains unchanged.

Run: `.venv/bin/python -m skill_router build --check`

Expected: staged catalog, deduplication report, quarantine report, and indexes validate without changing active state.

Review every new Quarantine item. Fix a local error, add an exact reviewed acknowledgement, or disable the affected entry; never use a wildcard acknowledgement.

Run: `.venv/bin/python -m skill_router build --publish`

Expected: the catalog profile gates pass, the active pointer switches to the complete lock/catalog/index version, and public mirrors update coherently. If no evaluated model selector adapter is configured, status must say `production_qualification: not_run`; this is a usable catalog publication but must not be represented as a production-qualified routing release.

If selector-adapter results are available, run `.venv/bin/python -m skill_router eval --strict --profile production --selection-results <reviewed-results.json>` and require the production profile before claiming end-to-end model selection quality. The core does not call a platform or model API to manufacture this evidence.

- [ ] **Step 8: Run the complete suite**

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: zero failures.

- [ ] **Step 9: Commit Task 11**

```bash
git add skill_router/cli.py skill_router/source_registry.py skill_router/publishing.py skill-system/sources.yaml skill-system/sources.lock.json skill-system/schemas/candidate-pointer.schema.json skill-system/overrides skill-system/generated tests/skill_router/test_end_to_end.py tests/fixtures/skill_router/aggregate-source tests/fixtures/skill_router/direct-cannbot
git commit -m "feat: onboard extensible skill sources"
```

---

### Task 12: Newcomer and AI Maintenance Documentation plus LLM Index Integration

**Files:**
- Create: `skill-system/README.md`
- Create: `docs/11-knowledge-index/skill-routing-system.md`
- Create: `docs/11-knowledge-index/skill-maintenance-runbook.md`
- Modify: `README.md`
- Modify: `skills/README.md`
- Modify: `AI_CONTINUATION_GUIDE.md`
- Modify: `docs/11-knowledge-index/index.md`
- Modify: `docs/knowledge-map.md`
- Modify: `mkdocs.yml`
- Modify: `scripts/generate_llms_files.py`
- Create: `tests/skill_router/test_llms_integration.py`

**Interfaces:**
- Consumes: generated catalog and router instructions.
- Produces dynamic Skill sections in `llms.txt` and compact catalog/protocol sections in `llms-full.txt`.
- Produces complete human/AI procedures for add, update, disable, override, evaluate, publish, troubleshoot, and rollback.
- Changes generator interfaces to `build_llms_txt(paths: list[str], catalog: list[dict[str, object]]) -> str` and `build_llms_full(paths: list[str], catalog: list[dict[str, object]], local_skill_paths: list[str]) -> str`.

- [ ] **Step 1: Write failing LLM-index integration tests**

```python
class LLMIndexIntegrationTests(TestCase):
    def test_llms_txt_contains_routing_protocol_before_skill_catalog(self) -> None:
        text = build_llms_txt(self.docs, self.catalog_entries)
        self.assertLess(text.index("## Skill Routing Protocol"), text.index("## Skill Catalog"))

    def test_external_skill_bodies_are_not_inlined_in_full_dump(self) -> None:
        text = build_llms_full(self.docs, self.catalog_entries, self.local_skill_paths)
        self.assertIn("cannbot:ops/ascendc-env-check", text)
        self.assertNotIn(self.external_skill_full_body, text)
```

- [ ] **Step 2: Run test and verify static index behavior fails**

Run: `.venv/bin/python -m unittest tests.skill_router.test_llms_integration -v`

Expected: FAIL because `generate_llms_files.py` still expects `PRIORITY_SKILLS` and inlines discovered local Skill bodies directly.

- [ ] **Step 3: Implement dynamic catalog loading in the generator**

Add:

```python
def catalog_entries(catalog_path: Path) -> list[dict[str, object]]:
    if not catalog_path.exists():
        return []
    entries = [json.loads(line) for line in catalog_path.read_text(encoding="utf-8").splitlines() if line]
    return sorted(entries, key=lambda item: str(item["skill_id"]))
```

Remove `PRIORITY_SKILLS`; catalog order is the stable generated order and routing quality must not depend on a hand-maintained list. Derive `local_skill_paths` only from enabled canonical catalog entries whose source is `local-aikg`, then verify each resolved path remains under the repository `skills/` root. Generate a routing-protocol section before the compact catalog. Keep those local Skill bodies available as current repository sources; list external Skill IDs, names, descriptions, scope summaries, locked parent/submodule revisions, and the model-facing `prepare` resolver command without inlining every external body. Fail with a clear message if an active catalog exists but is malformed; an entirely absent catalog may generate a documented empty catalog during first-time bootstrap only.

- [ ] **Step 4: Write the architecture guide and exact maintenance Runbook**

Every Runbook scenario must contain these literal headings:

```text
目的
前置条件
允许修改的文件
禁止修改的文件
精确操作步骤
成功时的预期输出
失败原因与处理
验证命令
回滚方法
提交前检查
```

Cover local Skill addition, external Source addition, source update, disable/restore/delete, Override, duplicates, Quarantine, index rebuild, routing evaluation, regression analysis, publish, and rollback. Include exact commands and exit-code meanings from Task 11.

Start the Runbook with a newcomer quick start, glossary, source-of-truth table, generated-file table, decision tree for choosing a scenario, and a copy-paste AI maintainer prompt. The AI protocol must require: identify one primary Runbook scenario and any explicitly chained scenarios; list intended files before editing; stop on unknown Schema fields, unexplained generated diffs, new quarantine, or missing model evidence; run each scenario's focused checks and the global checks; compare the final diff against the combined allowlist; and report immutable source revisions and gate status without overstating `not_run` metrics.

- [ ] **Step 5: Update AI continuation rules and all navigation entry points**

Add explicit instructions that AI maintainers must read the design and Runbook, never edit generated/cache state, never hardcode Skill names, stop on ambiguity, run all gates, disclose selected Skill names, and preserve source locks. Add both documentation pages to the Knowledge Index nav and knowledge map.

- [ ] **Step 6: Regenerate AI-readable indexes**

Run: `.venv/bin/python scripts/generate_llms_files.py`

Expected: root and `docs/` copies of `llms.txt` and `llms-full.txt` update consistently and include the routing protocol plus compact catalog.

- [ ] **Step 7: Run documentation and integration checks**

Run: `.venv/bin/python -m unittest tests.skill_router.test_llms_integration -v`

Expected: routing protocol ordering, catalog inclusion, local body inclusion, external non-inlining, and stable path tests PASS.

Run: `spec_site_dir=$(mktemp -d /tmp/aikg-skill-router-docs.XXXXXX) && .venv/bin/mkdocs build --strict --site-dir "$spec_site_dir"`

Expected: MkDocs strict build exits 0.

Run: `git diff --check`

Expected: no whitespace errors.

- [ ] **Step 8: Commit Task 12**

```bash
git add README.md skills/README.md AI_CONTINUATION_GUIDE.md docs/11-knowledge-index docs/knowledge-map.md mkdocs.yml scripts/generate_llms_files.py llms.txt llms-full.txt docs/llms.txt docs/llms-full.txt skill-system/README.md tests/skill_router/test_llms_integration.py
git commit -m "docs: document automatic skill routing operations"
```

---

### Task 13: Full Verification and Acceptance Evidence

**Files:**
- Modify only if verification exposes a concrete defect in files created by Tasks 1-12.

**Interfaces:**
- Consumes the complete implementation and acceptance scenarios from the design.
- Produces fresh acceptance evidence in the implementation handoff containing command, exit code, test counts, source revisions, catalog counts, routing metrics, disclosure checks, build ID, and active version.

- [ ] **Step 1: Run the full unit and contract suite**

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: zero failures and zero errors.

- [ ] **Step 2: Verify deterministic catalog generation**

Run twice without source changes:

```bash
.venv/bin/python -m skill_router build --check --report /tmp/skill-build-first.json
.venv/bin/python -m skill_router build --check --report /tmp/skill-build-second.json
```

Expected: both reports contain the same build ID, catalog hash, bundle counts, and source lock hash.

- [ ] **Step 3: Run catalog and production qualification gates**

Run: `.venv/bin/python -m skill_router eval --strict --profile catalog --report /tmp/skill-routing-eval.json`

Expected: Recall@10 at least 0.95, forbidden deterministic contract violations 0, Disclosure completion 1.0, and selected-bundle load success 1.0. The report must explicitly mark model-dependent metrics `not_run` when no captured adapter results are supplied.

When a selector adapter is part of the intended deployment, also run:

```bash
.venv/bin/python -m skill_router eval --strict --profile production --selection-results <reviewed-results.json> --report /tmp/skill-routing-production-eval.json
```

Expected: final Top-1 at least 0.90, no-match false selection at most 0.05, forbidden selection 0, and ordered multi-Skill cases pass for every declared deployment adapter. If these results are unavailable, core implementation can be complete and the catalog can be active, but the handoff must state `production_qualification: not_run`; it must not claim production model-selection quality.

- [ ] **Step 4: Exercise disclosure and rejection acceptance cases**

Run the CLI fixture sequence for one `notify` Skill, one `confirm` Skill, and one rejected Skill. Expected results:

```text
notify -> notified -> handoff_ready
confirm -> pending -> confirmed -> handoff_ready
confirm -> pending -> rejected -> reroute excludes rejected skill_id
```

- [ ] **Step 5: Exercise failure and rollback acceptance cases**

Introduce a malformed Skill only inside a temporary fixture source, run `build --check`, verify exit code 2 or 4 and unchanged active build ID, then remove the temporary fixture. Publish a valid second fixture version, roll back to the first, and verify the lock, catalog, index, and active build ID all match the first version.

- [ ] **Step 6: Regenerate indexes and build documentation**

Run:

```bash
.venv/bin/python scripts/generate_llms_files.py
final_site_dir=$(mktemp -d /tmp/aikg-skill-router-final.XXXXXX)
.venv/bin/mkdocs build --strict --site-dir "$final_site_dir"
git diff --check
git status --short --branch
```

Expected: index generation exits 0, MkDocs exits 0, whitespace check is clean, and status contains only intended implementation artifacts plus the pre-existing untracked `sanitize-for-intranet.patch`.

- [ ] **Step 7: Report acceptance evidence from fresh command outputs**

In the implementation handoff, report exact command strings, exit codes, test totals, parent and submodule source commits, build ID, catalog enabled/disabled/quarantined/duplicate counts, catalog metrics, production qualification status and metrics when available, disclosure state results, rollback result, and documentation build result. Do not claim a gate passed without fresh output from Steps 1-6. If verification required a code or documentation fix, rerun the affected focused test and the complete suite, then commit only that focused fix before reporting.

---

## Execution Notes

- Work task-by-task; do not start Task N+1 while Task N tests or review findings remain open.
- Use a fresh failing test before each production behavior and retain that test as a regression guard.
- Stage only the files named in the current task. Never stage `sanitize-for-intranet.patch`.
- When a real GitCode sync requires network approval, request it directly for the exact Git operation; do not replace the source with an unverified copy.
- If an upstream Skill fails validation, preserve the evidence in the build report and follow the exact Quarantine procedure instead of weakening global validation.
- A model-specific selector adapter is optional in the platform-neutral core, but a production release cannot claim the model-dependent Top-1 gate until that adapter has supplied evaluated results.
