# AI Knowledge Graph

An open AI infrastructure and efficient computing knowledge base for systems-oriented graduate students, engineers, and AI-assisted retrieval.

## Structure

- `docs/`: Markdown source documents.
- `docs/99-templates/`: Templates for reusable knowledge records.
- `llms.txt`: AI-readable entry index for the repository and documentation site.
- `llms-full.txt`: Aggregated Markdown context for AI ingestion.
- `scripts/generate_llms_files.py`: Generator for AI-readable index files.
- `mkdocs.yml`: MkDocs site configuration.
- `.github/workflows/deploy-pages.yml`: GitHub Pages build and deployment workflow.

## Knowledge Map

The outline is organized around AI systems, infrastructure, and efficient computing:

- Foundations and workloads: AI basics, Transformer, training/inference flow, Attention, MoE, precision formats, sequence length, batch shape, and data movement.
- Runtime path: inference serving, training systems, kernels, operators, compilers, and runtime systems.
- Infrastructure path: accelerators, memory hierarchy, interconnects, clusters, storage, networking, and schedulers.
- Measurement path: profiling, benchmark design, capacity modeling, efficiency analysis, observability, and incident review.
- Knowledge path: paper reproduction, system case studies, technical decision records, templates, metadata, and AI-readable indexing.

## Local Preview

```bash
python -m pip install -r requirements.txt
mkdocs serve
```

Then open `http://127.0.0.1:8801/aikg/`.

## AI-readable Index

When giving this knowledge base to an AI assistant, provide both the repository and the documentation entry files:

```text
GitHub: https://github.com/AmourSec/aikg
Docs: https://amoursec.github.io/aikg/
LLM index: https://amoursec.github.io/aikg/llms.txt
Full context: https://amoursec.github.io/aikg/llms-full.txt
```

Recommended instruction:

```text
Please use llms.txt as the entry index. Read llms-full.txt when you need a single-file context dump. Prefer citing source Markdown paths from the repository.
```

After adding or reorganizing documents, regenerate the AI index files:

```bash
python scripts/generate_llms_files.py
```

## Publishing

This repository is designed for GitHub Pages. After GitHub Pages is enabled, every push to `main` builds the documentation site automatically.

Recommended GitHub repository settings:

- Visibility: `Public`
- License: skip GitHub's license wizard if you are pushing this repository; the license files are already included
- Default branch: `main`
- Pages source: `GitHub Actions`

After creating the empty GitHub repository:

```bash
git add .
git commit -m "Initialize AI knowledge base"
git branch -M main
git remote add origin git@github.com:AmourSec/aikg.git
git push -u origin main
```

## License

Documentation content is licensed under Creative Commons Attribution 4.0 International. Code, configuration, and automation files are licensed under the MIT License. See `LICENSE`.

Third-party frontend assets used by the generated documentation site are listed in `THIRD_PARTY_NOTICES.md`.
