# AI Knowledge Graph

An open AI research knowledge base for graduate student onboarding, structured browsing, and AI-assisted retrieval.

## Structure

- `docs/`: Markdown source documents.
- `docs/99-templates/`: Templates for reusable knowledge records.
- `mkdocs.yml`: MkDocs site configuration.
- `.github/workflows/deploy-pages.yml`: GitHub Pages build and deployment workflow.

## Knowledge Map

The outline is organized as an academic and technical study map:

- Foundations: mathematics, statistical learning, deep learning, and task formulation.
- Core methods: model architectures, datasets, training algorithms, fine-tuning, and alignment.
- Inference and interaction: efficient inference, retrieval augmentation, tool use, and agents.
- Systems and hardware: ML frameworks, compilers, runtime systems, accelerators, clusters, and experiment platforms.
- Research practice: evaluation, reproducibility, trustworthy AI, paper reproduction, research records, and AI-readable knowledge organization.

## Local Preview

```bash
python -m pip install -r requirements.txt
mkdocs serve
```

Then open `http://127.0.0.1:8000`.

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
