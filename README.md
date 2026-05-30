# AI Knowledge Graph

An open AI knowledge base for human learning, structured browsing, and AI-assisted retrieval.

## Structure

- `docs/`: Markdown source documents.
- `docs/99-templates/`: Templates for reusable knowledge records.
- `mkdocs.yml`: MkDocs site configuration.
- `.github/workflows/deploy-pages.yml`: GitHub Pages build and deployment workflow.

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
