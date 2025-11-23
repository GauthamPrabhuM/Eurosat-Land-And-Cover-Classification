Credits
=======

This repository was updated with the assistance of generative AI (GenAI) tools to:

- Restructure the code into a package layout (`eurosat/`).
- Add clear module boundaries and lightweight wrappers for the top-level `train.py` and `evaluate.py` scripts.
- Improve documentation and add this acknowledgement.

What the AI changed
- Added `eurosat/` package (moved implementations from top-level scripts into the package).
- Updated top-level scripts to import and call into the package so existing CLI usage remains the same.
- Added this `CREDITS.md` and updated `readme.md`.

Human review
- The repository owner should review all changes, especially model training/evaluation logic and saved model paths.
- Confirm any credentials or remote repo push settings before pushing changes.

License and responsibility
- The AI suggestions are provided to accelerate development. The repository owner is responsible for final code correctness, licensing, and publication.
