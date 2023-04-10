# Recommendations Handbook

Welcome to my tiny project -- Recommendations Handbook. It is aimed to structure one of the most interesting ML tasks with
theory and practice in Python. So that anyone with some experience in classical ML can widen skills.

## Structure
- /docs - markdowns & html rendered pages (the book itself)
- /notebook_drafts - jupyter notebooks with practical part of the book
- /supplements - supplementary materials like images
- /supplements/recsys - basic mircoservice of recsys pipeline for inference

### Helpful Commands
- `jupyter-book build ./jupyter_books` -- to run locally html rendering for convenient check
- `poetry install` - to install dependencies from poetry config
- `poetry add {module_name}` - to add new modules into config


Note: poetry does not allow to install `lightfm` by using poetry add ..., therefore the workaround is to add library via `poetry run pip install lightfm==1.17 --no-use-pep517`
