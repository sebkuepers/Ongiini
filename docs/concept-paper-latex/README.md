# LaTeX source for the Ongiini-Eval-OW concept paper

This directory contains the ACL-LaTeX source for
**The Ongiini-Eval-OW Benchmark** concept paper, formatted to match
the standard ACL two-column layout used by FLORES, MAFAND-MT,
Nekoto et al. 2022, and other cs.CL papers on arXiv.

## Files

| File | Purpose |
|---|---|
| `oshiwambo-eval-concept.tex` | Main LaTeX source |
| `oshiwambo-eval.bib` | BibTeX bibliography (~24 references) |
| `acl.sty` | ACL style file (from `acl-org/acl-style-files`) |
| `acl_natbib.bst` | ACL bibliography style |

## Build options

### Option A — Overleaf (recommended for arXiv submission)

1. Open [Overleaf](https://www.overleaf.com), New Project → Upload Project.
2. Zip this directory (`docs/concept-paper-latex/`) and upload the ZIP.
3. In Overleaf, set the compiler to **pdfLaTeX** (Menu → Compiler).
4. Click **Recompile**. The PDF appears on the right pane.
5. **Submit to arXiv directly from Overleaf**: Menu → Submit → arXiv.
   This uploads the LaTeX source to arXiv, which compiles it on
   their servers and produces the public PDF. arXiv accepts source
   submissions and is the conventional submission method.

### Option B — Local build with pdfLaTeX + BibTeX

Requires `pdflatex` and `bibtex`. On macOS, install via
`brew install --cask basictex` (then run `eval "$(/usr/libexec/path_helper)"`
or open a new shell). On Linux, `apt-get install texlive-latex-base
texlive-bibtex-extra texlive-fonts-recommended`.

From this directory:

```sh
pdflatex oshiwambo-eval-concept.tex
bibtex   oshiwambo-eval-concept
pdflatex oshiwambo-eval-concept.tex
pdflatex oshiwambo-eval-concept.tex
```

(Two final `pdflatex` runs to resolve cross-references and
citations.)

The PDF will be `oshiwambo-eval-concept.pdf`.

## ACL template options

The first line of `oshiwambo-eval-concept.tex` uses:

```latex
\usepackage[preprint]{acl}
```

Options:

- `[review]` — anonymous, with line numbers, no page numbers — for
  submission to a venue under review.
- `[final]` — for camera-ready acceptance.
- `[preprint]` — **non-anonymous, with page numbers, no line
  numbers** — the right choice for arXiv submission. This is what
  is currently set.

## arXiv submission notes

- Submit to **cs.CL** (Computation and Language) as the primary
  category.
- Optional cross-listing: not needed for a benchmark paper.
- Licence: CC-BY 4.0 (matches the dataset licence).
- The submission requires an endorsement in cs.CL for first-time
  arXiv submitters. See the parent project README and the
  concept paper §6.5 for endorsement-request notes.

## Reference for the markdown version

The longer-form markdown version of this paper lives at
`docs/oshiwambo-eval-concept.md` in the parent repo. The LaTeX
version is a condensed, ACL-styled port of that markdown.
