# notes2tex

Turns handwritten notes exported from a tablet app (PDF or images) into structured LaTeX and a PDF. You review every page in a web app, and your corrections are saved as training data for a recognition model trained on your own handwriting.

## Run it

```bash
cd ~/Documents/Personal/Projects/notes2tex
.venv/bin/notes2tex serve          # then open http://localhost:8000
```

Or from the command line:

```bash
.venv/bin/notes2tex convert lecture4.pdf --subject math --name "Lecture 4: Oscillations"
.venv/bin/notes2tex list
.venv/bin/notes2tex eval --split test
```

Pick an engine with `--engine ollama|claude|offline` (CLI) or the **Engine** menu at the top right of the web app. `notes2tex engines` shows which ones are ready.

## Transcription engines

| Engine | Cost | Setup | Speed |
|---|---|---|---|
| `ollama` (**default**) | free, runs on your Mac | Ollama app running + `ollama pull qwen3-vl:8b-instruct` (6.2 GB) | a few minutes per dense page; slower when the Mac is short on memory |
| `claude` | about $0.05–0.10 per page (Opus) | `ANTHROPIC_API_KEY=...` in `.env`; `NOTES2TEX_MODEL=claude-haiku-5-5` is ~1/40 of the price | one request per page |
| `offline` | free | none: you type the LaTeX in the review screen | instant |

The local engine reads the page **five lines at a time**, stacked into one numbered strip image, with the previous lines as context. Small models are far more reliable on a few lines than on a whole page, and vision models scale every small image up to a minimum size (~1000 tokens), so one strip of five lines costs about what a single line would. `NOTES2TEX_LOCAL_BATCH` changes the lines per strip, `NOTES2TEX_LOCAL_MODE=page` sends the whole page instead, `NOTES2TEX_LOCAL_MODEL` picks another Ollama model, and `NOTES2TEX_ENGINE` changes the default engine.

Every engine's output goes through a deterministic tidy-up (`backend/notes2tex/tidy.py`) before you see it. It repairs backslashes a model forgot to escape (`\to` arriving as a tab), fixes garbled č/š/ž, strips handwritten labels like "Def:", "Primer" or "Dokaz" (keeping a name, so "Def (Homotopy):" becomes `\begin{definition}[Homotopy]`), wraps the maths in text lines in `$...$`, removes stray display delimiters, and drops TeX commands that have no place in a transcription (`\def`, `\input`, `\write`, …).

## How a page flows

1. **Ingest**: PDF pages are rendered at 200 DPI. Ink is whatever stands out from the paper around it, so white *and* dark (night-mode) paper work, every pen colour is kept, and highlighter bands count as paper. Lined/grid templates are removed. Dark pages are flipped to dark ink on white (colours kept) before anything else sees them.
2. **Segment**: each written line becomes a block. Fractions and exponents stay with their line.
3. **Transcribe**: for each line, the engine returns a role (heading, definition, theorem, lemma, proof, derivation, code…), the LaTeX, and whether the line continues the one above.
4. **Review**: in the web app you fix the LaTeX, the role, or the segmentation (split/merge), then **Approve**. Approved lines become (image, LaTeX) training pairs.
5. **Build**: blocks are grouped and rendered with the subject's templates, then compiled with tectonic. If a block breaks compilation, the selected engine tries to fix it once; if it still breaks, it is shown in the PDF as red raw text (so the rest of the document always builds) and linked to its block in the review screen.

## Subjects and your examples

Each folder in `subjects/` is a subject:

- `subject.yaml`: the notes' `language` (e.g. `Slovenian`: helps the model read words and keep č/š/ž), plain-language **style rules** (these go straight into the prompt) and optional `role_templates` that override how a role becomes LaTeX.
- `preamble.tex`: packages, macros and the **look** of the statements (definition, theorem, lemma, … boxes). The math preamble follows the style of the reference PDFs in `subjects/math/examples/`: inline sans-serif labels ("**Definition 1** (Homotopy).") in tinted boxes with a coloured bar. Uncomment `\notesframedtrue` at its top for plain framed boxes. `subjects/_base/` holds the shared preamble and layout.
- `examples/`: your reference pages. Put a page export next to the LaTeX you want for it, with the same name:
  ```
  subjects/math/examples/oscillations.pdf
  subjects/math/examples/oscillations.tex     # the ideal output, or:
  subjects/math/examples/oscillations.md      # a description of the structure you want
  ```
  Up to 6 example images per subject are sent to Claude with every page (prompt-cached); the local engine gets the `.tex`/`.md` text only. A PDF **without** a matching `.tex`/`.md` is treated as a style reference and not sent to any model. Example PDFs are gitignored.

You don't need any LaTeX files for training: every page you approve in the review screen adds (line image → corrected LaTeX) pairs to `data/notes.db`.

To add a subject, copy `subjects/math`, rename it and edit it.

## Accuracy

`notes2tex eval` and the **Accuracy** page report **token accuracy**: 1 − (edit distance between the model's draft and your approved LaTeX ÷ the length of yours). Differences that don't change the rendered output, like spacing or `\dfrac` vs `\frac`, aren't counted as errors. Each document is randomly assigned to train / val / test when it is imported. The **test** split is never used for training, so its number is the honest one.

## Development

```bash
.venv/bin/python -m pytest              # backend tests
cd frontend && npm run dev              # UI with hot reload on :5173 (run `notes2tex serve` too)
cd frontend && npm run build            # rebuild the UI that `serve` hosts
```

### iCloud note
`~/Documents` syncs to iCloud. Build folders are kept out of the sync:
- the Python environment lives in `.venv.nosync` (`.venv` is a symlink to it)
- `frontend/node_modules` has the `com.apple.fileprovider.ignore#P` attribute

Without this, iCloud hides the `.pth` file Python needs and the `notes2tex` command breaks. If you recreate the venv, create it as `.venv.nosync`. `data/` (your corrections) *does* sync, which doubles as a backup.
