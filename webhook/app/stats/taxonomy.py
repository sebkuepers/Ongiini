"""Schema-version tag for the qualitative-analysis pipeline.

Bumping this number invalidates all cached extractions and forces the
LLM to re-process everything. Bump when you change the extraction
prompt or the synthesis prompt in a way that would yield different
labels.

(File name kept as taxonomy.py for compatibility with earlier imports.
The system has no fixed taxonomy — categories are emergent from the
data via LLM analysis. See analyses.py for the framework.)
"""

ANALYSIS_VERSION = 2  # bumped: hierarchy split + multi-dim WHO

# Backwards-compat alias used by aggregator.py. Same number.
TAXONOMY_VERSION = ANALYSIS_VERSION
