# Evaluation image = the training image + IFEval checker dependencies.
# Kept separate so evaluation needs never change the training stack's versions.
#   docker build -t ongiini-evalsuite:latest deploy/eval
FROM ongiini-train:latest
RUN pip install --no-cache-dir absl-py langdetect immutabledict nltk \
 && python3 -c "import nltk; [nltk.download(p, download_dir='/usr/share/nltk_data', quiet=True) for p in ('punkt', 'punkt_tab')]"
