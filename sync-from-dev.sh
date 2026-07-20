#!/usr/bin/env bash
# Este repo e o ARTEFATO PUBLICADO. A fonte e a arvore de dev do nunchaku.
# Editar aqui direto ja causou um sweep inteiro invalido (20/07): as edicoes
# ficaram no dev e o pod rodou a copia velha daqui, silenciosamente.
set -euo pipefail
cd "$(dirname "$0")"
DEV="${1:-$HOME/aya-workspace/projetos/f1-recon/nunchaku}"
cp "$DEV/nunchaku/models/transformers/transformer_krea2.py" nunchaku/models/transformers/
cp "$DEV/nunchaku/models/attention_processors/krea2.py"     nunchaku/models/attention_processors/
echo "sincronizado a partir de $DEV"
git --no-pager diff --stat
