#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
npm run build
mkdir -p ../backend/templates
cp -f dist/index.html ../backend/templates/index.html
