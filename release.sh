#!/usr/bin/env bash
set -e
git checkout local-dev
git branch -D latest 2>/dev/null || true
git checkout --orphan latest
git rm -rf .
git checkout local-dev -- .
git commit -m "Public snapshot: local-dev@$(git rev-parse --short local-dev)"
git push --force opensource latest:main
git checkout local-dev
