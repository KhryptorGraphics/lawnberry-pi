#!/usr/bin/env bash
set -euo pipefail

# Docs drift detection: fail if code changes have no corresponding docs/spec/journal updates.
# Intended for CI, but runnable locally.
#
# Logic:
# - Determine the diff range. Pull requests: merge-base(BASE_REF, HEAD)..HEAD. A push to the base
#   branch itself: CI passes the pre-push commit as BASE_REF (github.event.before), so the range is
#   the pushed commits; origin/main already equals HEAD there and would compare nothing.
# - code_changed: files in backend/, frontend/src/, systemd/, scripts/, pyproject.toml
#                 excluding tests/** and this script itself and CI files.
# - docs_changed: files in docs/**, spec/**, README.md, .specify/memory/AGENT_JOURNAL.md
# If code_changed is true and docs_changed is false -> exit 1 with a helpful message.
#
# Exit codes: 0 = the diff WAS computed and shows no drift (clean, or genuinely empty),
#             1 = docs drift,
#             2 = the diff could not be computed: base or head ref unresolvable, history too
#                 shallow to find a merge-base, `git diff` failed, or a push event with nothing
#                 to compare. Never a pass: an unknown comparison must not look like "no drift".

BASE_REF=${BASE_REF:-"origin/${GITHUB_BASE_REF:-main}"}
HEAD_REF=${HEAD_REF:-"HEAD"}

fail() {
  echo "check_docs_drift: ERROR: $*" >&2
  exit 2
}

resolve() {
  git rev-parse --verify --quiet "$1^{commit}"
}

is_shallow() {
  [[ "$(git rev-parse --is-shallow-repository 2>/dev/null || echo false)" == "true" ]]
}

git rev-parse --git-dir >/dev/null 2>&1 || fail "not inside a git repository"

# Ensure we have sufficient git history (unshallow if needed). A failed deepen is judged below,
# where the merge-base lookup fails closed on a shallow repository.
if is_shallow; then
  echo "Shallow repository detected; unshallowing..." >&2
  git fetch --unshallow --no-tags origin || git fetch --unshallow --no-tags ||
    echo "Warning: could not deepen the shallow repository" >&2
fi

# Resolve the base. A CI checkout with fetch-depth: 0 already has origin/<branch>; a single-branch
# or shallow checkout may not, so fetch it by an explicit refspec (the ref name BASE_REF itself,
# "origin/main", is not a branch on the remote and can never be fetched).
BASE_SHA=$(resolve "$BASE_REF" || true)
if [[ -z "$BASE_SHA" && "$BASE_REF" == origin/* ]]; then
  git fetch --no-tags origin "+refs/heads/${BASE_REF#origin/}:refs/remotes/${BASE_REF}" || true
  BASE_SHA=$(resolve "$BASE_REF" || true)
fi
[[ -n "$BASE_SHA" ]] ||
  fail "base ref '$BASE_REF' does not resolve to a commit, so there is nothing to compare against. Fetch it (CI: actions/checkout with fetch-depth: 0) or set BASE_REF to a ref or SHA that exists."

HEAD_SHA=$(resolve "$HEAD_REF" || true)
[[ -n "$HEAD_SHA" ]] || fail "head ref '$HEAD_REF' does not resolve to a commit"

if [[ "$BASE_SHA" == "$HEAD_SHA" ]]; then
  if [[ "${GITHUB_EVENT_NAME:-}" == "push" ]]; then
    fail "push event, but BASE_REF ($BASE_REF) is the pushed commit itself, so nothing would be compared. Set BASE_REF to the pre-push commit (github.event.before)."
  fi
  echo "No changed files detected: $HEAD_REF is the same commit as $BASE_REF."
  exit 0
fi

# Compute merge-base range
MERGE_BASE=$(git merge-base "$BASE_SHA" "$HEAD_SHA" 2>/dev/null || true)
if [[ -z "$MERGE_BASE" ]]; then
  if is_shallow; then
    fail "no merge-base between $BASE_REF and $HEAD_REF in a shallow repository: the history that links them is missing. Fetch full history (CI: actions/checkout with fetch-depth: 0)."
  fi
  # Unrelated histories: a real tip-to-tip diff (noisy, but it compares the trees and can fail)
  echo "Warning: no merge-base found for ${BASE_REF}..${HEAD_REF}; using tip-to-tip diff" >&2
  MERGE_BASE=$BASE_SHA
fi
RANGE="${MERGE_BASE}..${HEAD_REF}"

# NUL-separated and without rename detection: -z keeps non-ASCII paths from being emitted quoted
# (and so matching no regex below); --no-renames lists both ends of a move, so moving a file out of
# backend/ is still a code change. The list goes through a file because bash drops NUL bytes
# from variables.
CHANGED_FILE_LIST=$(mktemp)
trap 'rm -f "$CHANGED_FILE_LIST"' EXIT
git diff --name-only -z --no-renames "$MERGE_BASE" "$HEAD_SHA" >"$CHANGED_FILE_LIST" ||
  fail "git diff failed for range $RANGE"

# No changes between two resolvable commits: genuinely nothing to check
if [[ ! -s "$CHANGED_FILE_LIST" ]]; then
  echo "No changed files detected in range $RANGE."
  exit 0
fi

echo "Changed files in $RANGE:" >&2
tr '\0' '\n' <"$CHANGED_FILE_LIST" >&2

# Define code vs docs patterns
CODE_REGEX='^(backend/|frontend/src/|systemd/|scripts/|pyproject\.toml)'
# Accept both singular spec/ (repo-local) and plural specs/ (external or future mirror)
DOCS_REGEX='^(docs/|spec/|specs/|README\.md|\.specify/memory/AGENT_JOURNAL\.md)'

# Exclusions (do not count as code for drift): tests/, .github/
EXCLUDE_CODE_REGEX='^(tests/|\.github/|scripts/check_docs_drift\.sh)'

code_changed=false
docs_changed=false

while IFS= read -r -d '' file; do
  [[ -z "$file" ]] && continue
  if [[ "$file" =~ $DOCS_REGEX ]]; then
    docs_changed=true
  fi
  if [[ "$file" =~ $CODE_REGEX && ! "$file" =~ $EXCLUDE_CODE_REGEX ]]; then
    code_changed=true
  fi
done <"$CHANGED_FILE_LIST"

echo "code_changed=$code_changed, docs_changed=$docs_changed" >&2

if [[ "$code_changed" == true && "$docs_changed" == false ]]; then
  echo "Docs drift detected: code changed without corresponding docs/spec/journal updates." >&2
  echo "Please update one of: docs/**, spec/**, README.md, .specify/memory/AGENT_JOURNAL.md" >&2
  exit 1
fi

exit 0
