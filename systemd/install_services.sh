#!/usr/bin/env bash
# LawnBerry Pi systemd installer.
#
# The unit files in this directory are templates: they carry @LB_USER@, @LB_GROUP@ and
# @LB_ROOT@ instead of a hard-coded account and checkout path. This script renders them for
# the real service account and checkout, provisions <root>/.env (a persistent
# LAWN_BERRY_AUTH_SECRET and a validated operator credential, see scripts/provision_env.py),
# installs the units and enables them.
#
#   sudo bash systemd/install_services.sh                  # account = $SUDO_USER, root = this checkout
#   sudo bash systemd/install_services.sh --user kp --operator-credential-file /root/cred
#   bash systemd/install_services.sh --dry-run             # print the rendered units, write nothing
#
# It REFUSES (exit 3, nothing changed) to install while the operator credential is missing,
# a published default or weak. Run with --help for every option.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Units rendered and installed, in order. tests/unit/test_install_services.py fails when a
# unit file in this directory is in none of these lists, so a new unit is a conscious choice.
INSTALL_UNITS=(
  lawnberry-database.service
  lawnberry-backend.service
  lawnberry-health.service
  lawnberry-frontend.service
  lawnberry-camera.service
  lawnberry-remote-access.service
  lawnberry-cert-renewal.service
  lawnberry-cert-renewal.timer
  lawnberry-backup.service
  lawnberry-backup.timer
  halow-link-watch.service
  halow-link-watch.timer
)
# Enabled at boot. Nothing is started here (see the note printed at the end).
ENABLE_UNITS=(
  lawnberry-database.service
  lawnberry-backend.service
  lawnberry-health.service
  lawnberry-frontend.service
  lawnberry-camera.service
  lawnberry-remote-access.service
  lawnberry-cert-renewal.service
  lawnberry-cert-renewal.timer
  lawnberry-backup.service
  lawnberry-backup.timer
  halow-link-watch.timer
)
# Superseded by lawnberry-cert-renewal.*: never installed, disabled when found.
LEGACY_UNITS=(lawnberry-acme-renew.service lawnberry-acme-renew.timer)
# Removed from the repository: disabled and deleted on hosts that still have them.
# lawnberry-sensors.service was a placeholder (ExecStart = echo + sleep infinity) that
# reported "active" with nothing behind it; sensors run inside the backend process.
REMOVED_UNITS=(lawnberry-sensors.service)

EXIT_ERROR=1
EXIT_USAGE=2
EXIT_REFUSED=3

LB_USER=""
LB_GROUP=""
LB_ROOT=""
DESTDIR=""
CRED_FILE=""
ROTATE=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: install_services.sh [options]

Render the LawnBerry systemd units for the real account and checkout, provision the env
file and install them.

  --user NAME        account the services run as (default: $SUDO_USER, else the owner of
                     --root; never root)
  --group NAME       that account's primary group (default: looked up)
  --root PATH        application checkout the units point at (default: this checkout)
  --operator-credential-file FILE
                     read the operator credential from FILE (first line) and store it in
                     <root>/.env. Also taken from LAWN_BERRY_OPERATOR_CREDENTIAL in the
                     environment, or from the value already in <root>/.env. It is never
                     accepted on the command line.
  --rotate-auth-secret
                     replace LAWN_BERRY_AUTH_SECRET (ends every operator session)
  --destdir DIR      stage the system files (/etc/systemd, /usr/local/sbin) under DIR; no
                     systemctl, no root needed. For packaging and tests.
  --dry-run          print the rendered units and every action, write nothing, need no root
  -h, --help         this text

The installer refuses (exit 3) when the operator credential is missing, a published default
or weak (fewer than 12 characters, ...), or the existing LAWN_BERRY_AUTH_SECRET is short.
LAWN_BERRY_AUTH_SECRET is generated once and kept across runs.
Exit status: 0 ok, 1 error, 2 usage, 3 refused (credential or secret policy). With --dry-run
the status is what a real run would return.
EOF
}

die() {
  echo "install_services.sh: ERROR: $*" >&2
  exit "$EXIT_ERROR"
}

usage_error() {
  echo "install_services.sh: $*" >&2
  echo "Try 'install_services.sh --help'." >&2
  exit "$EXIT_USAGE"
}

while (($#)); do
  case "$1" in
    --user | --group | --root | --operator-credential-file | --destdir)
      (($# >= 2)) || usage_error "$1 needs a value"
      opt="$1"
      val="$2"
      shift 2
      ;;
    --user=* | --group=* | --root=* | --operator-credential-file=* | --destdir=*)
      opt="${1%%=*}"
      val="${1#*=}"
      shift
      ;;
    --rotate-auth-secret)
      ROTATE=1
      shift
      continue
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      continue
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *) usage_error "unknown option: $1" ;;
  esac
  case "$opt" in
    --user) LB_USER="$val" ;;
    --group) LB_GROUP="$val" ;;
    --root) LB_ROOT="$val" ;;
    --operator-credential-file) CRED_FILE="$val" ;;
    --destdir) DESTDIR="$val" ;;
  esac
done

# ---------------------------------------------------------------------------------------
# resolve and validate what gets substituted
# ---------------------------------------------------------------------------------------
valid_name() { [[ "$1" =~ ^[A-Za-z_][A-Za-z0-9_-]{0,31}$ ]]; }
valid_path() { [[ "$1" =~ ^/[A-Za-z0-9_./-]+$ && "$1" != *"/.."* ]]; }

LB_ROOT="${LB_ROOT:-$REPO_DIR}"
while [[ "$LB_ROOT" != "/" && "$LB_ROOT" == */ ]]; do LB_ROOT="${LB_ROOT%/}"; done
valid_path "$LB_ROOT" ||
  die "--root '$LB_ROOT' must be an absolute path made of letters, digits and _ . / - only"

if [[ -n "$DESTDIR" ]]; then
  while [[ "$DESTDIR" != "/" && "$DESTDIR" == */ ]]; do DESTDIR="${DESTDIR%/}"; done
  [[ "$DESTDIR" == /* && "$DESTDIR" != "/" ]] || die "--destdir '$DESTDIR' must be an absolute path"
fi

if [[ -z "$LB_USER" ]]; then
  if [[ -n "${SUDO_USER:-}" && "$SUDO_USER" != "root" ]]; then
    LB_USER="$SUDO_USER"
  elif [[ -d "$LB_ROOT" ]]; then
    LB_USER="$(stat -c %U "$LB_ROOT")"
  fi
fi
[[ -n "$LB_USER" && "$LB_USER" != "root" ]] ||
  usage_error "cannot tell which account the services should run as (never root): pass --user NAME"
valid_name "$LB_USER" || usage_error "'$LB_USER' is not a usable account name: pass --user NAME"

GROUP_ASSUMED=0
if [[ -z "$LB_GROUP" ]]; then
  if LB_GROUP="$(id -gn "$LB_USER" 2>/dev/null)"; then
    :
  elif ((DRY_RUN)) || [[ -n "$DESTDIR" ]]; then
    LB_GROUP="$LB_USER"
    GROUP_ASSUMED=1
  else
    die "account '$LB_USER' does not exist on this machine"
  fi
fi
valid_name "$LB_GROUP" || usage_error "'$LB_GROUP' is not a usable group name: pass --group NAME"

if ((!DRY_RUN)) && [[ -z "$DESTDIR" ]]; then
  ((EUID == 0)) || die "this script must be run as root (use sudo), or use --dry-run / --destdir"
  getent passwd "$LB_USER" >/dev/null || die "account '$LB_USER' does not exist"
  getent group "$LB_GROUP" >/dev/null || die "group '$LB_GROUP' does not exist"
  [[ -d "$LB_ROOT" ]] || die "application root '$LB_ROOT' does not exist"
fi

command -v python3 >/dev/null 2>&1 || die "python3 is required (credential and secret checks)"

SYSTEMD_DIR="$DESTDIR/etc/systemd/system"
SYSTEMD_CONF_DIR="$DESTDIR/etc/systemd/system.conf.d"
SBIN_DIR="$DESTDIR/usr/local/sbin"

# ---------------------------------------------------------------------------------------
# render every unit into a scratch directory (nothing under $DESTDIR is touched yet)
# ---------------------------------------------------------------------------------------
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

render() {
  sed -e "s|@LB_USER@|$LB_USER|g" -e "s|@LB_GROUP@|$LB_GROUP|g" -e "s|@LB_ROOT@|$LB_ROOT|g" "$1"
}

render_checked() { # render_checked SRC DEST
  local left
  render "$1" >"$2"
  if left="$(grep -n '@LB_[A-Z_]*@' "$2")"; then
    die "$(basename "$1") has a token this installer does not know: $left"
  fi
}

mkdir -p "$STAGE/units"
DROPINS=()
for unit in "${INSTALL_UNITS[@]}"; do
  [[ -f "$SCRIPT_DIR/$unit" ]] || die "$unit is listed in install_services.sh but is not in $SCRIPT_DIR"
  render_checked "$SCRIPT_DIR/$unit" "$STAGE/units/$unit"
  if [[ -d "$SCRIPT_DIR/$unit.d" ]]; then # drop-ins: <unit>.d/*.conf
    for conf in "$SCRIPT_DIR/$unit.d/"*.conf; do
      [[ -f "$conf" ]] || continue
      mkdir -p "$STAGE/units/$unit.d"
      render_checked "$conf" "$STAGE/units/$unit.d/$(basename "$conf")"
      DROPINS+=("$unit.d/$(basename "$conf")")
    done
  fi
done

if [[ ! -x "$LB_ROOT/.venv/bin/uvicorn" ]]; then
  echo "install_services.sh: WARNING: $LB_ROOT/.venv/bin/uvicorn not found; the backend unit" \
    "will not start until the virtualenv exists (uv sync)." >&2
fi

provision() { # provision [provision_env.py args]: the credential and signing-secret gate
  local args=(--env-file "$LB_ROOT/.env")
  if [[ -n "$CRED_FILE" ]]; then args+=(--credential-file "$CRED_FILE"); fi
  if ((ROTATE)); then args+=(--rotate-auth-secret); fi
  python3 "$REPO_DIR/scripts/provision_env.py" "${args[@]}" "$@"
}

# ---------------------------------------------------------------------------------------
# dry run: show what would happen, write nothing
# ---------------------------------------------------------------------------------------
if ((DRY_RUN)); then
  echo "=== install_services.sh --dry-run: nothing is written ==="
  echo "service account : $LB_USER"
  if ((GROUP_ASSUMED)); then
    echo "service group   : $LB_GROUP (assumed: '$LB_USER' does not exist on this machine)"
  else
    echo "service group   : $LB_GROUP"
  fi
  echo "application root: $LB_ROOT"
  echo "system files    : ${DESTDIR:-/}"
  echo
  PROVISION_RC=0
  provision --check || PROVISION_RC=$?
  echo
  for unit in "${INSTALL_UNITS[@]}"; do
    echo "################ $SYSTEMD_DIR/$unit (0644 root:root)"
    cat "$STAGE/units/$unit"
    echo
  done
  for dropin in "${DROPINS[@]}"; do
    echo "################ $SYSTEMD_DIR/$dropin (0644 root:root)"
    cat "$STAGE/units/$dropin"
    echo
  done
  echo "################ would install $SBIN_DIR/halow-link-watch.sh (0755) from scripts/halow_link_watch.sh"
  echo "################ would install $SYSTEMD_CONF_DIR/10-watchdog.conf (0644) from systemd/10-watchdog.conf"
  for unit in "${REMOVED_UNITS[@]}"; do
    if [[ -e "$SYSTEMD_DIR/$unit" ]]; then
      echo "################ would disable and delete $SYSTEMD_DIR/$unit (removed from the repository)"
    fi
  done
  if [[ -z "$DESTDIR" ]]; then
    echo "################ would run: systemctl daemon-reload"
    echo "################ would run: systemctl enable ${ENABLE_UNITS[*]}"
    echo "################ would run: systemctl disable --now ${LEGACY_UNITS[*]} (those that exist)"
  fi
  echo
  if ((PROVISION_RC == 0)); then
    echo "dry-run result: a real run would SUCCEED"
  elif ((PROVISION_RC == EXIT_REFUSED)); then
    echo "dry-run result: a real run would REFUSE (exit $PROVISION_RC); see the messages above"
  else
    echo "dry-run result: a real run would FAIL (exit $PROVISION_RC); see the messages above"
  fi
  exit "$PROVISION_RC"
fi

# ---------------------------------------------------------------------------------------
# apply. The env file comes first: provision_env.py validates everything before it writes
# anything, so a refusal leaves the machine exactly as it was.
# ---------------------------------------------------------------------------------------
echo "Installing LawnBerry Pi systemd services for $LB_USER:$LB_GROUP, root $LB_ROOT${DESTDIR:+ (staged in $DESTDIR)}..."

PROVISION_OWNER=()
if getent passwd "$LB_USER" >/dev/null 2>&1 && getent group "$LB_GROUP" >/dev/null 2>&1; then
  PROVISION_OWNER=(--owner "$LB_USER" --group "$LB_GROUP")
fi
PROVISION_RC=0
provision "${PROVISION_OWNER[@]}" || PROVISION_RC=$?
if ((PROVISION_RC != 0)); then
  if ((PROVISION_RC == EXIT_REFUSED)); then
    echo "install_services.sh: REFUSED: nothing was installed or changed." >&2
  fi
  exit "$PROVISION_RC"
fi

INSTALL_OWNER=()
if ((EUID == 0)); then INSTALL_OWNER=(-o root -g root); fi

mkdir -p "$SYSTEMD_DIR" "$SYSTEMD_CONF_DIR" "$SBIN_DIR"
for unit in "${INSTALL_UNITS[@]}"; do
  install -m 0644 "${INSTALL_OWNER[@]}" "$STAGE/units/$unit" "$SYSTEMD_DIR/$unit"
done
for dropin in "${DROPINS[@]}"; do
  mkdir -p "$SYSTEMD_DIR/$(dirname "$dropin")"
  install -m 0644 "${INSTALL_OWNER[@]}" "$STAGE/units/$dropin" "$SYSTEMD_DIR/$dropin"
done
# HaLow link recovery (the shed Pi's only link) and the hardware watchdog
install -m 0755 "${INSTALL_OWNER[@]}" "$REPO_DIR/scripts/halow_link_watch.sh" "$SBIN_DIR/halow-link-watch.sh"
install -m 0644 "${INSTALL_OWNER[@]}" "$SCRIPT_DIR/10-watchdog.conf" "$SYSTEMD_CONF_DIR/10-watchdog.conf"

if [[ -z "$DESTDIR" ]]; then
  for unit in "${REMOVED_UNITS[@]}"; do
    if [[ -e "$SYSTEMD_DIR/$unit" ]]; then
      echo "Removing $unit (removed from the repository)..."
      systemctl disable --now "$unit" || true
      rm -f "$SYSTEMD_DIR/$unit"
    fi
  done
  echo "Reloading systemd daemon..."
  systemctl daemon-reload
  echo "Enabling LawnBerry Pi services..."
  systemctl enable "${ENABLE_UNITS[@]}"
  # If legacy ACME units exist, disable them to avoid conflicts with lawnberry-cert-renewal.
  for unit in "${LEGACY_UNITS[@]}"; do
    if [[ -n "$(systemctl list-unit-files --no-legend "$unit" 2>/dev/null || true)" ]]; then
      systemctl disable --now "$unit" || true
    fi
  done
else
  for unit in "${REMOVED_UNITS[@]}"; do rm -f "$SYSTEMD_DIR/$unit"; done
  echo "DESTDIR set: not running systemctl."
fi

cat <<EOF

LawnBerry Pi services installed and enabled (not started).

To start them (database first):
  sudo systemctl start lawnberry-database lawnberry-backend lawnberry-health \\
       lawnberry-frontend lawnberry-camera lawnberry-remote-access
  sudo systemctl start lawnberry-cert-renewal.timer lawnberry-backup.timer halow-link-watch.timer

On the shed Pi do NOT stop or restart lawnberry-backend without physical access: it also
stops lawnberry-camera (PartOf=), and on 2026-10-06 the HaLow radio failed about 10 s after
such a stop and stayed dark until a power cycle (docs/autonomy-toolchain-matrix.md).

Status and logs:
  sudo systemctl status lawnberry-backend lawnberry-camera lawnberry-frontend
  sudo journalctl -u lawnberry-backend -f
EOF
