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
#
# --root must be the checkout itself: a real directory that contains backend/src/main.py and is
# not a system directory or the parent of one. That is checked in every mode and a bad --root
# is a usage error (exit 2). The units run as the service account and nothing runs "chown -R"
# as root at boot any more; the only ownership change this script makes under the root is a
# one-off, bounded one inside <root>/data, <root>/logs and <root>/config (see own_state_dirs).
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

# The directories under the root that the services write to. On a real run this script creates
# them when they are missing and gives the files in them that root owns back to the service
# account (own_state_dirs). It changes ownership nowhere else.
STATE_DIRS=(data logs config)
# What --root may never be (system_dir): these top-level directories and everything below them,
SYSTEM_TREES=(/bin /boot /dev /etc /lib /lib32 /lib64 /libx32 /proc /root /run /sbin /sys)
# and these directories themselves. A checkout BELOW them is the normal layout
# (/opt/lawnberry-pi, /srv/lawnberry, /home/kp/lawnberry-pi); the directory itself never is.
# Every parent of a system directory is in one of the two lists, so "an ancestor of one" needs
# no rule of its own (system_dir adds anything directly under /usr and /var).
SYSTEM_DIRS=(/ /home /media /mnt /opt /srv /tmp /usr /var)
# Upper bound for the ownership pass over one state directory: a stalled filesystem must not
# hang the installer.
OWNERSHIP_TIMEOUT=600

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
  --root PATH        application checkout the units point at (default: this checkout). It must
                     be the checkout itself: a directory that contains backend/src/main.py and
                     is not a system directory or the parent of one (/, /etc, /home, /opt, /srv,
                     /tmp, /usr, /usr/local, /var, /var/lib, ...). A symbolic link is resolved;
                     the real path is what the units contain.
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

On a real run (not --dry-run, not --destdir) it also creates <root>/data, <root>/logs and
<root>/config when they are missing and gives the files in them that root owns back to the
service account, once. It never follows a symbolic link there (a link is an error), never
crosses into another filesystem and changes nothing else under the root: the units run no
boot-time chown.
Exit status: 0 ok, 1 error, 2 usage (including a --root that is a system directory or not a
LawnBerry checkout), 3 refused (credential or secret policy). With --dry-run the status is what
a real run would return.
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

# system_dir PATH: true when PATH may not be the application root: a system directory, the
# parent of one, or anything inside a tree the OS owns. PATH is absolute and normalised.
system_dir() {
  local path="$1" tree
  for tree in "${SYSTEM_TREES[@]}"; do
    if [[ "$path" == "$tree" || "$path" == "$tree"/* ]]; then return 0; fi
  done
  for tree in "${SYSTEM_DIRS[@]}"; do
    if [[ "$path" == "$tree" ]]; then return 0; fi
  done
  # anything directly under /usr or /var: /usr/local, /usr/share, /var/lib, /var/log, ...
  [[ "${path%/*}" == /usr || "${path%/*}" == /var ]]
}

# resolve_root: LB_ROOT becomes the real path of the LawnBerry checkout, or the script exits 2.
# The units give the service account write access to the root and this script changes ownership
# under it, so it must be the checkout itself and never a system directory. The path as typed
# and its real path (symbolic links followed) are both checked: neither "--root /home" nor a
# link that points at /usr/local gets through. The real path is what goes into the units.
resolve_root() {
  local name hint why lexical real
  if [[ -n "$LB_ROOT" ]]; then
    name="--root '$LB_ROOT'"
    hint="Pass the directory your LawnBerry checkout is in (the one that contains backend/src/main.py), for example --root /apps/lawnberry-pi, or leave --root out to use this checkout ($REPO_DIR)."
  else
    LB_ROOT="$REPO_DIR"
    name="the default application root '$LB_ROOT' (this checkout)"
    hint="Pass the directory your LawnBerry checkout is in (the one that contains backend/src/main.py), for example --root /apps/lawnberry-pi."
  fi
  why="the units give the service account write access to it and this installer changes ownership under it, so it must be your LawnBerry checkout itself, never a system directory or the parent of one."
  while [[ "$LB_ROOT" != "/" && "$LB_ROOT" == */ ]]; do LB_ROOT="${LB_ROOT%/}"; done
  command -v realpath >/dev/null 2>&1 || die "realpath (coreutils) is required to resolve --root"
  [[ "$LB_ROOT" == "/" ]] || valid_path "$LB_ROOT" ||
    usage_error "$name must be an absolute path made of letters, digits and _ . / - only. $hint"
  lexical="$(realpath -m -s -- "$LB_ROOT")" || usage_error "$name cannot be resolved. $hint"
  real="$(realpath -m -- "$LB_ROOT")" || usage_error "$name cannot be resolved. $hint"
  if system_dir "$lexical"; then
    usage_error "$name is a system directory (or inside one): $why $hint"
  fi
  if system_dir "$real"; then
    usage_error "$name resolves to '$real', a system directory (or inside one): $why $hint"
  fi
  valid_path "$real" ||
    usage_error "$name resolves to '$real', which has characters other than letters, digits and _ . / - (they would end up in the unit files). $hint"
  [[ -e "$real" ]] ||
    usage_error "$name does not exist (or this account cannot read it). $hint"
  [[ -d "$real" ]] || usage_error "$name is a file, not a directory. $hint"
  [[ -f "$real/backend/src/main.py" ]] ||
    usage_error "$name is not a LawnBerry checkout: $real/backend/src/main.py is missing or unreadable. $hint"
  if [[ "$real" != "$LB_ROOT" ]]; then
    echo "install_services.sh: NOTE: $name resolves to '$real'; the units and every message use that real path." >&2
  fi
  LB_ROOT="$real"
}
resolve_root

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

# state_dir_problem CHECK_PATH SHOW_PATH: say why CHECK_PATH cannot be one of the state
# directories, or print nothing. A symbolic link is never followed: the service account owns
# the checkout and could point it anywhere, and this script changes ownership as root.
state_dir_problem() {
  if [[ -L "$1" ]]; then
    # %q: the target is whatever the owner of the tree made it, so it is never printed raw
    printf '%s is a symbolic link (to %q); the installer will not follow it or change ownership through it. Replace it with a real directory (move the data into it, or mount the target on it) and run the installer again.\n' "$2" "$(readlink -- "$1")"
  elif [[ -e "$1" && ! -d "$1" ]]; then
    echo "$2 exists but is not a directory; the services need a directory there. Move it away and run the installer again."
  fi
}

# ownership_argv NAME: set OWN_ARGV to the command that gives the entries under ./NAME that root
# owns back to the service account. It runs with the application root as the working directory
# (own_state_dirs); the dry run prints this very array and the apply phase executes it.
#   -P            never follow a symbolic link: a link is chown'ed itself (chown -h), its
#                 target is not touched
#   -xdev         never descend into another filesystem (a mount point inside is skipped)
#   -uid 0        only what root owns; files the service account or anyone else owns stay
#   hard links    a name in this tree can be a second name for a file anywhere on the same
#                 filesystem, so a root-owned non-directory with more than one link is
#                 reported and left alone
#   -execdir      chown runs inside the directory find is standing in, on ./name only, so a
#                 directory swapped for a symbolic link after find looked at it cannot redirect
#                 it (plain -exec resolves the whole path again)
#   env PATH=...  find refuses -execdir while PATH has an empty or relative entry
# Residual risk: the service account owns this tree and can change it while the pass runs. A
# symbolic link or a renamed parent cannot make chown reach outside ./NAME. A hard link planted
# in the window between find looking at a name and chown running can still be handed over when
# fs.protected_hardlinks is 0 (check: sysctl fs.protected_hardlinks). The installer
# also runs scripts/provision_env.py from this checkout as root, so whoever controls the
# checkout is trusted at install time anyway; this pass only has to be safe against mistakes
# and leftovers.
ownership_argv() {
  OWN_ARGV=(
    /usr/bin/env PATH=/usr/sbin:/usr/bin:/sbin:/bin
    timeout "$OWNERSHIP_TIMEOUT"
    find -P "./$1" -xdev -uid 0
    '(' '(' ! -type d -links +1
    -printf 'install_services.sh: WARNING: left %p as it is (root owns it and it has more than one hard link; chown it by hand if the services should own it)\n'
    ')' -o -execdir chown -h -- "$LB_USER:$LB_GROUP" '{}' + ')'
  )
}

# own_state_dirs: create the state directories that are missing and give the files in them that
# root owns back to the service account (real run as root; never under --destdir). This
# replaces the database unit's old recursive chown of the whole checkout at every boot.
# Everything is relative to the application root after one cd -P, so a path above it that the
# service account renames or replaces afterwards cannot redirect anything; the root is checked
# again once we stand in it. mkdir never follows a link (it fails when the name exists).
own_state_dirs() {
  (
    cd -P -- "$LB_ROOT" || die "cannot enter the application root '$LB_ROOT'"
    if [[ "$(pwd -P)" != "$LB_ROOT" || ! -f backend/src/main.py ]]; then
      die "the application root '$LB_ROOT' changed while the installer was running; nothing under it was touched, run the installer again"
    fi
    local name problem
    for name in "${STATE_DIRS[@]}"; do
      problem="$(state_dir_problem "./$name" "$LB_ROOT/$name")"
      [[ -z "$problem" ]] || die "$problem"
      if [[ ! -e "./$name" ]]; then
        # created by root, so the pass below (-uid 0) hands it to the service account
        echo "Creating $LB_ROOT/$name (0755)..."
        mkdir -m 0755 -- "./$name" || die "cannot create $LB_ROOT/$name"
      fi
      echo "Giving the files in $LB_ROOT/$name that root owns back to $LB_USER:$LB_GROUP (paths in messages are relative to $LB_ROOT)..."
      ownership_argv "$name"
      # cat -v: find prints file names raw, and in this tree their names are not ours to trust
      "${OWN_ARGV[@]}" | cat -v >&2 ||
        die "changing ownership under $LB_ROOT/$name failed or took longer than ${OWNERSHIP_TIMEOUT}s (see above); nothing outside it was touched"
    done
  )
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
  STATE_FAILED=0
  if [[ -n "$DESTDIR" ]]; then
    echo "################ DESTDIR set: would not create or change ownership of ${STATE_DIRS[*]} under $LB_ROOT"
  else
    for name in "${STATE_DIRS[@]}"; do
      problem="$(state_dir_problem "$LB_ROOT/$name" "$LB_ROOT/$name")"
      if [[ -n "$problem" ]]; then
        echo "################ would REFUSE: $problem"
        STATE_FAILED=1
        continue
      fi
      if [[ ! -e "$LB_ROOT/$name" ]]; then
        echo "################ would create $LB_ROOT/$name (0755; the pass below hands it to $LB_USER:$LB_GROUP)"
      fi
      ownership_argv "$name"
      echo "################ would give the files in $LB_ROOT/$name that root owns back to $LB_USER:$LB_GROUP, as root:"
      echo "cd $LB_ROOT && $(printf '%q ' "${OWN_ARGV[@]}")"
    done
  fi
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
  # a real run stops on an unusable state directory before it provisions anything
  DRY_RC="$PROVISION_RC"
  if ((STATE_FAILED)); then DRY_RC="$EXIT_ERROR"; fi
  if ((DRY_RC == 0)); then
    echo "dry-run result: a real run would SUCCEED"
  elif ((DRY_RC == EXIT_REFUSED)); then
    echo "dry-run result: a real run would REFUSE (exit $DRY_RC); see the messages above"
  else
    echo "dry-run result: a real run would FAIL (exit $DRY_RC); see the messages above"
  fi
  exit "$DRY_RC"
fi

# ---------------------------------------------------------------------------------------
# apply. The env file comes first: provision_env.py validates everything before it writes
# anything, so a refusal leaves the machine exactly as it was.
# ---------------------------------------------------------------------------------------
if [[ -z "$DESTDIR" ]]; then # an unusable state directory stops the run before anything changes
  for name in "${STATE_DIRS[@]}"; do
    problem="$(state_dir_problem "$LB_ROOT/$name" "$LB_ROOT/$name")"
    [[ -z "$problem" ]] || die "$problem"
  done
fi

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

if [[ -z "$DESTDIR" ]]; then
  own_state_dirs || exit "$EXIT_ERROR"
else
  echo "DESTDIR set: not creating or changing ownership of ${STATE_DIRS[*]} under $LB_ROOT."
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
