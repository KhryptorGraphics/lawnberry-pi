#!/usr/bin/env bash
# LawnBerry Pi certificate renewal + validation + fallback
# - Tries to renew Let's Encrypt certificates via certbot
# - Validates expiry and reloads nginx on success
# - Falls back to self-signed certificate if renewal fails or cert expiring imminently
# - Emits syslog messages for monitoring/alerts
#
# WHO RUNS THIS, AND WHAT IT TRUSTS
#   lawnberry-cert-renewal.service runs this as ROOT (daily, from lawnberry-cert-renewal.timer,
#   and once at boot) from the copy that systemd/install_services.sh installs as
#   /usr/local/sbin/lawnberry-renew-certificates.sh (root:root 0755). This file is the source
#   of that copy (re-run the installer after changing it) and still works when run by hand:
#   sudo bash scripts/renew_certificates.sh
#   The checkout and the .env in it belong to the service account, whose backend is a
#   network-facing process. Root therefore never runs anything from the checkout and treats
#   .env as DATA: it is never sourced, evaluated or executed, and the only things taken from
#   it are LB_DOMAIN and DOMAIN, after the strict parse and the validation described below.
#
# USAGE
#   renew_certificates.sh                  renew/validate/fall back (needs root)
#   renew_certificates.sh --print-config   print what this run would use and exit 0; changes
#                                          nothing, needs no root, no openssl, no nginx:
#                                            domain=<host name, or empty>
#                                            env_file=<path, or empty>
#                                          (warnings about the configuration go to stderr)
#   renew_certificates.sh --help
#
# CONFIGURATION (the only inputs)
#   LB_DOMAIN, DOMAIN   the certificate's host name: it names /etc/letsencrypt/live/<name> and is
#                       the CN of the self-signed fallback. Optional: with none, the first
#                       directory under /etc/letsencrypt/live is used. Order, first match wins:
#                       LB_DOMAIN in the environment, LB_DOMAIN in the env file, DOMAIN in the
#                       environment, DOMAIN in the env file. The environment (the unit, or sudo -E
#                       when run by hand) is trusted more than the file, so it wins over it.
#   LAWNBERRY_ENV_FILE  absolute path of the env file. The unit sets it to <checkout>/.env.
#                       Unset: <checkout>/.env when this file sits in a checkout's scripts/
#                       directory, otherwise no file. Set but empty: no file is read.
#
# THE ENV FILE is read line by line by a strict parser (read_env_file), never by the shell:
#   * a symbolic link, a directory or any other non-regular file is refused (warning);
#   * at most the first 32 KiB are read, and a line of 1024 characters or more is skipped
#     unread (a service-owned file must not cost the root job time or memory); a skipped
#     LB_DOMAIN=/DOMAIN= line counts as an invalid value;
#   * KEY=value, KEY = value, KEY='value' and KEY="value" (one pair of matching quotes, no
#     escapes) are understood. Blanks around the value and the CR of a CRLF line end are
#     dropped. There are no inline comments. "export KEY=value" is NOT understood: systemd's
#     EnvironmentFile (which reads this same file for the backend) and scripts/provision_env.py
#     do not understand it either, so one file keeps one syntax. It is ignored with a warning;
#   * only LB_DOMAIN and DOMAIN are looked at; every other line is ignored, whatever it holds;
#   * the LAST assignment of a key wins (as with systemd); an empty value leaves it unset;
#   * a value must be a host name (valid_domain). Anything else is ignored with a warning that
#     never repeats the value, and the key counts as unset. The same check applies to
#     LB_DOMAIN and DOMAIN taken from the environment.
set -euo pipefail
export LC_ALL=C # byte-wise matching: [A-Za-z0-9] means ASCII letters and digits only

LOG_TAG="lawnberry-cert"
ENV_FILE_MAX_BYTES=32768
ENV_LINE_MAX=1024
PRINT_ONLY=0
log_info(){ echo "[INFO] $*"; logger -t "$LOG_TAG" "$*" || true; }
log_warn(){ echo "[WARN] $*"; logger -p daemon.warning -t "$LOG_TAG" "$*" || true; }
log_err(){ echo "[ERR ] $*"; logger -p daemon.err -t "$LOG_TAG" "$*" || true; }

usage(){
  # builtins only: --help must work with nothing on PATH, like --print-config
  printf '%s\n' \
    'Usage: renew_certificates.sh [--print-config | --help]' \
    '' \
    '  (no option)     renew or validate the certificate, fall back to a self-signed one and' \
    '                  reload nginx. Needs root.' \
    '  --print-config  print the configuration this run would use and exit 0, changing nothing' \
    '                  (no root, openssl or nginx needed):' \
    '                    domain=<host name from LB_DOMAIN/DOMAIN, or empty>' \
    '                    env_file=<the env file it reads, or empty>' \
    '  -h, --help      this text' \
    '' \
    'The domain is read from LB_DOMAIN / DOMAIN in the environment, then from the env file named by' \
    'LAWNBERRY_ENV_FILE (default: .env in the checkout this script is in). The file is parsed, never' \
    'executed; see the comments at the top of this script.'
}

case "${1:-}" in
  "") ;;
  --print-config) PRINT_ONLY=1 ;;
  -h | --help) usage; exit 0 ;;
  *) echo "renew_certificates.sh: unknown option: $1" >&2; usage >&2; exit 2 ;;
esac
if (($# > 1)); then
  echo "renew_certificates.sh: takes at most one option" >&2
  exit 2
fi

# ---------------------------------------------------------------------------------------
# configuration: the domain and the env file it may come from
# ---------------------------------------------------------------------------------------

# A problem with the configuration. It never carries a value from the env file.
config_warn(){
  if ((PRINT_ONLY)); then echo "[WARN] $*" >&2; else log_warn "$*"; fi
}

# valid_domain VALUE: a host name: 1-253 characters, ASCII letters, digits, dots and hyphens,
# starting and ending with a letter or digit, no "..". That is the set of
# ^[A-Za-z0-9]([A-Za-z0-9.-]{0,251}[A-Za-z0-9])?$ (the length is tested apart because a repeat
# bound that large is slow to compile). It leaves out everything a path, a shell or an openssl
# subject (/CN=...) would give a meaning to: slash, space, quotes, $, backtick, ;, & and a
# leading dot or hyphen.
valid_domain(){
  local re='^[A-Za-z0-9]([A-Za-z0-9.-]*[A-Za-z0-9])?$'
  [[ ${#1} -le 253 && $1 =~ $re && $1 != *..* ]]
}

# accept_domain WHAT VALUE: ACCEPTED becomes VALUE when it is a host name, else "" (an empty
# VALUE silently, anything else with a warning that does not repeat it).
accept_domain(){
  ACCEPTED=""
  if [[ -z $2 ]]; then return 0; fi
  if valid_domain "$2"; then
    ACCEPTED=$2
  else
    config_warn "$1 is not a valid host name; ignored"
  fi
}

# resolve_env_file: ENV_FILE is the file to read ("" for none); ENV_FILE_EXPLICIT is 1 when
# LAWNBERRY_ENV_FILE named it.
resolve_env_file(){
  local src dir path_re='^/[[:print:]]*$'
  ENV_FILE=""
  ENV_FILE_EXPLICIT=0
  if [[ -v LAWNBERRY_ENV_FILE ]]; then
    if [[ -z $LAWNBERRY_ENV_FILE ]]; then return 0; fi
    if [[ $LAWNBERRY_ENV_FILE =~ $path_re ]]; then
      ENV_FILE=$LAWNBERRY_ENV_FILE
      ENV_FILE_EXPLICIT=1
      return 0
    fi
    config_warn "LAWNBERRY_ENV_FILE is not an absolute path of printable characters; ignored"
  fi
  # the checkout this script is in: <checkout>/scripts/renew_certificates.sh
  src=${BASH_SOURCE[0]}
  dir=.
  if [[ $src == */* ]]; then dir=${src%/*}; fi
  if dir=$(cd -- "$dir" 2>/dev/null && pwd -P) && [[ ${dir##*/} == scripts ]]; then
    ENV_FILE="${dir%/*}/.env"
  fi
}

# set_file_domain KEY VALUE: record what the env file says for LB_DOMAIN or DOMAIN
set_file_domain(){
  if [[ $1 == LB_DOMAIN ]]; then FILE_LB_DOMAIN=$2; else FILE_DOMAIN=$2; fi
}

# read_env_file FILE: FILE_LB_DOMAIN and FILE_DOMAIN get the host names that the LAST
# LB_DOMAIN= / DOMAIN= lines of FILE hold ("" when absent, empty, invalid or the file is
# refused). Nothing in FILE is ever executed or expanded: each line is matched against fixed
# patterns and handled as data, one line at a time, within a byte budget. A symbolic-link
# check cannot be made race-free in bash; what it guards is the ordinary case, and what the
# file can say stays bounded anyway (a host name; nothing else is used).
read_env_file(){
  local file=$1 line chunk key raw val lineno=0 used=0 long
  local assign_re='^[[:space:]]*(LB_DOMAIN|DOMAIN)[[:space:]]*=[[:space:]]*(.*)$'
  local export_re='^[[:space:]]*export[[:space:]]+(LB_DOMAIN|DOMAIN)[[:space:]]*='
  local key_re='^[[:space:]]*(LB_DOMAIN|DOMAIN)[[:space:]]*='
  local single_re="^'([^']*)'\$"
  local double_re='^"([^"]*)"$'
  FILE_LB_DOMAIN=""
  FILE_DOMAIN=""
  if [[ -z $file ]]; then return 0; fi
  if [[ -L $file ]]; then
    config_warn "env file $file is a symbolic link; not read"
    return 0
  fi
  if [[ ! -e $file ]]; then
    if ((ENV_FILE_EXPLICIT)); then config_warn "env file $file does not exist"; fi
    return 0
  fi
  if [[ ! -f $file ]]; then
    config_warn "env file $file is not a regular file; not read"
    return 0
  fi
  # not "if ! { ...; } <file": bash drops the "!" when the redirection itself fails
  if {
    while IFS= read -r -n "$ENV_LINE_MAX" line || [[ -n $line ]]; do
      lineno=$((lineno + 1))
      used=$((used + ${#line} + 1))
      long=0
      if ((${#line} >= ENV_LINE_MAX)); then
        # No assignment of ours is that long: skip to the end of the line without keeping it
        # (read -n stops after ENV_LINE_MAX characters), but never past the byte budget.
        long=1
        while ((used <= ENV_FILE_MAX_BYTES)) && IFS= read -r -n "$ENV_LINE_MAX" chunk &&
          ((${#chunk} >= ENV_LINE_MAX)); do
          used=$((used + ${#chunk}))
        done
      fi
      if ((used > ENV_FILE_MAX_BYTES)); then
        config_warn "env file $file is larger than $ENV_FILE_MAX_BYTES bytes; the rest is not read"
        break
      fi
      if ((long)); then
        # A long value of ours is an invalid one (the key counts as unset); any other long line
        # is not ours and is skipped without a word.
        if [[ $line =~ $key_re ]]; then
          key=${BASH_REMATCH[1]}
          config_warn "$key on line $lineno of $file has $ENV_LINE_MAX characters or more; ignored"
          set_file_domain "$key" ""
        fi
        continue
      fi
      if [[ $line =~ $assign_re ]]; then
        key=${BASH_REMATCH[1]}
        raw=${BASH_REMATCH[2]}
        raw=${raw%"${raw##*[![:space:]]}"} # trailing blanks, and the CR of a CRLF line end
        val=$raw
        if [[ $raw =~ $single_re ]]; then
          val=${BASH_REMATCH[1]}
        elif [[ $raw =~ $double_re ]]; then
          val=${BASH_REMATCH[1]}
        fi
        accept_domain "$key on line $lineno of $file" "$val"
        set_file_domain "$key" "$ACCEPTED"
      elif [[ $line =~ $export_re ]]; then
        config_warn "${BASH_REMATCH[1]} on line $lineno of $file: 'export' lines are not read; write ${BASH_REMATCH[1]}=value"
      fi
    done
    :
  } <"$file"; then
    :
  else
    config_warn "env file $file cannot be read"
  fi
}

# resolve_config: CERT_DOMAIN ("" for none) and ENV_FILE
resolve_config(){
  local env_lb env_dom
  accept_domain "LB_DOMAIN in the environment" "${LB_DOMAIN-}"
  env_lb=$ACCEPTED
  accept_domain "DOMAIN in the environment" "${DOMAIN-}"
  env_dom=$ACCEPTED
  resolve_env_file
  read_env_file "$ENV_FILE"
  CERT_DOMAIN=${env_lb:-${FILE_LB_DOMAIN:-${env_dom:-$FILE_DOMAIN}}}
}

resolve_config
if ((PRINT_ONLY)); then
  printf 'domain=%s\nenv_file=%s\n' "$CERT_DOMAIN" "$ENV_FILE"
  exit 0
fi

# ---------------------------------------------------------------------------------------
# renewal
# ---------------------------------------------------------------------------------------

require_cmd(){ command -v "$1" >/dev/null 2>&1 || { log_err "Missing command: $1"; exit 1; }; }

require_cmd openssl
require_cmd nginx

# Determine Let's Encrypt live path
LE_PATH=""
if [[ -n "$CERT_DOMAIN" && -d "/etc/letsencrypt/live/$CERT_DOMAIN" ]]; then
  LE_PATH="/etc/letsencrypt/live/$CERT_DOMAIN"
else
  # Pick the first live directory if domain not specified
  FIRST_LIVE=$(find /etc/letsencrypt/live -mindepth 1 -maxdepth 1 -type d | head -n 1 || true)
  if [[ -n "$FIRST_LIVE" ]]; then
    LE_PATH="$FIRST_LIVE"
  fi
fi

SELF_DIR="/etc/lawnberry/certs/selfsigned"
# Under lawnberry-cert-renewal.service /etc is read-only except /etc/lawnberry, /etc/nginx and
# /etc/letsencrypt, and a missing /etc/lawnberry cannot be created there: say so, not just "EROFS".
mkdir -p "$SELF_DIR" || {
  log_err "Cannot create $SELF_DIR (scripts/setup_https.sh creates it; as a systemd job this script cannot create /etc/lawnberry)"
  exit 1
}

generate_self_signed(){
  local cn=${CERT_DOMAIN:-lawnberry.local}
  log_warn "Generating self-signed certificate for CN=$cn as fallback"
  openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout "$SELF_DIR/privkey.pem" \
    -out "$SELF_DIR/fullchain.pem" \
    -subj "/CN=$cn"
}

switch_nginx_to_self_signed(){
  local ts; ts=$(date +%s)
  local files=(/etc/nginx/nginx.conf /etc/nginx/sites-available/* /etc/nginx/sites-enabled/*)
  for f in "${files[@]}"; do
    [[ -f "$f" ]] || continue
    if grep -q "/etc/letsencrypt/live/" "$f"; then
      cp "$f" "$f.bak.$ts" || true
      sed -i "s#ssl_certificate\s\+/etc/letsencrypt/live/[^;]*/fullchain.pem;#ssl_certificate     $SELF_DIR/fullchain.pem;#" "$f" || true
      sed -i "s#ssl_certificate_key\s\+/etc/letsencrypt/live/[^;]*/privkey.pem;#ssl_certificate_key $SELF_DIR/privkey.pem;#" "$f" || true
    fi
  done
  if nginx -t; then systemctl reload nginx; fi
}

check_cert_expiry(){
  local pem=$1
  local seconds=$2
  if [[ ! -f "$pem" ]]; then
    return 2
  fi
  # returns 0 if certificate will NOT expire in next $seconds
  if openssl x509 -checkend "$seconds" -noout -in "$pem" >/dev/null 2>&1; then
    return 0
  else
    return 1
  fi
}

renew(){
  if ! command -v certbot >/dev/null 2>&1; then
    log_info "Installing certbot..."
    apt-get update -y && apt-get install -y certbot
  fi
  log_info "Running certbot renew --quiet"
  if ! certbot renew --quiet; then
    log_warn "certbot renew returned non-zero"
    return 1
  fi
  if nginx -t; then
    systemctl reload nginx
  fi
}

main(){
  # If no LE path exists yet, just ensure self-signed exists and exit 0
  if [[ -z "$LE_PATH" ]]; then
    log_warn "No Let's Encrypt live directory found; ensuring self-signed certificate exists"
    [[ -f "$SELF_DIR/fullchain.pem" && -f "$SELF_DIR/privkey.pem" ]] || generate_self_signed
    switch_nginx_to_self_signed
    exit 0
  fi

  local pem="$LE_PATH/fullchain.pem"
  local key="$LE_PATH/privkey.pem"
  if [[ ! -f "$pem" || ! -f "$key" ]]; then
    log_warn "LE files missing; falling back to self-signed"
    generate_self_signed
    switch_nginx_to_self_signed
    exit 0
  fi

  # If expires in less than 30 days, try renew
  if ! check_cert_expiry "$pem" $((30*24*3600)); then
    log_info "Certificate expiring within 30 days; attempting renewal"
    renew || true
  fi

  # Validate again with 7-day window
  if check_cert_expiry "$pem" $((7*24*3600)); then
    log_info "Certificate is valid for at least 7 more days; keeping Let's Encrypt configuration"
    if nginx -t; then systemctl reload nginx; fi
    exit 0
  else
    log_warn "Certificate still expiring soon (<7 days) or invalid after renewal"
    generate_self_signed
    switch_nginx_to_self_signed
    exit 0
  fi
}

main
