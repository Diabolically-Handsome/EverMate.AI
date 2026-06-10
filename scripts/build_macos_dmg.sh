#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_NAME="EverMate"
APP_BUNDLE_ID="ai.evermate.desktop"
VENV_DIR="${ROOT}/.build-venv"
BUILD_DIR="${ROOT}/build/macos"
DIST_DIR="${ROOT}/dist"
SIGN_CACHE_DIR="${HOME}/Library/Caches/EverMateBuild"
PYI_DIST_DIR="${SIGN_CACHE_DIR}/pyinstaller-dist"
PYI_WORK_DIR="${SIGN_CACHE_DIR}/pyinstaller-build"
STAGE_DIR="${BUILD_DIR}/dmg-root"
ICONSET_DIR="${BUILD_DIR}/EverMate.iconset"
ICNS_PATH="${BUILD_DIR}/EverMate.icns"
DMG_PATH="${DIST_DIR}/EverMate-macOS-arm64.dmg"
SIGNING_REPORT="${DIST_DIR}/EverMate-macOS-arm64-signing-report.txt"
PYTHON_BIN="${PYTHON_BIN:-python3}"

assess_spctl() {
  local type="$1"
  local target="$2"
  local output=""

  if output=$(spctl --assess --type "${type}" -vv "${target}" 2>&1); then
    printf 'status=accepted\n%s\n' "${output}"
  else
    local exit_code=$?
    printf 'status=rejected (exit %s)\n%s\n' "${exit_code}" "${output}"
  fi
}

verify_codesign() {
  local target="$1"
  local output=""

  if output=$(codesign --verify --deep --strict "${target}" 2>&1); then
    printf 'status=accepted\n'
  else
    local exit_code=$?
    printf 'status=rejected (exit %s)\n%s\n' "${exit_code}" "${output}"
  fi
}

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This build script only supports macOS." >&2
  exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "This build targets Apple Silicon (arm64) only." >&2
  exit 1
fi

echo "[1/6] Preparing build environment..."
mkdir -p "${BUILD_DIR}" "${DIST_DIR}"

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${ROOT}/requirements.txt" pyinstaller
xattr -cr "${ROOT}/assets" "${VENV_DIR}" || true

echo "[2/6] Generating macOS app icon..."
rm -rf "${ICONSET_DIR}" "${ICNS_PATH}"
mkdir -p "${ICONSET_DIR}"

SOURCE_ICON="${ROOT}/assets/icons/app_icon_256.png"
if [[ ! -f "${SOURCE_ICON}" ]]; then
  echo "Missing source icon: ${SOURCE_ICON}" >&2
  exit 1
fi

make_icon() {
  local size="$1"
  local name="$2"
  sips -z "${size}" "${size}" "${SOURCE_ICON}" --out "${ICONSET_DIR}/${name}" >/dev/null
}

make_icon 16 "icon_16x16.png"
make_icon 32 "icon_16x16@2x.png"
make_icon 32 "icon_32x32.png"
make_icon 64 "icon_32x32@2x.png"
make_icon 128 "icon_128x128.png"
make_icon 256 "icon_128x128@2x.png"
make_icon 256 "icon_256x256.png"
make_icon 512 "icon_256x256@2x.png"
make_icon 512 "icon_512x512.png"
make_icon 1024 "icon_512x512@2x.png"

iconutil -c icns "${ICONSET_DIR}" -o "${ICNS_PATH}"

echo "[3/6] Building ${APP_NAME}.app with PyInstaller..."
rm -rf "${DIST_DIR}/${APP_NAME}" "${DIST_DIR}/${APP_NAME}.app" "${PYI_DIST_DIR}" "${PYI_WORK_DIR}"
mkdir -p "${SIGN_CACHE_DIR}"
pyinstaller --noconfirm --clean --distpath "${PYI_DIST_DIR}" --workpath "${PYI_WORK_DIR}" "${ROOT}/EverMate.spec"

RAW_APP_BUNDLE="${PYI_DIST_DIR}/${APP_NAME}.app"
SIGNED_APP_BUNDLE="${SIGN_CACHE_DIR}/${APP_NAME}.app"
APP_BUNDLE_LINK="${DIST_DIR}/${APP_NAME}.app"

if [[ ! -d "${RAW_APP_BUNDLE}" ]]; then
  echo "Expected app bundle not found: ${RAW_APP_BUNDLE}" >&2
  exit 1
fi

rm -rf "${SIGNED_APP_BUNDLE}"
mv "${RAW_APP_BUNDLE}" "${SIGNED_APP_BUNDLE}"
xattr -cr "${SIGNED_APP_BUNDLE}" || true

echo "[4/6] Applying ad-hoc signature..."
codesign --force --deep --sign - --timestamp=none "${SIGNED_APP_BUNDLE}"
codesign --verify --deep --strict "${SIGNED_APP_BUNDLE}"

rm -rf "${APP_BUNDLE_LINK}"
ln -sfn "${SIGNED_APP_BUNDLE}" "${APP_BUNDLE_LINK}"

cat > "${SIGNING_REPORT}" <<EOF
EverMate macOS Signing Report
=============================

Built: $(date)
Signed app bundle: ${SIGNED_APP_BUNDLE}
Dist app alias: ${APP_BUNDLE_LINK}
Bundle ID: ${APP_BUNDLE_ID}
Signing mode: ad-hoc
Notarized: no

[codesign verify]
$(verify_codesign "${SIGNED_APP_BUNDLE}")

[spctl assess app]
$(assess_spctl execute "${SIGNED_APP_BUNDLE}")
EOF

echo "[5/6] Staging DMG contents..."
rm -rf "${STAGE_DIR}" "${DMG_PATH}"
mkdir -p "${STAGE_DIR}"
cp -R "${SIGNED_APP_BUNDLE}" "${STAGE_DIR}/"
ln -s /Applications "${STAGE_DIR}/Applications"

cat > "${STAGE_DIR}/README.txt" <<'EOF'
EverMate macOS Test Build (Ad-hoc Signed)
=========================================

1. Open this DMG.
2. Drag EverMate.app into Applications.
3. Launch it from /Applications.
4. This build is ad-hoc signed but not Apple notarized, so macOS may block the
   first launch:
   - Right-click the app and choose "Open", then confirm.
5. After that first approval, you can launch it normally by double-clicking.
6. EverMate stores writable memory and app state in:
   ~/Library/Application Support/EverMate/

Ollama is not bundled. Start your local Ollama server separately if you want
local model-backed chat and memory analysis features.
EOF
xattr -cr "${STAGE_DIR}" || true

echo "[6/6] Creating DMG..."
hdiutil create \
  -volname "${APP_NAME}" \
  -srcfolder "${STAGE_DIR}" \
  -ov \
  -format UDZO \
  "${DMG_PATH}" >/dev/null

cat >> "${SIGNING_REPORT}" <<EOF

[spctl assess dmg]
$(assess_spctl open "${DMG_PATH}")
EOF

echo "[6/6] Done."
echo "Signed app bundle: ${SIGNED_APP_BUNDLE}"
echo "Dist app alias: ${APP_BUNDLE_LINK}"
echo "DMG: ${DMG_PATH}"
echo "Signing report: ${SIGNING_REPORT}"
