# Packaging Notes

## macOS DMG

- Target: ad-hoc signed Apple Silicon (`arm64`) test build
- Build command: `./scripts/build_macos_dmg.sh`
- Outputs:
  - `dist/EverMate.app`
  - `dist/EverMate-macOS-arm64.dmg`
  - `dist/EverMate-macOS-arm64-signing-report.txt`

The build script creates a clean local build venv in `.build-venv`, packages the
GUI with PyInstaller, bundles the app assets, applies an explicit ad-hoc
signature, verifies it with `codesign --verify --deep --strict`, records
`spctl --assess` output, and wraps the result in a DMG with an `/Applications`
shortcut for drag-and-drop installation.

Because the build is not signed with a Developer ID certificate and is not
notarized, Gatekeeper may still block the first launch. The intended user path
is: drag to `Applications`, then right-click `Open` once, then launch normally.

If the workspace lives inside a Desktop/iCloud-synced folder, macOS may attach
Finder metadata that breaks ad-hoc signatures on copied `.app` bundles. In that
case the script keeps the authoritative signed bundle in
`~/Library/Caches/EverMateBuild/EverMate.app` and exposes `dist/EverMate.app`
as a symlink to that signed copy.

EverMate keeps writable state outside the app bundle. Source runs and bundled
builds now share the same default location:

`~/Library/Application Support/EverMate/memory`

## Toward a distributable build

The current pipeline stops at an ad-hoc signature. For public distribution:

1. Enroll in the Apple Developer Program and create a `Developer ID
   Application` certificate.
2. Sign with the hardened runtime (`codesign --options runtime` per binary;
   prefer signing inside-out rather than `--deep`).
3. Notarize (`xcrun notarytool submit --wait`) and staple
   (`xcrun stapler staple dist/EverMate.app`).
4. Re-wrap the DMG and notarize the DMG as well.

Until then, the right-click → Open flow documented in the README is the
expected user path.
