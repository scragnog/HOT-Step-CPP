#!/usr/bin/env python3
"""Make a macOS binary self-contained: copy every non-system dylib it loads
(transitively) next to it and rewrite the load paths to @loader_path.

Adapted from the script @beaudamion posted in
https://github.com/scragnog/HOT-Step-CPP/issues/122 for the Apple Silicon
Essentia build. Changes from the original:

  * any dependency outside /usr/lib and /System is bundled, not only
    /opt/homebrew, so an Intel Homebrew under /usr/local or a self-built
    FFmpeg prefix is caught too;
  * @rpath/ and @loader_path/ references are resolved, not skipped. Homebrew
    dylibs reference each other that way (libwebpmux -> @rpath/libwebp), and
    an unresolved one only shows up as "Library not loaded" at run time;
  * every rewritten file is re-signed ad hoc, since install_name_tool
    invalidates the signature and arm64 refuses to run an unsigned binary.

Usage: bundle-dylibs.py <dir> <binary-name> [search-dir ...]
  search-dir   extra directories to resolve @rpath/ references in, tried after
               the referencing library's own original directory.
"""
import os
import shutil
import subprocess
import sys

SYSTEM_PREFIXES = ("/usr/lib/", "/System/")


def load_refs(path):
    """Every install-name reference in `path` that is not a system library."""
    out = subprocess.run(["otool", "-L", path], capture_output=True, text=True, check=True).stdout
    refs = []
    for line in out.splitlines()[1:]:
        lib = line.strip().split(" ")[0]
        if not lib or lib.startswith(SYSTEM_PREFIXES) or lib.startswith("@executable_path"):
            continue
        refs.append(lib)
    return refs


def rpaths(path):
    out = subprocess.run(["otool", "-l", path], capture_output=True, text=True, check=True).stdout
    found, lines = [], out.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "cmd LC_RPATH":
            for j in range(i, min(i + 4, len(lines))):
                if lines[j].strip().startswith("path "):
                    found.append(lines[j].strip().split(" ")[1])
    return found


def resolve(ref, referrer_orig, search_dirs):
    """Original on-disk path for a reference, or None."""
    if ref.startswith("@rpath/") or ref.startswith("@loader_path/"):
        name = ref.split("/", 1)[1]
        candidates = [os.path.join(os.path.dirname(referrer_orig), name)]
        for rp in rpaths(referrer_orig):
            rp = rp.replace("@loader_path", os.path.dirname(referrer_orig))
            candidates.append(os.path.join(rp, name))
        candidates += [os.path.join(d, name) for d in search_dirs]
        for c in candidates:
            if os.path.exists(c):
                return os.path.realpath(c)
        return None
    return os.path.realpath(ref) if os.path.exists(ref) else None


def main(target_dir, bin_name, search_dirs):
    bin_path = os.path.join(target_dir, bin_name)
    copied = {}              # basename -> original path
    unresolved = []
    # (file to scan, its original location for relative resolution)
    queue = [(bin_path, bin_path)]
    while queue:
        scan, orig = queue.pop(0)
        for ref in load_refs(scan):
            name = os.path.basename(ref)
            if name in copied or name == bin_name:
                continue
            src = resolve(ref, orig, search_dirs)
            if not src:
                unresolved.append(f"{ref} (from {os.path.basename(scan)})")
                continue
            dst = os.path.join(target_dir, name)
            shutil.copy2(src, dst)
            os.chmod(dst, 0o755)
            copied[name] = src
            queue.append((dst, src))
    if unresolved:
        print("error: could not locate these references:", *unresolved, sep="\n  ", file=sys.stderr)
        sys.exit(1)

    files = [bin_path] + [os.path.join(target_dir, n) for n in copied]
    for f in files:
        for ref in load_refs(f):
            name = os.path.basename(ref)
            if name in copied:
                subprocess.run(["install_name_tool", "-change", ref, f"@loader_path/{name}", f],
                               check=False, capture_output=True)
        if f != bin_path:
            subprocess.run(["install_name_tool", "-id", f"@loader_path/{os.path.basename(f)}", f],
                           check=False, capture_output=True)
        subprocess.run(["codesign", "--force", "--sign", "-", f], check=False, capture_output=True)

    leftovers = [f"{d} (in {os.path.basename(f)})" for f in files for d in load_refs(f)
                 if not d.startswith("@loader_path/")]
    if leftovers:
        print("error: non-system references remain after rewriting:", *leftovers, sep="\n  ", file=sys.stderr)
        sys.exit(1)
    print(f"bundled {len(copied)} dylibs into {target_dir}: {' '.join(sorted(copied))}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
