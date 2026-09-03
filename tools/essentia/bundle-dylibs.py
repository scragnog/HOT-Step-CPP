#!/usr/bin/env python3
"""Make a macOS binary self-contained: copy every non-system dylib it loads
(transitively) next to it and rewrite the load paths to @loader_path.

Adapted from the script @beaudamion posted in
https://github.com/scragnog/HOT-Step-CPP/issues/122 for the Apple Silicon
Essentia build. Two changes from the original: any dependency outside
/usr/lib and /System is bundled (not only /opt/homebrew, so an Intel Homebrew
under /usr/local or a build-tree libessentia.dylib is caught too), and every
rewritten file is re-signed ad hoc, since install_name_tool invalidates the
signature and arm64 refuses to run an unsigned binary.

Usage: bundle-dylibs.py <dir> <binary-name>
"""
import os
import shutil
import subprocess
import sys

SYSTEM_PREFIXES = ("/usr/lib/", "/System/")


def deps(path):
    out = subprocess.run(["otool", "-L", path], capture_output=True, text=True, check=True).stdout
    found = []
    for line in out.splitlines()[1:]:
        lib = line.strip().split(" ")[0]
        if not lib or lib.startswith("@") or lib.startswith(SYSTEM_PREFIXES):
            continue
        found.append(lib)
    return found


def main(target_dir, bin_name):
    bin_path = os.path.join(target_dir, bin_name)
    copied = {}
    queue = deps(bin_path)
    while queue:
        lib = queue.pop(0)
        name = os.path.basename(lib)
        if name in copied:
            continue
        src = os.path.realpath(lib)
        if not os.path.exists(src):
            print(f"warning: {lib} not found; leaving the reference as is", file=sys.stderr)
            continue
        dst = os.path.join(target_dir, name)
        shutil.copy2(src, dst)
        os.chmod(dst, 0o755)
        copied[name] = lib
        queue.extend(deps(dst))

    files = [bin_path] + [os.path.join(target_dir, n) for n in copied]
    for f in files:
        for d in deps(f):
            subprocess.run(["install_name_tool", "-change", d, f"@loader_path/{os.path.basename(d)}", f],
                           check=False, capture_output=True)
        if f != bin_path:
            subprocess.run(["install_name_tool", "-id", f"@loader_path/{os.path.basename(f)}", f],
                           check=False, capture_output=True)
        subprocess.run(["codesign", "--force", "--sign", "-", f], check=False, capture_output=True)

    leftovers = [d for f in files for d in deps(f)]
    if leftovers:
        print("error: unresolved non-system references remain:", *leftovers, sep="\n  ", file=sys.stderr)
        sys.exit(1)
    print(f"bundled {len(copied)} dylibs into {target_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
