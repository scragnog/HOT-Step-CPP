"""Add genre/bpm/key from the ACE sidecars to an existing yue2_preprocess.json
(sources[] and clips[]), so a cache made before preprocess wrote those fields
can train the upstream style template without re-preprocessing.
usage: patch_manifest_meta.py <manifest.json>"""
import sys, json, os, re, shutil
p = sys.argv[1]; m = json.load(open(p, encoding="utf-8"))
def fields(txt):
    out = {}; key = None
    for line in txt.splitlines():
        mm = re.match(r"^([A-Za-z_]+):\s*(.*)$", line)
        if mm and mm.group(1).lower() in ("caption","genre","bpm","key","lyrics","signature","is_instrumental","duration","language","title","artist","album","year","mood","tags","style","energy","time_signature","trigger"):
            key = mm.group(1).lower(); out[key] = mm.group(2).strip()
            if key == "lyrics": break
        elif key and key != "lyrics": out[key] += "\n" + line
    return {k: v.strip() for k, v in out.items()}
by_src = {}
for s in m["sources"]:
    side = os.path.splitext(s["source"])[0] + ".txt"
    f = fields(open(side, encoding="utf-8-sig").read()) if os.path.exists(side) else {}
    meta = {k: f.get(k, "") for k in ("genre", "bpm", "key")}
    s.update(meta); by_src[s["source"]] = meta
    print(f"{s['name']:<40} {meta}")
for c in m.get("clips", []):
    c.update(by_src.get(c["source"], {"genre": "", "bpm": "", "key": ""}))
shutil.copy(p, p + ".pre-meta.bak")
json.dump(m, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("patched", p, "sources", len(m["sources"]), "clips", len(m.get("clips", [])))
