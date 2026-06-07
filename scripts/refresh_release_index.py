#!/usr/bin/env python3
"""Refresh the "Changelog index" footer on every GitHub release.

GitHub has no built-in cross-release index, and cutting a new release makes
every prior release's index stale. This regenerates a uniform index from
``gh release list`` (versions + links) and ``CHANGELOG.md`` (per-version
blurbs), then appends it to each release body — marking that release as the
current one.

Idempotent: any existing ``## Changelog index`` section is stripped before the
fresh one is appended, so re-running never duplicates.

Usage:
    python scripts/refresh_release_index.py            # update all releases
    python scripts/refresh_release_index.py --dry-run  # print, change nothing

Requires the `gh` CLI authenticated to the repo. Run from the repo root.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

INDEX_HEADER = "## Changelog index"


def sh(args: list[str], stdin: str | None = None) -> str:
    return subprocess.run(
        args, input=stdin, capture_output=True, text=True, check=True
    ).stdout


def semver_key(tag: str) -> tuple[int, ...]:
    parts = tag.lstrip("v").split(".")
    return tuple(int(p) for p in (parts + ["0", "0", "0"])[:3] if p.isdigit())


def _trim(text: str) -> str:
    """First sentence of a blurb, capped to a one-liner."""
    text = text.replace("**", "").strip()
    if ". " in text:
        text = text.split(". ")[0]
    return text.rstrip(".").strip()[:90]


def changelog_blurb(changelog: str, version: str) -> str | None:
    """The one-line summary for a version, taken from its CHANGELOG section.

    Prefers the section's first prose paragraph (joining markdown soft-wraps);
    if the section opens straight into a list (no prose), uses the first item.
    """
    m = re.search(r"^## \[" + re.escape(version) + r"\][^\n]*\n", changelog, re.M)
    if not m:
        return None
    rest = changelog[m.end():]
    nxt = re.search(r"^## ", rest, re.M)
    section = rest[: nxt.start()] if nxt else rest

    para: list[str] = []
    started = False
    for raw in section.splitlines():
        s = raw.strip()
        if not started:
            if not s or s.startswith("#"):
                continue
            if re.match(r"^[-*]\s", s):  # section opens with a list
                item = re.sub(r"^[-*]\s+", "", s).split(" — ")[0]
                return _trim(item)
            started = True
            para.append(s)
        elif not s or s.startswith("#") or re.match(r"^[-*]\s", s):
            break  # paragraph ended
        else:
            para.append(s)
    return _trim(" ".join(para)) if para else None


def build_index(releases, repo: str, branch: str, current_tag: str) -> str:
    base = f"https://github.com/{repo}/releases/tag"
    lines = [INDEX_HEADER, ""]
    for r in releases:
        tag = r["tagName"]
        blurb = r.get("_blurb") or f"released {r['publishedAt'][:10]}"
        mark = " *(this release)*" if tag == current_tag else ""
        lines.append(f"- [{tag}]({base}/{tag}) — {blurb}{mark}")
    lines += [
        "",
        f"Full notes for every version: "
        f"[CHANGELOG.md](https://github.com/{repo}/blob/{branch}/CHANGELOG.md)",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="print, change nothing")
    args = ap.parse_args()

    info = json.loads(sh(["gh", "repo", "view", "--json", "nameWithOwner,defaultBranchRef"]))
    repo = info["nameWithOwner"]
    branch = info["defaultBranchRef"]["name"]

    releases = json.loads(
        sh(["gh", "release", "list", "--json", "tagName,publishedAt", "--limit", "200"])
    )
    if not releases:
        print("no releases found")
        return 0
    releases.sort(key=lambda r: semver_key(r["tagName"]), reverse=True)

    changelog = Path("CHANGELOG.md").read_text() if Path("CHANGELOG.md").exists() else ""
    for r in releases:
        r["_blurb"] = changelog_blurb(changelog, r["tagName"].lstrip("v"))

    for r in releases:
        tag = r["tagName"]
        body = sh(["gh", "release", "view", tag, "--json", "body", "--jq", ".body"]).rstrip("\n")
        body = body.split("\n" + INDEX_HEADER)[0].rstrip("\n")
        new = body + "\n\n" + build_index(releases, repo, branch, tag) + "\n"
        if args.dry_run:
            print(f"=== {tag} (dry-run) ===\n{build_index(releases, repo, branch, tag)}\n")
            continue
        sh(["gh", "release", "edit", tag, "--notes-file", "-"], stdin=new)
        print(f"refreshed {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
