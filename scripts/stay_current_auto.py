#!/usr/bin/env python3
"""Automated daily stay-current: find the highest-scoring public NeuroGolf submission, download it,
validate it's a clean 400-task zip, and submit it IF it beats what we've already submitted.

Safe by design: Kaggle ranks a team by its BEST submission, so submitting a candidate that turns out
worse never lowers our rank (it only spends one of the ~5/day submissions). State is tracked in
artifacts/stay_current_state.json so we don't re-submit the same source twice.

Run manually:  python scripts/stay_current_auto.py
Or via cron (see scripts/install_cron.sh) for a hands-off daily run.
"""
import subprocess, re, json, sys, os, zipfile, glob, shutil, datetime

# cron runs with a minimal PATH that omits ~/.local/bin (where pipx puts `kaggle`).
# Prepend the known install dirs so subprocess(["kaggle", ...]) resolves under cron too.
for _p in ("/home/make/.local/bin", os.path.expanduser("~/.local/bin")):
    if os.path.isdir(_p) and _p not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")
KAGGLE = shutil.which("kaggle") or "/home/make/.local/bin/kaggle"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "artifacts", "stay_current_state.json")
DL = os.path.join(ROOT, "data", "auto_dl")
SUB = os.path.join(ROOT, "artifacts", "submission.zip")
COMP = "neurogolf-2026"


def log(msg):
    print(f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def parse_score(text):
    best = 0.0
    for m in re.finditer(r"\b([678][0-9]{3})[.\-_]([0-9]{2})\b", text):
        val = float(m.group(1)) + float(m.group(2)) / 100.0
        if 6000.0 <= val <= 8000.0:
            best = max(best, val)
    return best


def scan(kind):
    sort = "dateRun" if kind == "kernels" else "updated"
    out = subprocess.run([KAGGLE,kind, "list", "-s", "neurogolf", "--sort-by", sort,
                          "--page-size", "100"], capture_output=True, text=True).stdout
    rows = {}
    for line in out.splitlines():
        s = line.strip()
        if not s or s.lower().startswith("ref") or set(s) <= set("- "):
            continue
        ref = s.split()[0]
        if ref.count("/") != 1:
            continue
        sc = parse_score(s)
        if sc > 0:
            rows[(kind, ref)] = max(sc, rows.get((kind, ref), 0.0))
    return rows


def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"best_score": 0.0, "submitted_refs": []}


def fetch_submission(kind, ref):
    if os.path.isdir(DL):
        shutil.rmtree(DL)
    os.makedirs(DL, exist_ok=True)
    if kind == "kernels":
        subprocess.run([KAGGLE,"kernels", "output", ref, "-p", DL], capture_output=True, text=True)
    else:
        subprocess.run([KAGGLE,"datasets", "download", "-d", ref, "-p", DL, "--unzip"],
                       capture_output=True, text=True)
    # find a complete 400-task submission.zip (or assemble from a zip of onnx)
    for z in sorted(glob.glob(os.path.join(DL, "**", "*.zip"), recursive=True)):
        try:
            zf = zipfile.ZipFile(z)
            nums = sorted(int(os.path.basename(n)[4:7]) for n in zf.namelist()
                          if os.path.basename(n).startswith("task") and n.endswith(".onnx"))
            if sorted(set(nums)) == list(range(1, 401)):
                return z
        except Exception:
            pass
    return None


def validate(zip_path):
    z = zipfile.ZipFile(zip_path)
    ns = [n for n in z.namelist() if n.endswith(".onnx")]
    nums = sorted(int(os.path.basename(n)[4:7]) for n in ns if os.path.basename(n).startswith("task"))
    LIM = int(1.44 * 1024 * 1024)
    return (nums == list(range(1, 401)) and all(0 < z.getinfo(n).file_size <= LIM for n in ns))


def main():
    state = load_state()
    log(f"current best submitted score: {state['best_score']}")
    cands = {}
    cands.update(scan("kernels"))
    cands.update(scan("datasets"))
    ranked = sorted(cands.items(), key=lambda kv: -kv[1])
    log(f"found {len(ranked)} scored public sources; top: " +
        ", ".join(f"{ref}={sc:.2f}" for (k, ref), sc in ranked[:5]))

    for (kind, ref), sc in ranked:
        if sc <= state["best_score"] + 1e-6:
            log(f"top candidate {ref} ({sc:.2f}) does not beat our best ({state['best_score']}); stop.")
            break
        if ref in state["submitted_refs"]:
            continue
        log(f"trying {kind}:{ref} (claimed {sc:.2f}) ...")
        zp = fetch_submission(kind, ref)
        if not zp or not validate(zp):
            log(f"  {ref}: no valid 400-task submission.zip; skip.")
            state["submitted_refs"].append(ref)
            continue
        shutil.copy(zp, SUB)
        r = subprocess.run([KAGGLE,"competitions", "submit", "-c", COMP, "-f", SUB,
                            "-m", f"auto stay-current: {ref} (claimed {sc:.2f})"],
                           capture_output=True, text=True)
        ok = "Successfully submitted" in (r.stdout + r.stderr)
        log(f"  submit {'OK' if ok else 'FAILED'}: {(r.stdout + r.stderr).strip().splitlines()[-1:]}")
        if ok:
            state["best_score"] = sc
            state["submitted_refs"].append(ref)
            json.dump(state, open(STATE, "w"), indent=1)
            log(f"  submitted; new claimed best {sc:.2f}. (Verify real LB with: "
                f"kaggle competitions submissions -c {COMP})")
            return
    json.dump(state, open(STATE, "w"), indent=1)
    log("done; nothing new to submit.")


if __name__ == "__main__":
    main()
