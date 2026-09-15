"""Free phone notifications via ntfy.sh -- no account, no API key.

Set credentials.ntfy_topic in config.yaml (or NTFY_TOPIC in the environment)
to any hard-to-guess string, then open https://ntfy.sh/<that string> on your
phone. Messages arrive as push notifications while the job runs.

Every call is best-effort and never raises: a monitoring failure must not kill
a three-hour pipeline.
"""
from __future__ import annotations

from datetime import datetime


def notify(cfg, title, message, priority="default", tags=""):
    topic = str(cfg.get("credentials", {}).get("ntfy_topic", "") or "").strip()
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[notify {stamp}] {title} -- {message}")
    if not topic:
        return False
    try:
        import urllib.request
        req = urllib.request.Request(
            f"https://ntfy.sh/{topic}",
            data=message.encode("utf-8"),
            headers={"Title": title, "Priority": priority, "Tags": tags},
            method="POST")
        urllib.request.urlopen(req, timeout=5)
        return True
    except Exception as e:
        print(f"[notify] delivery failed ({e}) -- continuing")
        return False


def notify_step(cfg, step_id, ok, elapsed_min, detail=""):
    wanted = {str(s).lstrip("0") or "0"
              for s in cfg.get("runtime", {}).get("notify_steps", [])}
    if str(step_id).lstrip("0") not in wanted and not ok:
        pass          # always notify on failure
    elif str(step_id).lstrip("0") not in wanted:
        return
    mark = "done" if ok else "FAILED"
    tag = "white_check_mark" if ok else "rotating_light"
    notify(cfg, f"indicvad step {step_id} {mark}",
           f"{elapsed_min:.1f} min elapsed. {detail}".strip(),
           priority="default" if ok else "high", tags=tag)
