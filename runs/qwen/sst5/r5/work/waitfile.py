"""Local helper: block (<170 s) until a log file contains a terminal marker."""
import pathlib
import re
import sys
import time

MAX_SECONDS = 165
POLL_SECONDS = 5
path, pattern = pathlib.Path(sys.argv[1]), sys.argv[2]
start = time.time()
text = ""
while time.time() - start < MAX_SECONDS:
    text = path.read_text() if path.exists() else ""
    if re.search(pattern, text):
        break
    time.sleep(POLL_SECONDS)
print("\n".join(text.splitlines()[-int(sys.argv[3]) if len(sys.argv) > 3 else -6:]))
