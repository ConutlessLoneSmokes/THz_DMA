"""Record PowerShell argument expansion without running an experiment."""
import json
import sys
from pathlib import Path

with (Path(__file__).parent/"powershell_argv.jsonl").open("a",encoding="utf-8") as handle:
    handle.write(json.dumps(sys.argv[1:])+"\n")
