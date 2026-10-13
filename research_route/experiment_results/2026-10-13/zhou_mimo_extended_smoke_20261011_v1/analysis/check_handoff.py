"""Verify handoff arguments and local documentation without launching work."""
import ast
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
commands = [json.loads(line) for line in (OUT/"powershell_argv.jsonl").read_text(encoding="utf-8").splitlines()]
assert len(commands)==20
new_ids = []
for command in commands:
    assert "--smoke" not in command
    config = command[command.index("--config")+1]
    assert (ROOT/config).is_file()
    identifier = command[command.index("--run-id")+1]
    assert not (ROOT/"runs"/identifier).exists(), "a formal command unexpectedly ran"
    new_ids.append(identifier)
assert len(set(new_ids))==20
for command in commands[:2]:
    start = command.index("--train-run-id")+1
    end = command.index("--split")
    assert end-start==7
    assert len(set(command[start:end]))==7
tests = [command for command in commands if "--split" in command and command[command.index("--split")+1]=="test"]
assert len(tests)==6 and all("--tuning-run-id" in command for command in tests)
assert sum(command[0]=="train" for command in commands)==4

files = [ROOT/name for name in ("README.md","START_HERE.md","research_route/README.md",
         "research_route/00_研究路线总纲.md","research_route/01_AI_Edge与太赫兹DMA_RHS的概念和公共模型.md",
         "research_route/02_文献地图与研究现状比较框架.md","research_route/90_统一仿真与复现规范.md",
         "research_route/91_方向一实验运行索引.md","research_route/99_研究决策记录.md",
         "research_route/19i_方向一传统参照与分步场景迁移.md")]
links = 0
for path in files:
    text = path.read_text(encoding="utf-8")
    assert "\ufffd" not in text
    for target in re.findall(r"(?<!!)\[[^\]]*\]\(([^)]+)\)",text):
        if re.match(r"[a-z]+://",target):
            continue
        target = target.split("#",1)[0]
        if target:
            assert (path.parent/target).exists(), (path,target)
            links += 1
for path in [ROOT/"scripts/run_zhou2025_repro.py", *[ROOT/"src/zhou2025_repro"/name for name in
              ("mimo.py","data.py","parametric_baselines.py","wideband.py")],
             *[ROOT/"tests"/name for name in ("test_zhou2025_extensions.py","test_zhou2025_parametric_baselines.py",
                                              "test_zhou2025_wideband.py","test_zhou2025_repro.py")]]:
    ast.parse(path.read_text(encoding="utf-8"))
with (ROOT/"research_route/literature_evidence_matrix.csv").open(encoding="utf-8-sig",newline="") as handle:
    evidence_count = len(list(csv.DictReader(handle)))
bib_count = len(re.findall(r"(?m)^@",(ROOT/"research_route/references.bib").read_text(encoding="utf-8")))
assert evidence_count==bib_count==50
result = {"powershell_blocks_syntax_passed":4,"commands_argument_expansion_passed":20,
          "formal_runs_launched":0,"local_document_links_checked":links,"python_syntax_files_checked":9,
          "evidence_records":evidence_count,"bib_records":bib_count,"unit_tests_passed":40}
(OUT/"handoff_checks.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result))
