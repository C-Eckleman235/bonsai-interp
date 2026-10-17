import json, subprocess, sys, pathlib
sp = pathlib.Path(sys.argv[1]); repo = pathlib.Path(sys.argv[2])
data = subprocess.run([str(repo / ".venv/Scripts/python.exe"), "-m", "mono.report_data"], cwd=repo, capture_output=True,
                      text=True, encoding="utf-8", env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}, check=True).stdout
json.loads(data)  # validate
data = data.replace("</", "<\/")  # keep </script> out of the JSON block
html = (sp / "report_template.html").read_text(encoding="utf-8").replace("__NEURON_DATA__", data)
(sp / "bonsai-neuron-inversions.html").write_text(html, encoding="utf-8")
print("built", len(html), "bytes")
