You are reviewing a proposed change to a Python library before it is merged. You did not write it.

You are given two things:

1. SPECIFICATION: what the code stated about itself before the change. That means the signatures and
   docstrings of every function and class the change touches, taken from the unchanged code. Where the
   change edits a docstring, the new version is shown as well.
2. CHANGE: the diff against the unchanged code. Each line is prefixed with its line number in the
   changed file. Added lines are marked `+` and removed lines `-`; a removed line has no number.

Report every place where the changed code does something the specification or the surrounding code
says it should not do, or where the change introduces a defect or a security weakness a careful
reviewer would block the merge for.

Do not report style, naming, missing tests, or performance. Do not report something only because it
is new; an intended change that the specification or the change's own docstrings and tests announce is
not a finding. If you are not sure that a line breaks something the specification states, leave it out.

Answer with one JSON object and nothing else:

{"findings": [{"file": "<path as shown in the diff>", "line": <line number in the changed file>,
               "quote": "<the line of the diff you mean, copied exactly, without its number and marker>",
               "concern": "<one sentence: what is stated, and what the change does instead>"}]}

For a removed line, give the number of the nearest numbered line above it and quote the removed line.
If nothing qualifies, answer {"findings": []}.
