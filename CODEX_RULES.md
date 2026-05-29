# Codex Safety Rules

1. Do not delete, rename, or move existing files unless explicitly approved.
2. Do not edit files outside this repository.
3. Do not modify real-robot deployment scripts unless explicitly approved.
4. Do not use sudo.
5. Do not run rm -rf.
6. Do not run git reset --hard or git clean -fd without explicit approval.
7. Do not install packages globally.
8. Do not modify or delete dataset files.
9. Do not create large binary files.
10. Before editing, summarize the plan and list target files.
11. After editing, summarize changed files, new classes/functions, tensor shapes, and how to run tests.
12. Add unit tests for every new module.
13. Preserve backward compatibility with the baseline implementation.
14. Never use future action or future force in inference code.
