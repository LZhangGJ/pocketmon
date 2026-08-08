# Deprecated prompt

Do not use the previous direct-training workflow. It incorrectly treated the teammate ZIP as if it were an already integrated runnable artifact.

Use the corrected Windows Codex prompt:

```text
experiment7/CODEX_WINDOWS_PROMPT.md
```

The ZIP is only a code-review source snapshot. The first task is to validate it, extract its source into ordinary Git-tracked files, review and adapt that source to the pocketmon replay/runtime interfaces, add tests, and commit the integrated implementation. Only after that fixed commit passes smoke tests may the remote Linux servers perform cache generation, GPU training, export and Arena evaluation.
