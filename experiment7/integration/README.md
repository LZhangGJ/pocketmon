# Experiment 7 pocketmon integration

Execution order:

1. `windows_controller.py probe`
2. `windows_controller.py bootstrap`
3. `windows_controller.py prepare`
4. `windows_controller.py train`
5. `windows_controller.py status`
6. Remote `finalize_candidate.py`
7. `generate_arena_schedule.py` + repository league runner
8. `summarize_challenger_results.py`

The adapters consume pocketmon's already-audited canonical replay format. They do not reimplement action/observation alignment. The high-score model and cache code remain under `experiment7/reference_impl`.
