import json

path = "/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811/state/league.json"
data = json.load(open(path, encoding="utf-8"))
out = {"sourceRound": data.get("trainingControlSourceRoundId"), "chains": {}}
for name, chain in data["chains"].items():
    control = chain.get("trainingControl", {})
    rollout = control.get("rollout", {})
    learner = control.get("learner", {})
    evidence = control.get("evidence", {})
    out["chains"][name] = {
        "frozen": evidence.get("frozenScoreRate"),
        "delta": evidence.get("deltaVsPrevious"),
        "seatGap": evidence.get("seatGap"),
        "selfPlay": rollout.get("selfPlayFraction"),
        "seat1Fraction": rollout.get("learnerSeat1Fraction"),
        "minDecisions": learner.get("minDecisions"),
        "seat1Weight": learner.get("seat1Weight"),
        "anchor": learner.get("teacherAnchorCoefficient"),
        "lr": learner.get("learningRate"),
    }
print(json.dumps(out, separators=(",", ":")))
