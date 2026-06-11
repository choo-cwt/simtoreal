# Codex Git And Handoff Workflow

Date: 2026-06-11

This document tells a future Codex session how to continue this workspace.

## Repository Location

Local workspace:

```text
/home/chichoo/squint-master6.6winproplace/squint-master
```

GitHub remote:

```text
https://github.com/choo-cwt/simtoreal.git
```

Important: this environment mounts `.git/` as a read-only tmpfs. The real Git database for this workspace is stored in:

```text
.gitrepo/
```

Always use the wrapper:

```bash
scripts/git_project.sh status
scripts/git_project.sh diff
scripts/git_project.sh log --oneline
scripts/git_project.sh add <files>
scripts/git_project.sh commit -m "message"
scripts/git_project.sh push
```

Do not use plain `git status` in this workspace unless it has been cloned normally in a different folder.

## Authentication

Use GitHub HTTPS with a personal access token when pushing.

The token must not be committed to the repository. If a token is needed:

1. Generate a GitHub classic token with `repo` permission.
2. Run:

```bash
scripts/git_project.sh push
```

3. When Git asks for username, use:

```text
choo-cwt
```

4. When Git asks for password, paste the token.
5. Delete or rotate the token after use if it was shared in chat.

Never write the token into `README.md`, scripts, docs, `.git/config`, or remote URLs.

## Files That Should Stay Out Of Git

These are ignored on purpose:

```text
runs/
*.pt
*.pth
*.ckpt
*.mp4
*.avi
*.mov
*.mkv
*.zip
*.pdf
__pycache__/
squint-master-0/
.gitrepo/
.agents/
.codex/
```

Reason:

- `runs/` contains videos, logs, and checkpoints that can be large.
- model checkpoints should be copied/uploaded explicitly when needed, not committed.
- `squint-master-0/` is a local unmodified reference copy, not part of this repo.

Before every commit, check:

```bash
scripts/git_project.sh status --short --ignored
scripts/git_project.sh diff --cached --name-only
```

Make sure no training artifact, token, or accidental archive is staged.

## Standard Codex Start Checklist

When a new Codex conversation starts, first read:

```text
HANDOFF_PLACE_TASK.md
docs/place_experiment_versions.md
docs/codex_git_and_handoff_workflow.md
```

Then inspect:

```bash
scripts/git_project.sh status --short --branch
```

If the user asks to change training/task/deploy behavior:

1. Read the relevant file before editing.
2. Make scoped edits.
3. Update `docs/place_experiment_versions.md` with a new version entry.
4. If the change affects current handoff or commands, update `HANDOFF_PLACE_TASK.md`.
5. Run at least syntax checks for edited Python files.
6. Commit the changed code and docs.
7. Push when credentials are available.

## Version Record Rules

Append every deliberate experiment/code change to:

```text
docs/place_experiment_versions.md
```

Record:

```text
Version number:
Date:
Status: proposed / applied / trained / abandoned

Changed from previous version:
-

Files changed:
-

Training command:
    bash command here

Run directory:

Result summary:
- Final eval:
- Best eval:
- Peak GPU memory:
- Runtime:

Observed failure modes:
- unreachable:
- cannot grasp:
- grasped but cannot place:
- camera unclear:
- bad gripper pose:
- cube pushed away:
- early close:
- no lift after grasp:
- no bin approach after grasp:

Decision:
- continue / retrain / resume best / change action scale / change reward / change camera / abandon
```

## What To Preserve For Every Training Run

For each important run, preserve:

```text
runs/<EXP_NAME>/command.txt
runs/<EXP_NAME>/train.log
runs/<EXP_NAME>/eval_summary.txt
runs/<EXP_NAME>/gpu_peak.txt
runs/<EXP_NAME>/failure_notes.md
runs/<EXP_NAME>/videos/
runs/<EXP_NAME>/ckpt.pt
runs/<EXP_NAME>/best_ckpt.pt
```

Especially keep eval videos where the policy fails. These are needed to classify whether the issue is reach, grasp, lift, carry, release, camera, or gripper pose.

## Cloud Upload Checklist

When uploading only changed files to the cloud server, usually upload:

```text
train_squint.py
envs/place.py
envs/base_random_env.py
envs/robot/xlerobot.py
scripts/train_place_6gb_with_logs.sh
HANDOFF_PLACE_TASK.md
docs/place_experiment_versions.md
README.md
```

Also upload deploy files if real-robot deployment changed:

```text
deploy.py
scripts/deploy_place_v11_routeA.sh
scripts/deploy_place_b1_diag_policy_input.sh
deploy_utils/
```

After upload, run this on the cloud server:

```bash
cd /home/gpu/squint

python - <<'PY'
from pathlib import Path
for p in [Path("envs/place.py"), Path("envs/base_random_env.py"), Path("envs/robot/xlerobot.py"), Path("train_squint.py")]:
    compile(p.read_text(), str(p), "exec")
    print("OK", p)
PY
```

Use this compile snippet instead of `python -m py_compile` if `__pycache__` is read-only.

## Current Recommended Reproduction Commands

4090 fresh pure visual slow-real run:

```bash
cd /home/gpu/squint

env \
  EXP_NAME=place_xlerobot_v33_visible_range_mindist012_binexclude_softpregrasp_64img_1024env_16eval_256upd_buf300k_5500k_4090 \
  NO_PRIVILEGED_STATE=true \
  IMAGE_SIZE=64 \
  RENDER_SIZE=128 \
  NUM_ENVS=1024 \
  NUM_EVAL_ENVS=16 \
  NUM_UPDATES=256 \
  BATCH_SIZE=512 \
  BUFFER_SIZE=300000 \
  TOTAL_TIMESTEPS=5500000 \
  EVAL_FREQ=100000 \
  GPU_LOG_INTERVAL=10 \
  scripts/train_place_6gb_with_logs.sh
```

3060 local low-memory run:

```bash
cd /home/chichoo/squint-master6.6winproplace/squint-master

env \
  EXP_NAME=place_xlerobot_v33_visible_range_mindist012_binexclude_softpregrasp_64img_12env_8eval_8upd_buf40k_3500k_3060 \
  NO_PRIVILEGED_STATE=true \
  IMAGE_SIZE=64 \
  RENDER_SIZE=128 \
  NUM_ENVS=12 \
  NUM_EVAL_ENVS=8 \
  NUM_UPDATES=8 \
  BATCH_SIZE=48 \
  BUFFER_SIZE=40000 \
  TOTAL_TIMESTEPS=3500000 \
  EVAL_FREQ=100000 \
  GPU_LOG_INTERVAL=10 \
  scripts/train_place_6gb_with_logs.sh
```

## Current Known State

Current important state is summarized in:

```text
HANDOFF_PLACE_TASK.md
```

Short version:

- task: `SO101PlaceCube-v1`
- robot: `xlerobot_right_head`
- current policy route: pure visual+qpos with `NO_PRIVILEGED_STATE=true`
- current image size: `64`
- current Place horizon: `100`
- current bin color: green
- current controller: `pd_joint_target_delta_pos`
- current action limit: arm `+-0.07`, gripper `+-0.10`
- current effective sampling range: `x=[0.27,0.40]`, `y=[0.05,0.15]`
- current item/bin minimum center distance: `0.12m`
- current item/bin bin-footprint exclusion margin: `0.01m`
- current reward route: v29a soft pre-grasp shaping with light push/early-close penalties
- current recommended experiment: train v33 fresh; do not resume older checkpoints because the action distribution, sampling range/spacing/exclusion, and reward changed
