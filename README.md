IAM26 — IEEE Robot Arm AI Challenge 2026
Master Plan + Code Snippets + GitHub Setup
Last updated: 2026-10-14

COMPETITION: Nov 28-29, Malta

==
URGENT DEADLINES
==

  ~Oct 28  AI Pitch Proposal (1 month before competition)  *** DO THIS NOW ***
           - Submit via Google Form (shared by organizers)
           - Describe: problem addressed, tech idea, how robot+AI advances humanity
           - Include: innovative idea, review of current tech, strategy from idea to realization

  Nov 1    'Making Of' Video  *** COMPULSORY — POINTS DEDUCTED IF MISSING ***
           - 3 to 8 minutes long
           - Must show: team member backgrounds, robot specs, building process
           - Must show health & safety approach when using tools
           - Need licensed music — submit music + license too
           - Record clips at EVERY lab visit from now on!

  Before   Software Stack / AI Report
  comp     - Report on software stack, AI models, methods, insights
           - No specific deadline given but must be before competition

  Nov 27   Robot Hand-in + Inventory Sheet (Friday before competition)
  (Fri)    - Register robot at venue by 16:00
           - Submit inventory sheet at least 1 day prior

  Nov 28   Competition Day 1
           - 5-minute presentation + demo of AI pitch proposal
           - Game 1 + Game 2 qualification rounds

  Nov 29   Competition Day 2 (if applicable — finals)


==
SYSTEM ARCHITECTURE — HOW EVERYTHING FITS TOGETHER
==

GAME 1 — PICK AND PLACE (Color Sort, Shape Sort, Dynamic Sort)
--

  ZED 2 (via ZED SDK)
    |
    +--> RGB image --> HSV color detection --> "that's a RED object"
    |                                          (your slider values)
    |
    +--> RGB image --> YOLO segmentation ----> "that's a CUBE"
    |                                          (when objects printed)
    |
    +--> Depth/Point Cloud --> 3D position --> "object is at [0.12, 0.08, 0.01]m"
    |
    +--> HSV on bins --> "red bin is at [0.15, 0.20, 0.02]m"
    |
    v
  QR calibration transform (camera coords --> robot coords)
    |
    v
  IK (from URDF) --> joint angles to move arm near object  <-- CLASSICAL
    |
    v
  ACT policy (wrist camera) --> grip, pick up, place in bin <-- LEARNED
    |
    v
  Done. Repeat for next object.


GAME 2 — TWO-ARM COORDINATION (Pouring, Folding, Wrap & Pack)
--

  Pure imitation learning. Both cameras recorded through LeRobot.

  ZED 2 (via OpenCV @ 1280x720) --> ACT sees full workspace
  Wrist camera (via OpenCV)      --> ACT sees close-up
  Both arms controlled by trained ACT policy.

  Separate trained model per task:
    - mithi/pouring-v1
    - mithi/folding-v1
    - mithi/wrap-pack-v1


WHAT NEEDS AI vs WHAT DOESN'T:

  Does NOT need training:
    - Color detection        --> HSV values (already done with slider)
    - Object 3D position     --> ZED 2 depth gives XYZ per pixel
    - Bin position           --> HSV detects colored bins too
    - Which bin to go to     --> if/else in code (red object -> red bin)
    - Shape detection        --> YOLO or contour analysis

  DOES need imitation learning:
    - Grasping the object    --> varies per situation, needs wrist cam feedback
    - Placing in bin         --> learned drop motion
    - Pouring rice           --> dexterous, two-arm
    - Folding cloth          --> dexterous, two-arm
    - Wrapping + packing     --> dexterous, two-arm


==
CAMERA SETUP — WHO DOES WHAT
==

  GAME 1 (pick-and-place):
    ZED 2 --> ZED SDK (pyzed) --> classical perception (HSV, depth, positions)
    Wrist --> OpenCV via LeRobot --> ACT training data for grasping

  GAME 2 (pouring, folding, wrapping):
    ZED 2 --> OpenCV @ 1280x720 via LeRobot --> ACT sees full workspace
    Wrist --> OpenCV via LeRobot --> ACT sees close-up
    Both cameras in LeRobot for training.

  LeRobot does NOT support ZED SDK natively.
  Use OpenCV for LeRobot recording. Use ZED SDK for runtime perception.


==
GITHUB REPO SETUP — STEP BY STEP FOR BEGINNERS
==

STEP 4: Copy your existing files in
--
  # Copy the URDF your teammates got working
  cp /home/cris/.cache/huggingface/lerobot/robot_urdfs/robot.urdf urdf/

  # Copy scripts (see CODE SNIPPETS section below for contents)
  # You'll create these files next

STEP 5: First commit and push
--
  git add -A
  git commit -m "Initial repo structure with scripts and URDF"
  git push

STEP 6: Your teammates clone on their machines
--
  git clone https://github.com/YOUR_USERNAME/iam26-robot.git

STEP 7: Daily workflow
--
  # Before starting work — pull latest changes
  git pull

  # After making changes
  git add -A
  git commit -m "describe what you changed"
  git push

  # If there's a conflict
  git pull --rebase
  # fix conflicts, then:
  git add -A
  git rebase --continue
  git push


==
LEROBOT COMMANDS — RECORDING + TRAINING + DEPLOYMENT
==

GAME 1 — Record grasping episodes (wrist camera only):

  lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=follower_left \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30} }" \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=leader_left \
    --dataset.repo_id=mithi/grasp-place-v2 \
    --dataset.single_task="Grasp object and place in target location" \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=20 \
    --dataset.reset_time_s=5 \
    --dataset.push_to_hub=false \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    --dataset.fps=30 \
    --display_data=false

GAME 2 — Record with both cameras (pouring/folding/wrapping):

  lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=follower_left \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 0, width: 1280, height: 720, fps: 30} }" \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=leader_left \
    --dataset.repo_id=mithi/pouring-v1 \
    --dataset.single_task="Pour rice from container into bowl" \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=10 \
    --dataset.push_to_hub=false \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    --dataset.fps=30 \
    --display_data=false

TRAIN:

  cd ~/LeRobot
  lerobot-train \
    --dataset.repo_id=mithi/grasp-place-v2 \
    --dataset.root=/home/cris/.cache/huggingface/lerobot/mithi/YOUR_FOLDER \
    --policy.type=act \
    --output_dir=outputs/train/grasp-place-v2-act \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --wandb.enable=false \
    --batch_size=2 \
    --dataset.image_transforms.enable=true

DEPLOY (test trained policy on real arm):

  lerobot-rollout \
    --strategy.type=base \
    --policy.path=outputs/train/grasp-place-v2-act/checkpoints/last/pretrained_model \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=follower_left \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30} }" \
    --task="Grasp object and place in target location" \
    --duration=30


RECORDING PROTOCOL (CRITICAL — read before every recording session):
  - 50+ episodes, 10 per object position variation
  - Move object to DIFFERENT position every episode
  - Move bin to at least 3 different positions across all episodes
  - Keep cameras FIXED between episodes
  - Grip naturally — vary approach angle, sometimes side, sometimes top
  - Object MUST be visible on camera at all times
  - If you fumble: press Left Arrow to re-record that episode
  - Controls: Right Arrow = save + next, Left Arrow = re-record, Esc = stop

KEYBOARD WARNING:
  - NEVER fully close the leader gripper — causes servo overload crash
  - Close just enough to grip the object firmly


==
RECORDING TIPS FROM HF DOCS
==

  - Start with one simple task (pick object, place in bin)
  - Get that working reliably FIRST
  - Then add variations (more positions, different objects)
  - Then move to next task
  - Don't add too much variation too quickly
  - "You should be able to do the task yourself by only looking at the camera images"
  - Use --dataset.image_transforms.enable=true during training for lighting robustness
  - Use --resume=true to add more episodes to existing dataset if recording crashes


==
KNOWN HARDWARE ISSUES + FIXES
==

  1. GRIPPER OVERLOAD (servo ID 6)
     Symptom: arm loses all connection, "Overload error", core dump
     Cause: leader gripper closed fully, follower stalls against object
     Fix: NEVER fully close leader gripper. Close just enough to grip.
     Better fix: add rubber/silicone pads to gripper fingers (rules allow this)

  2. WRIST CAMERA DISCONNECT
     Symptom: "OpenCVCamera(2) read failed", recording crashes
     Cause: USB cable comes loose as arm moves
     Fix: secure cable with clips/tape along the arm, leave strain relief loop

  3. ZED 2 SIDE-BY-SIDE IMAGE
     Symptom: 2560x720 frame instead of normal 1280x720
     Cause: OpenCV opens ZED in raw stereo mode
     Fix: request width=1280 in LeRobot config, OR crop frame[:, :1280, :]
     Better: use ZED SDK (pyzed) for proper single-eye output at runtime

  4. GPU OUT OF MEMORY DURING TRAINING
     Symptom: CUDA OOM error
     Fix: --batch_size=2 (default is 8)

  5. SLOW RECORD LOOP ON ACER LAPTOP
     Symptom: 7-25 Hz instead of 30 Hz target
     Fix: use desktop (29.93 Hz), or reduce to --dataset.fps=15 on laptop


==
TIMELINE — WHAT TO DO WHEN
==

  THIS WEEK (Oct 14-20):
    You (remote):
      [x] Set up GitHub repo with structure above
      [x] Fix URDF loading in test_ik.py
      [x] Start AI Pitch Proposal (DUE ~Oct 28)
      [x] Start 'Making Of' video clips (DUE Nov 1)
    Teammates (lab):
      [ ] Test IK on physical arm with URDF values
      [ ] Take YOLO training photos with current green objects
      [ ] Record more ACT episodes with varied positions + wrist cam only
      [ ] Film 'making of' clips while working

  WEEK 2-3 (Oct 21 - Nov 3):
    [ ] Submit AI Pitch Proposal (~Oct 28)
    [ ] Submit 'Making Of' video (Nov 1)
    [ ] Print competition objects when filament arrives
    [ ] Train YOLO on real colored objects
    [ ] Record 50 episodes with colored objects -> train color sort grasping
    [ ] Record 50 episodes of shape sorting
    [ ] Write unified pipeline code (YOLO + IK + ACT)

  WEEK 4-5 (Nov 4-17):
    [ ] End-to-end testing: full pipeline on physical arm
    [ ] Record + train Game 2 tasks (pouring, folding, wrapping)
    [ ] Write software/AI report
    [ ] Set up laptop (WSL + full stack) for competition day
    [ ] Test inference speed on laptop

  WEEK 6 (Nov 18-27):
    [ ] Robustness testing (run each task 10+ times)
    [ ] HSV re-tuning for different lighting
    [ ] Final debugging
    [ ] Prepare inventory sheet
    [ ] Nov 27 (Fri): Robot hand-in at venue by 16:00

  COMPETITION (Nov 28-29):
    [ ] 5-minute presentation
    [ ] 1 hour calibration window before Game 1
    [ ] Games 1 & 2
    [ ] Finals (if top 2)


==
DESKTOP INFO
==

  User: cris
  OS: Ubuntu 22 + NVIDIA GPU
  LeRobot: ~/LeRobot/
  Datasets: ~/.cache/huggingface/lerobot/mithi/
  Trained models: ~/LeRobot/outputs/train/
  Conda env: lerobot
  Arm ports: /dev/ttyACM0 (leader), /dev/ttyACM1 (follower)
  Camera indices: 0 = ZED 2 (2560x720 or 1280x720), 2 = wrist webcam
  Remote access: AnyDesk

  Before using servos: sudo chmod 666 /dev/ttyACM*
  Calibrate arms (once per device):
    lerobot-calibrate --robot.type=so100_follower --robot.port=/dev/ttyACM1 --robot.id=follower_left
    lerobot-calibrate --teleop.type=so100_leader --teleop.port=/dev/ttyACM0 --teleop.id=leader_left
