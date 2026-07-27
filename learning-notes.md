# Learning notes

## Phase 0 - Environment and repository

### What I set up

- A dedicated Python virtual environment (`.venv`) so this project's packages do not affect other projects.
- A Git repository so changes can be saved as understandable milestones.
- A repository layout that keeps source code, tests, raw data, trained models, and generated output separate.

### Key ideas in my own words

< Ok, so now i am starting this project of object detection , it is divided into 8 phases and the 0th step is all about intialization .>

### Evidence

<!-- Add the output of `python --version` and `git status --short` here. -->
< Python 3.14.6 >
< So i got understanding of virtual environment. it is place where we can install specific libraries related to our project which does not hinder the other files.>
< and git ignore tells the git to ignopre specific installations or files which are as such not necessary keeping the repo short and sweet.>
### Questions I still have

<!-- Add questions here. Questions are a useful part of learning. -->

## Phase 1 - Video fundamentals

### Concepts to explain after the exercise

- A video is a sequence of image frames plus timing information (FPS).
- An image is a NumPy array: height x width x color channels.
- OpenCV uses BGR color order by default, whereas many other tools use RGB.
- `VideoCapture.read()` returns one frame at a time; `VideoWriter.write()` saves one frame at a time.

### Evidence to add

<!-- Paste your command output and note your input video's resolution, FPS, and duration. -->

### Questions I still have

<!-- Add questions here. -->
