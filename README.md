# Stabilizing Pose Trajectories: CS663 Project 1 Research Tutorial

**Temporal Filtering and Keypoint Trajectory Stabilization for Human Pose Estimation in Exercise Videos**
Marianna Belmares, CS663 Computer Vision, Fall 2026

A static HTML/CSS/JS tutorial site for GitHub Pages. It needs no build step.

## Publish on GitHub Pages

1. Create a public repo (for example `pose-stabilization-tutorial`) and put the contents of this folder at the repo root. `index.html` must be at the top level.
2. In the repo, go to **Settings → Pages → Build and deployment**, choose **Deploy from a branch**, then pick `main` and `/ (root)`.
3. After a minute the site is live at `https://<username>.github.io/<repo>/`. Open every page and click through the Next buttons, the Lab and the Quiz to test it.

`.nojekyll` is included so GitHub serves every file as-is.

## Structure

| Path | What it is |
|---|---|
| `index.html` | Intro page and contents |
| `01-…10-*.html` | The ten tutorial sections, following the proposal's table of contents |
| `lab.html` | Interactive Filter Lab with sliders, live metrics and an animated stick figure (simulated data) |
| `webcam.html` | Live Webcam Lab: MediaPipe in the browser on your camera or a video file, raw vs. filtered, rep counter, CSV export |
| `quiz.html` | 10-question self-check quiz |
| `bibliography.html` | Annotated bibliography with 12 references, numbered in order of first appearance |
| `css/style.css`, `js/site.js` | Shared styles, navigation, dark mode, and narration player |
| `js/filters.js` | JavaScript MA / EMA / One Euro / Kalman filters |
| `experiment/` | Python: `filters.py`, `run_experiment.py` (simulation + figures), `extract_landmarks.py` (MediaPipe on your own video) |
| `img/` | Figures created by `run_experiment.py` |
| `data/` | Simulated squat data used by the Lab |
| `audio/` | **Empty: put your recorded narration here** |
| `narration/scripts.md` | Narration script for every page |

## To-do before submitting

- [ ] **Voice on every page (required).** Record each script in `narration/scripts.md` and save it as `audio/<page>.mp3`, for example `audio/01-intro.mp3`. A page with no MP3 falls back to a "Listen to this page" button that reads the script with the browser's voice. Your own recording is what the rubric asks for.
- [ ] **Originality.** The text is a draft. Read every page and rewrite it in your own words and voice. The rubric gives 10 points for originality, and you need to be able to present and defend all of it.
- [ ] **Check facts I couldn't fully confirm:**
  - the affiliations in the Section 10.6 table and in the bibliography reliability notes (Zhejiang Univ. / State Key Lab of CAD&CG; NECTEC Thailand for Pongnumkul; Lille/Inria and Waterloo for the 1€ authors)
  - the publication status of the two arXiv preprints [9] and [10]
- [ ] **Section 10.7, last paragraph:** this is written as your Project 3 plan. Change it to match what you actually intend.
- [ ] **Optional but strong: real-video result.** On your Mac:
  ```
  cd experiment
  pip install mediapipe opencv-python numpy scipy matplotlib
  curl -L -o pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
  python extract_landmarks.py my_squats.mp4 --side left --image-mode
  python run_experiment.py --csv landmarks.csv     # writes img/exp_real_video.png
  ```
  Then add `img/exp_real_video.png` as a figure in Section 9. You can also record on `webcam.html` and use **Download raw landmarks (CSV)**.
- [ ] **Test the Webcam Lab on GitHub Pages.** It loads MediaPipe from jsDelivr and the model from Google Storage. Camera access needs https, which GitHub Pages provides. Its code was checked up to the model download, but it couldn't be run end-to-end with a camera here.
- [ ] **Presentation:** a short PowerPoint plus an unlisted YouTube video walking through the deck and the site. Post both URLs to Canvas.

## About the experiment

All numbers in Section 9 and the Lab come from a **simulated** squat with known ground truth (seed 7). Because the truth is known, error can be measured. The page says this clearly. To reproduce:
`python experiment/run_experiment.py --simulate`

## Sources

See `bibliography.html`. Main references: Tharatipyakul et al. 2024 (Heliyon); Bazarevsky et al. 2020 (BlazePose); MediaPipe docs; Ullah et al. 2025 (Sci. Reports); Savitzky & Golay 1964; Casiez et al. 2012 (1€ filter); Kalman 1960; Welch & Bishop; Hu & Ye 2025; Eghbalian & Desai 2026; Zeng et al. 2022 (SmoothNet); Wei et al. 2026 (HTD-Refine, CVPR).
