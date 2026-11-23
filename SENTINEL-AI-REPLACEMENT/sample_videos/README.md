# Sample Videos

This folder is intended for short demo videos used when running the system.

## ⚠️ Privacy Notice

**Do NOT place private CCTV footage or unauthorised recordings here.**

Only use:
- Public domain footage
- Synthetic / generated footage
- Footage you have explicit permission to use

---

## Recommended Free Test Videos

The following public datasets and sources provide suitable test footage:

### 1. VIRAT Video Dataset (Public Outdoor)
- https://viratdata.org/
- Outdoor campus scenes with people walking, gathering, and vehicles.

### 2. CAVIAR Dataset (Indoor)
- https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/
- Classic surveillance benchmark: shopping centre, meeting room.
- Includes labelled falls, fights, and abandoned objects.
- Recommended for testing fall and abandoned-object detectors.

### 3. Pexels / Pixabay (Royalty-Free Stock)
- https://www.pexels.com/search/videos/crowd/
- Search for "crowd", "walking people", "train station" etc.
- Download MP4 and rename to `demo.mp4`.

### 4. ShanghaiTech Campus Dataset
- https://svip-lab.github.io/dataset/campus_dataset.html
- Used in anomaly detection research.

---

## Quickstart

1. Download any of the above videos.
2. Rename to `demo.mp4` and place it here.
3. Run:

```bash
python main.py --input sample_videos/demo.mp4 --output outputs/result.mp4
```

Or launch the Streamlit dashboard and upload via the UI:

```bash
streamlit run app.py
```
