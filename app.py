"""
app.py
------
Streamlit dashboard for the AI Surveillance & Anomaly Detection System.

Run with:
    streamlit run app.py

Features
--------
  - Upload a video file or enter a path / webcam index.
  - Adjust detection thresholds via sidebar sliders.
  - Process and stream annotated frames live.
  - Display summary metrics and alert charts after processing.
"""

import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import yaml

# ── Page config (must be first Streamlit call) ─────────────────────────
st.set_page_config(
    page_title="SENTINEL AI | Surveillance Dashboard",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Load config ─────────────────────────────────────────────────────────
@st.cache_data
def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


try:
    base_config = load_config()
except FileNotFoundError:
    st.error("config.yaml not found.  Make sure you are running from the project root.")
    st.stop()


# ── Sidebar — configuration ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg",
             width=180)
    st.title("⚙️ Configuration")

    st.subheader("Detection")
    conf_threshold = st.slider(
        "Confidence threshold", 0.10, 0.95,
        float(base_config["detection"]["confidence_threshold"]), 0.05,
    )
    model_name = st.selectbox(
        "YOLO model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        index=0,
        help="Larger models are more accurate but slower.",
    )

    st.subheader("Crowd Analysis")
    crowd_threshold = st.slider(
        "Crowd threshold (people)", 2, 30,
        int(base_config["crowd_analysis"]["crowd_threshold"]), 1,
    )

    st.subheader("Anomaly Detection")
    inactivity_sec = st.slider(
        "Inactivity alert (seconds)", 5, 60,
        int(base_config["suspicious_behaviour"]["inactivity_seconds"]), 5,
    )
    stationary_sec = st.slider(
        "Abandoned object alert (seconds)", 5, 60,
        int(base_config["abandoned_object"]["stationary_seconds"]), 5,
    )
    fall_frames = st.slider(
        "Fall detection consecutive frames", 3, 20,
        int(base_config["fall_detection"]["consecutive_frames"]), 1,
    )

    st.divider()
    st.caption(
        "⚠️ This system is for **educational & portfolio purposes only**.\n\n"
        "It is rule-based and **not production-grade**.\n"
        "Do not use with private/unauthorised footage."
    )


# ── Build runtime config from sidebar values ─────────────────────────────
def build_config(base: dict) -> dict:
    import copy
    cfg = copy.deepcopy(base)
    cfg["detection"]["confidence_threshold"] = conf_threshold
    cfg["detection"]["model_name"] = model_name
    cfg["crowd_analysis"]["crowd_threshold"] = crowd_threshold
    cfg["suspicious_behaviour"]["inactivity_seconds"] = inactivity_sec
    cfg["abandoned_object"]["stationary_seconds"] = stationary_sec
    cfg["fall_detection"]["consecutive_frames"] = fall_frames
    return cfg


# ── Main layout ──────────────────────────────────────────────────────────
st.title("🎥 SENTINEL AI")
st.subheader("AI Surveillance & Anomaly Detection")
st.markdown(
    "Upload a video to detect people, track movement, and flag "
    "potential anomalies such as falls, abandoned objects, and crowd density."
)

tab_upload, tab_results, tab_about = st.tabs(
    ["📂 Upload & Analyse", "📊 Results", "ℹ️ About"]
)

# ── Upload tab ───────────────────────────────────────────────────────────
with tab_upload:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input")
        upload_mode = st.radio(
            "Source", ["Upload video file", "Enter file path / webcam index"],
            horizontal=True,
        )

        video_source = None

        if upload_mode == "Upload video file":
            uploaded = st.file_uploader(
                "Choose a video", type=["mp4", "avi", "mov", "mkv"]
            )
            if uploaded:
                suffix = Path(uploaded.name).suffix
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(uploaded.read())
                tmp.close()
                video_source = tmp.name
                st.video(uploaded)
        else:
            video_source = st.text_input(
                "File path or webcam index (0, 1, …)",
                placeholder="sample_videos/demo.mp4",
            )

        save_output = st.checkbox("Save annotated output video", value=True)
        if save_output:
            output_path = st.text_input(
                "Output path", value="outputs/streamlit_result.mp4"
            )
        else:
            output_path = None

    with col2:
        st.subheader("Run")
        run_btn = st.button("▶ Run Analysis", type="primary", use_container_width=True)

    # ── Processing ──────────────────────────────────────────────────────
    if run_btn:
        if not video_source:
            st.error("Please select or enter a video source first.")
        else:
            from src.video_processor import VideoProcessor

            cfg = build_config(base_config)
            processor = VideoProcessor(cfg)

            st.subheader("Live Preview")
            preview_placeholder = st.empty()
            metrics_placeholder = st.empty()
            progress_bar = st.progress(0)
            status_text  = st.empty()

            # Determine total frames for progress (best-effort)
            try:
                cap_probe = cv2.VideoCapture(video_source)
                total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                cap_probe.release()
            except Exception:
                total_frames = 1

            frame_count = 0
            last_metrics = {}
            start_time = time.time()

            try:
                for annotated_frame, current_metrics in processor.process_frame_stream(video_source):
                    frame_count += 1
                    last_metrics = current_metrics

                    # Show every 3rd frame to keep the UI snappy
                    if frame_count % 3 == 0:
                        rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        preview_placeholder.image(rgb, channels="RGB", use_container_width=True)

                    progress = min(frame_count / total_frames, 1.0)
                    progress_bar.progress(progress)
                    elapsed = time.time() - start_time
                    fps_live = frame_count / max(elapsed, 0.01)
                    status_text.text(
                        f"Frame {frame_count}/{total_frames}  |  "
                        f"Live FPS: {fps_live:.1f}  |  "
                        f"Alerts: {sum(v for k, v in current_metrics.items() if 'alert' in k)}"
                    )

                    # Live mini-metrics row
                    with metrics_placeholder.container():
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Peak People", current_metrics.get("peak_person_count", 0))
                        c2.metric("Fall Alerts", current_metrics.get("total_fall_alerts", 0))
                        c3.metric("Abandoned Objects", current_metrics.get("total_abandoned_alerts", 0))
                        c4.metric("Crowd Alerts", current_metrics.get("total_crowd_alerts", 0))

            except FileNotFoundError as exc:
                st.error(str(exc))
                st.stop()
            except Exception as exc:
                st.error(f"Processing error: {exc}")
                raise

            progress_bar.progress(1.0)
            status_text.text("✅ Processing complete!")
            st.session_state["last_metrics"] = last_metrics
            st.session_state["processed"] = True
            st.success("Analysis finished! View the **Results** tab for full metrics.")

            if save_output and output_path:
                # Re-run with file output (stream doesn't write file)
                with st.spinner(f"Saving annotated video to {output_path}…"):
                    p2 = VideoProcessor(cfg)
                    p2.process_video(
                        video_source, output_path=output_path, show_window=False
                    )
                st.success(f"Video saved → `{output_path}`")


# ── Results tab ─────────────────────────────────────────────────────────
with tab_results:
    if not st.session_state.get("processed"):
        st.info("Run an analysis first to see results here.")
    else:
        metrics = st.session_state.get("last_metrics", {})

        st.subheader("📊 Session Summary")
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Frames Processed", metrics.get("total_frames_processed", 0))
        col_b.metric("Peak People Count", metrics.get("peak_person_count", 0))
        col_c.metric("Fall Alerts", metrics.get("total_fall_alerts", 0))
        col_d.metric("Abandoned Object Alerts", metrics.get("total_abandoned_alerts", 0))

        col_e, col_f, col_g, col_h = st.columns(4)
        col_e.metric("Crowd Density Alerts", metrics.get("total_crowd_alerts", 0))
        col_f.metric("Inactivity Alerts", metrics.get("total_inactivity_alerts", 0))
        col_g.metric("Erratic Movement Alerts", metrics.get("total_erratic_alerts", 0))
        col_h.metric("Zone Breach Alerts", metrics.get("total_zone_alerts", 0))

        st.divider()

        # Bar chart of alert types
        st.subheader("Alert Breakdown")
        alert_data = {
            "Fall": metrics.get("total_fall_alerts", 0),
            "Abandoned Object": metrics.get("total_abandoned_alerts", 0),
            "Crowd Density": metrics.get("total_crowd_alerts", 0),
            "Inactivity": metrics.get("total_inactivity_alerts", 0),
            "Erratic Movement": metrics.get("total_erratic_alerts", 0),
            "Zone Breach": metrics.get("total_zone_alerts", 0),
        }
        df_alerts = pd.DataFrame.from_dict(
            {"Alert Type": list(alert_data.keys()),
             "Count": list(alert_data.values())}
        )
        st.bar_chart(df_alerts.set_index("Alert Type"))

        st.subheader("Raw Metrics")
        st.dataframe(
            pd.DataFrame(
                {"Metric": list(metrics.keys()), "Value": list(metrics.values())}
            ),
            use_container_width=True,
        )


# ── About tab ───────────────────────────────────────────────────────────
with tab_about:
    st.subheader("About This Project")
    st.markdown(
        """
**SENTINEL AI — AI Surveillance & Anomaly Detection System** is an open-source, educational
computer vision pipeline built with YOLOv8, ByteTrack, and OpenCV.

### What it does
| Module | Description |
|---|---|
| Object Detection | YOLOv8 detects people, bags, and suitcases |
| Object Tracking | ByteTrack assigns persistent IDs across frames |
| Fall Detection | Bounding-box aspect ratio rules flag possible falls |
| Abandoned Object | Bags left unattended for configurable seconds |
| Crowd Analysis | People count with density threshold alerts |
| Anomaly Detection | Inactivity, erratic movement, restricted zones |

### Tech Stack
`Python` · `YOLOv8 / Ultralytics` · `OpenCV` · `ByteTrack` · `Streamlit` · `NumPy` · `Pandas`

### ⚠️ Privacy & Ethics
- **Educational use only.**  Do not use this system with private CCTV footage
  or any recordings without explicit consent and legal authorisation.
- All detections are **rule-based heuristics** and will produce false positives.
- This is **not a production surveillance system**.
- See `docs/limitations.md` for full discussion.

### Licence
MIT — free to use, modify, and share with attribution.
        """
    )
