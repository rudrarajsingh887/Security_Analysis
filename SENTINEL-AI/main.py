"""
main.py
-------
Command-line interface for the AI Surveillance & Anomaly Detection System.

Usage examples
--------------
  # Process a video file:
  python main.py --input sample_videos/demo.mp4 --output outputs/result.mp4

  # Use webcam (index 0):
  python main.py --input 0 --show

  # Override config thresholds:
  python main.py --input demo.mp4 --conf 0.5 --crowd 5

  # Quiet mode (no preview window):
  python main.py --input demo.mp4 --output outputs/result.mp4 --no-show
"""

import argparse
import sys
from pathlib import Path

import yaml
from loguru import logger


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML config, raising a clear error if missing."""
    p = Path(config_path)
    if not p.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    with open(p) as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Surveillance & Anomaly Detection — CLI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to input video file or webcam index (e.g. 0).",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Path for annotated output video (e.g. outputs/result.mp4).",
    )
    parser.add_argument(
        "--config", "-c", default="config.yaml",
        help="Path to config.yaml (default: config.yaml).",
    )
    parser.add_argument(
        "--show", action="store_true", default=True,
        help="Display live preview window (default: True).",
    )
    parser.add_argument(
        "--no-show", dest="show", action="store_false",
        help="Disable live preview window.",
    )
    parser.add_argument(
        "--conf", type=float, default=None,
        help="Override detection confidence threshold.",
    )
    parser.add_argument(
        "--crowd", type=int, default=None,
        help="Override crowd-density person count threshold.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)
    logger.add("outputs/run.log", level="DEBUG", rotation="5 MB")

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.conf is not None:
        config["detection"]["confidence_threshold"] = args.conf
        logger.info(f"Confidence threshold overridden to {args.conf}")
    if args.crowd is not None:
        config["crowd_analysis"]["crowd_threshold"] = args.crowd
        logger.info(f"Crowd threshold overridden to {args.crowd}")

    # Lazy import to avoid slow startup when running --help
    from src.video_processor import VideoProcessor

    processor = VideoProcessor(config)

    logger.info("=" * 60)
    logger.info("  AI Surveillance & Anomaly Detection System")
    logger.info("=" * 60)
    logger.info(f"  Input  : {args.input}")
    logger.info(f"  Output : {args.output or '(not saving)'}")
    logger.info(f"  Model  : {config['detection']['model_name']}")
    logger.info("=" * 60)

    try:
        metrics = processor.process_video(
            input_source=args.input,
            output_path=args.output,
            show_window=args.show,
        )
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)

    # ---- Print summary ----
    logger.info("")
    logger.info("=" * 60)
    logger.info("  PROCESSING SUMMARY")
    logger.info("=" * 60)
    for key, value in metrics.items():
        logger.info(f"  {key:<35}: {value}")
    logger.info("=" * 60)

    if args.output:
        logger.success(f"Annotated video saved → {args.output}")


if __name__ == "__main__":
    main()
