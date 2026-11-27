# Assets

| File | Description |
|------|-------------|
| `architecture_diagram.png` | Pipeline architecture diagram (raster, used in main README) |
| `architecture_diagram.svg` | Same diagram as scalable vector — edit this if the pipeline changes |
| `demo_screenshot.png` | Screenshot of the Streamlit dashboard in action — **add your own** |

## Adding a demo screenshot

1. Run `streamlit run app.py` and process a sample video.
2. Take a screenshot of the dashboard showing the live preview and metrics.
3. Save it here as `demo_screenshot.png`.
4. Reference it in the main `README.md`, e.g.:
   ```markdown
   ![Dashboard screenshot](assets/demo_screenshot.png)
   ```

## Editing the architecture diagram

The SVG is plain hand-built markup (no external tooling required) — open it
in any text editor or vector editor (Inkscape, Figma, VS Code) and adjust
colours, labels, or boxes directly. Re-export to PNG if you change it and
want the raster version to match.
