# Phase 7: Streamlit app and deployment

## Local use

```bash
source .venv/bin/activate
python -m pip install 'streamlit>=1.40,<2'
streamlit run app.py
```

Open the local URL shown in the terminal, upload a short traffic video, choose the model and settings, then download the annotated MP4.

For 4K/60 FPS input, choose the **Fast** profile. It halves output resolution, reduces YOLO input to 416 pixels, and processes every second frame. The output retains approximately the original duration but has a lower frame rate, which is an intentional speed/temporal-smoothness trade-off.

## Deployment checklist

1. Keep `requirements.txt`, `app.py`, `src/`, `.streamlit/config.toml`, and `assets/traffic_yolo11n.pt` in Git.
2. Keep raw videos, BDD100K data, temporary output, and `runs/` out of Git.
3. Push the repository to GitHub.
4. On Streamlit Community Cloud, select the repository, branch `main`, and entrypoint `app.py`, then click Deploy.
5. Test with a small MP4 first. Community Cloud has finite CPU/RAM; large/high-resolution videos may take a long time.

## Concepts to learn

- Stateless script reruns vs `st.session_state`
- `st.file_uploader`, `st.video`, and `st.download_button`
- Temporary files and cleanup
- Why a deployed application needs all model artifacts it loads
- Deployment constraints: memory, CPU, upload size, and cold-start time
