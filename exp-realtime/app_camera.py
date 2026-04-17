import streamlit as st
import cv2
import numpy as np
import time
import threading
import queue
import tempfile
import os

from classification.classification_model import carregar_recursos, classificar_imagem

st.set_page_config(page_title="Realtime", page_icon="🩺", layout="centered")

FRAME_QUEUE_MAXSIZE = 1
RESULT_QUEUE_MAXSIZE = 1

def probe_camera(index: int, timeout_sec: float = 1.0) -> bool:
    cap = None
    try:
        cap = cv2.VideoCapture(index, cv2.CAP_ANY)
        if not cap.isOpened():
            return False

        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            ret, _ = cap.read()
            if ret:
                return True
        return False
    except Exception:
        return False
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass

def list_available_cameras(max_search: int = 6) -> list[int]:
    available = []
    for i in range(max_search):
        if probe_camera(i, timeout_sec=0.6):
            available.append(i)
    return available

def camera_capture_loop(frame_q: queue.Queue, stop_event: threading.Event, device_index: int = 0, width:int=640, height:int=480):
    cap = cv2.VideoCapture(device_index, cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print(f"ERROR: Could not open camera {device_index}.")
        stop_event.set()
        return

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            try:
                if frame_q.full():
                    _ = frame_q.get_nowait()
                frame_q.put_nowait(frame)
            except Exception:
                pass

            time.sleep(0.01)
    finally:
        cap.release()

def processing_loop(frame_q: queue.Queue, result_q: queue.Queue, stop_event: threading.Event):
    while not stop_event.is_set():
        try:
            frame = frame_q.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
            cv2.imwrite(tmp_path, frame)

            result = classificar_imagem(tmp_path)

            try:
                os.remove(tmp_path)
            except Exception:
                pass

        except Exception as e:
            result = {"status": "erro", "message": str(e)}

        try:
            if result_q.full():
                _ = result_q.get_nowait()
            result_q.put_nowait((frame, result))
        except Exception:
            pass

        time.sleep(0.01)

def start_camera_threads(cam_state: dict, device_index: int):
    cam_state['frame_q'] = queue.Queue(maxsize=FRAME_QUEUE_MAXSIZE)
    cam_state['result_q'] = queue.Queue(maxsize=RESULT_QUEUE_MAXSIZE)
    cam_state['stop_event'] = threading.Event()

    cam_state['capture_thread'] = threading.Thread(
        target=camera_capture_loop,
        args=(cam_state['frame_q'], cam_state['stop_event'], device_index),
        daemon=True
    )
    cam_state['process_thread'] = threading.Thread(
        target=processing_loop,
        args=(cam_state['frame_q'], cam_state['result_q'], cam_state['stop_event']),
        daemon=True
    )
    cam_state['capture_thread'].start()
    cam_state['process_thread'].start()
    cam_state['device_index'] = device_index

def stop_camera_threads(cam_state: dict):
    try:
        if cam_state.get('stop_event'):
            cam_state['stop_event'].set()
        if cam_state.get('capture_thread'):
            cam_state['capture_thread'].join(timeout=1.0)
        if cam_state.get('process_thread'):
            cam_state['process_thread'].join(timeout=1.0)
    except Exception:
        pass
    finally:
        cam_state['capture_thread'] = None
        cam_state['process_thread'] = None
        cam_state['frame_q'] = None
        cam_state['result_q'] = None
        cam_state['stop_event'] = None
        cam_state['device_index'] = None

def main():
    st.title("🔬 Wound Classifier — Real-Time (OpenCV)")
    st.markdown("Run locally (`python -m streamlit run app_camera.py`). The camera must be accessible to the server (your computer).")

    if 'model_loaded' not in st.session_state:
        with st.spinner("Loading model... (may take a moment)"):
            st.session_state.model_loaded = carregar_recursos()
        if st.session_state.model_loaded:
            st.success("✅ Model loaded!")
        else:
            st.error("❌ Failed to load model — check path/files.")
            return

    st.sidebar.header("Cameras")
    max_probe = st.sidebar.number_input("Search indices 0..N-1 (N)", min_value=1, max_value=16, value=6, step=1)
    if 'available_cameras' not in st.session_state:
        st.session_state.available_cameras = []

    if st.sidebar.button("🔎 Detect cameras"):
        with st.spinner("Detecting cameras..."):
            st.session_state.available_cameras = list_available_cameras(max_search=int(max_probe))
        st.sidebar.success(f"Found: {len(st.session_state.available_cameras)}")

    cam_options = [f"{i} - Camera {i}" for i in st.session_state.available_cameras]
    cam_options.insert(0, "manual: enter index")

    selected_option = st.sidebar.selectbox("Select available camera", cam_options, index=0)
    if selected_option.startswith("manual"):
        selected_index = st.sidebar.number_input("Manual camera index", min_value=0, max_value=31, value=0, step=1)
    else:
        selected_index = int(selected_option.split(" - ")[0])

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        fps = st.slider("Target FPS (UI)", 1, 10, 2)
    with col2:
        start = st.checkbox("🔴 Start/Stop camera", value=False, key="start_camera")
    with col3:
        refresh_detect = st.button("🔁 Re-detect")

    if refresh_detect:
        with st.sidebar.spinner("Re-detecting..."):
            st.session_state.available_cameras = list_available_cameras(max_search=int(max_probe))
        st.sidebar.success(f"Found: {len(st.session_state.available_cameras)}")

    image_placeholder = st.empty()
    info_placeholder = st.empty()

    if 'cam_state' not in st.session_state:
        st.session_state.cam_state = {
            'frame_q': None,
            'result_q': None,
            'stop_event': None,
            'capture_thread': None,
            'process_thread': None,
            'device_index': None,
        }

    cam = st.session_state.cam_state

    if start:
        need_start = False
        if cam['capture_thread'] is None or not cam['capture_thread'].is_alive():
            need_start = True
        elif cam.get('device_index') is not None and cam['device_index'] != selected_index:
            stop_camera_threads(cam)
            need_start = True

        if need_start:
            try:
                start_camera_threads(cam, selected_index)
                st.success(f"Camera started (device {selected_index}).")
            except Exception as e:
                st.error(f"Failed to start camera {selected_index}: {e}")
                stop_camera_threads(cam)
    else:
        if cam['capture_thread'] is not None:
            stop_camera_threads(cam)
            st.info("Camera stopped.")

    try:
        while start:
            frame = None
            result = None

            try:
                frame, result = cam['result_q'].get_nowait()
            except Exception:
                try:
                    frame = cam['frame_q'].get_nowait()
                except Exception:
                    frame = None

            if frame is not None:
                display = frame.copy()
                if result and result.get('status') == 'sucesso':
                    h, w = display.shape[:2]
                    cv2.rectangle(display, (30, 30), (w-30, h-30), (0, 0, 255), 2)
                    text = f"{result['translated_class']} ({result['predicted_percentage_confidence']})"
                    cv2.putText(display, text, (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                    info_placeholder.success(f"Prediction: {result['translated_class']} | Confidence: {result['predicted_percentage_confidence']}")
                elif result and result.get('status') == 'erro':
                    cv2.putText(display, f"ERROR: {result.get('message')}", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    info_placeholder.error(f"Error: {result.get('message')}")
                else:
                    info_placeholder.info("Waiting for prediction...")

                display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                image_placeholder.image(display_rgb, width='content')
            else:
                image_placeholder.text("Waiting for camera frame...")
                info_placeholder.text("...")

            time.sleep(1.0 / max(1, fps))

            if not st.session_state.get("start_camera", False):
                break

    except Exception as e:
        st.error(f"UI Error: {e}")

    if cam.get('stop_event') and not cam['stop_event'].is_set():
        cam['stop_event'].set()

if __name__ == "__main__":
    main()