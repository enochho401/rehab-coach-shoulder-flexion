import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import time
import requests
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

def speak(text):
    if not text:
        return
    safe_text = text.replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'").replace('\n', ' ')
    components.html(
        f"""
        <script>
        function doSpeak() {{
            const u = new SpeechSynthesisUtterance("{safe_text}");
            u.rate = 0.95; u.pitch = 1; u.volume = 1; u.lang = "en-US";
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(u);
        }}
        doSpeak();
        setTimeout(doSpeak, 200);
        </script>
        """,
        height=0,
    )

def listen_and_send():
    components.html(
        """
        <script>
        const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SR) { alert("Please use Chrome for voice input."); }
        else {
            const rec = new SR();
            rec.lang = "en-US";
            rec.onresult = (e) => {
                const said = e.results[0][0].transcript;
                alert("You said: " + said + "\\n\\nPlease type or paste it into the chat box.");
            };
            rec.onerror = (e) => alert("Mic error: " + e.error);
            rec.start();
        }
        </script>
        """,
        height=0,
    )

def get_pose_keypoints(image):
    h, w = image.shape[:2]
    # Demo keypoints (replace with real OpenCV DNN model later if needed)
    return {
        "left_shoulder": (int(w * 0.40), int(h * 0.35)),
        "left_elbow": (int(w * 0.58), int(h * 0.42)),
        "left_wrist": (int(w * 0.72), int(h * 0.28)),
        "left_hip": (int(w * 0.40), int(h * 0.68)),
    }

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def analyse_shoulder_flexion(keypoints):
    try:
        shoulder = keypoints["left_shoulder"]
        elbow = keypoints["left_elbow"]
        wrist = keypoints["left_wrist"]
        hip = keypoints["left_hip"]
    except Exception:
        return {"status": "not_visible", "issues": [], "angles": {}}

    shoulder_angle = calculate_angle(hip, shoulder, elbow)
    elbow_angle = calculate_angle(shoulder, elbow, wrist)

    issues = []
    if shoulder_angle < 70:
        issues.append("arm_too_low")
    if elbow_angle < 130:
        issues.append("elbow_bent")
    if abs(shoulder[0] - hip[0]) > 40:
        issues.append("body_leaning")

    status = "correct" if not issues else "needs_improvement"
    return {
        "status": status,
        "issues": issues,
        "angles": {"shoulder": round(shoulder_angle, 1), "elbow": round(elbow_angle, 1)},
    }

def issues_to_text(issues):
    mapping = {
        "arm_too_low": "arm is not raised high enough",
        "elbow_bent": "elbow is bent too much",
        "body_leaning": "body is leaning",
    }
    return ", ".join(mapping.get(i, i) for i in issues) if issues else "form looks good"

def call_openrouter(api_key, messages, posture_context=""):
    system_prompt = (
        "You are a friendly rehab coach for seated shoulder flexion. "
        "Keep every reply short (1-2 sentences). Be encouraging and clear. "
        "Speak like a real therapist talking to a patient. "
        f"Current posture: {posture_context if posture_context else 'no recent capture'}."
    )
    full_messages = [{"role": "system", "content": system_prompt}] + messages[-6:]

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "baidu/ernie-5.5-flash-preview:free",
                "messages": full_messages,
                "max_tokens": 120,
                "temperature": 0.7,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return "Sorry, I could not reach the AI right now."

class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        self.latest_frame = None
        self.latest_analysis = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.latest_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        keypoints = get_pose_keypoints(img)
        self.latest_analysis = analyse_shoulder_flexion(keypoints)

        for name, (x, y) in keypoints.items():
            cv2.circle(img, (x, y), 6, (0, 255, 0), -1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.set_page_config(page_title="Rehab Coach", page_icon="💪", layout="wide")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenRouter API Key", type="password")
    st.caption("Get a free key at openrouter.ai")
    st.divider()
    st.markdown("**Pose:** OpenCV (cloud compatible)")
    st.markdown("**Chat:** ERNIE-5.5-Flash (free)")

st.title("💪 AI Rehab Coach – Seated Shoulder Flexion")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "captured_analysis" not in st.session_state:
    st.session_state.captured_analysis = None
if "captured_image" not in st.session_state:
    st.session_state.captured_image = None

col1, col2 = st.columns([1.3, 1])

with col1:
    st.subheader("Live Camera")
    ctx = webrtc_streamer(
        key="pose",
        video_processor_factory=PoseProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False},
    )

    if st.button("⏱️ Start 5-second Countdown & Auto Capture"):
        if ctx.video_processor:
            placeholder = st.empty()
            for i in range(5, 0, -1):
                placeholder.markdown(f"### Get ready... **{i}**")
                time.sleep(1)
            placeholder.markdown("### Capturing now!")
            st.session_state.captured_analysis = ctx.video_processor.latest_analysis
            st.session_state.captured_image = ctx.video_processor.latest_frame
            st.rerun()
        else:
            st.warning("Please start the camera first (click START).")

    if st.session_state.captured_image is not None:
        st.image(st.session_state.captured_image, channels="RGB", caption="Captured frame")
        a = st.session_state.captured_analysis
        if a:
            st.write(f"**Status:** {a['status'].replace('_', ' ').title()}")
            if a.get("angles"):
                st.write(f"Shoulder: {a['angles'].get('shoulder', '–')}° | Elbow: {a['angles'].get('elbow', '–')}°")
            if a["issues"]:
                st.warning("Issues: " + issues_to_text(a["issues"]))
            elif a["status"] == "correct":
                st.success("Form looks good!")

with col2:
    st.subheader("Coach Chat")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if st.button("🔊 Speak Last Reply") and st.session_state.messages:
        last = st.session_state.messages[-1]
        if last["role"] == "assistant":
            speak(last["content"])

    if st.button("🔊 Get Feedback on Captured Posture") and st.session_state.captured_analysis:
        a = st.session_state.captured_analysis
        posture_context = issues_to_text(a["issues"]) if a["issues"] else "form looks good"
        if api_key:
            feedback = call_openrouter(
                api_key,
                [{"role": "user", "content": "Please give me feedback on my current shoulder flexion posture."}],
                posture_context,
            )
        else:
            feedback = "Great job!" if a["status"] == "correct" else f"I noticed that your {posture_context}."
        st.session_state.messages.append({"role": "assistant", "content": feedback})
        speak(feedback)
        st.rerun()

    if st.button("🎤 Speak to Coach (use Chrome)"):
        listen_and_send()

    if prompt := st.chat_input("Chat with the coach..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        if api_key:
            posture_context = ""
            if st.session_state.captured_analysis:
                posture_context = issues_to_text(st.session_state.captured_analysis.get("issues", []))
            reply = call_openrouter(api_key, st.session_state.messages, posture_context)
        else:
            reply = "Please enter your OpenRouter API key in the sidebar."
        st.session_state.messages.append({"role": "assistant", "content": reply})
        speak(reply)
        st.rerun()

st.caption("Cloud-compatible version. Paste OpenRouter key in the sidebar.")
