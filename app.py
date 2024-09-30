import streamlit as st
from openai import OpenAI
import dotenv
import os
from PIL import Image
from audio_recorder_streamlit import audio_recorder
import base64
from io import BytesIO

dotenv.load_dotenv()

openai_models = [
    "gpt-4o", 
    "gpt-4-turbo", 
    "gpt-3.5-turbo-16k", 
    "gpt-4", 
    "gpt-4-32k",
]

#Function to query and stream the respose from the LLM
def stream_llm_response(client, model_params):
    response_message = ""

    for chunk in client.chat.completions.create(
        model=model_params["model"] if "model" in model_params else "gpt-4o-2024-05-13",
        messages=st.session_state.messages,
        temperature=model_params["temperature"] if "temperature" in model_params else 0.3,
        max_tokens=4096,
        stream=True,
    ):
        response_message += chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
        yield chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""

    st.session_state.messages.append({
        "role": "assistant",
        "content": [
                {
                    "type": "text",
                    "text": response_message,
                }
            ]
    })

# Function to convert file to base64
def get_image_base64(image_raw):
    buffered = BytesIO()
    image_raw.save(buffered, format=image_raw.format)
    img_byte = buffered.getvalue()

    return base64.b64encode(img_byte).decode('utf-8')

def file_to_base64(file):
    with open(file, "rb") as f:

        return base64.b64encode(f.read())

def base64_to_image(base64_string):
    base64_string = base64_string.split(",")[1]
    
    return Image.open(BytesIO(base64.b64decode(base64_string)))

def main():

    # ---Page Config---
    st.set_page_config(
        page_title="The Datavoro Project1",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # ---Header ---
    st.html("""<h1 style="text-align: center; color: #6ca395;">🤖 <i>Datavoro Project1</i> </h1>""")

    with st.sidebar:
        default_openai_api_key = os.getenv("OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") is not None else "" #Only for development environment,otherwise it should return None
        with st.popover("🔐 OpenAI"):
            openai_api_key = st.text_input("Introduce your OpenAI API Key (https://platform.openai.com/)", value=default_openai_api_key, type="password")
        if not (openai_api_key == "" or openai_api_key is None or "sk-" not in openai_api_key):
            st.divider()

    # --- Main Content ---
    # Checking if the user has introduced the OpenAI API Key, if not, a warning is displayed
    if (openai_api_key == "" or openai_api_key is None or "sk-" not in openai_api_key):
        st.write("#")
        st.warning("⬅️ Please introduce an API Key to continue...")
    
    else:
        client = OpenAI(api_key=openai_api_key)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Displaying the previous messages if there are any
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                for content in message["content"]:
                    if content["type"] == "text":
                        st.write(content["text"])
                    elif content["type"] == "image_url":      
                        st.image(content["image_url"]["url"])
                    elif content["type"] == "video_file":
                        st.video(content["video_file"])
                    elif content["type"] == "audio_file":
                        st.audio(content["audio_file"])
    with st.sidebar:

        available_models = [] + openai_models if openai_api_key else []

        model = st.selectbox("Select a model:", available_models, index=0)
        
        with st.popover("⚙️ Model parameters"):
            model_temp = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.1)

        audio_response = st.toggle("Audio response", value=False)
        if audio_response:
            cols = st.columns(2)
            with cols[0]:
                tts_voice = st.selectbox("Select a voice:", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"])
            with cols[1]:
                tts_model = st.selectbox("Select a model:", ["tts-1", "tts-1-hd"], index=1)

        model_params = {
            "model": model,
            "temperature": model_temp,
        }

        def reset_conversation():
            if "messages" in st.session_state and len(st.session_state.messages) > 0:
                st.session_state.pop("messages", None)

        st.button(
            "🗑️ Reset conversation", 
            on_click=reset_conversation,
        )

        st.divider()

        if model in ["gpt-4o", "gpt-4-turbo", "gemini-1.5-flash", "gemini-1.5-pro", "claude-3-5-sonnet-20240620"]:
            
            st.write(f"### **🖼️ Add an image")

            def add_image_to_messages():
                if st.session_state.uploaded_img or ("camera_img" in st.session_state and st.session_state.camera_img):
                    img_type = st.session_state.uploaded_img.type if st.session_state.uploaded_img else "image/jpeg"
                    if img_type == "video/mp4":
                        # save the video file
                        video_id = random.randint(100000, 999999)
                        with open(f"video_{video_id}.mp4", "wb") as f:
                            f.write(st.session_state.uploaded_img.read())
                        st.session_state.messages.append(
                            {
                                "role": "user", 
                                "content": [{
                                    "type": "video_file",
                                    "video_file": f"video_{video_id}.mp4",
                                }]
                            }
                        )
                    else:
                        raw_img = Image.open(st.session_state.uploaded_img or st.session_state.camera_img)
                        img = get_image_base64(raw_img)
                        st.session_state.messages.append(
                            {
                                "role": "user", 
                                "content": [{
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{img_type};base64,{img}"}
                                }]
                            }
                        )
            cols_img = st.columns(2)

            with cols_img[0]:
                with st.popover("📁 Upload"):
                    st.file_uploader(
                        "Upload an image", 
                        type=["png", "jpg", "jpeg"], 
                        accept_multiple_files=False,
                        key="uploaded_img",
                        on_change=add_image_to_messages,
                    )

            with cols_img[1]:                    
                with st.popover("📸 Camera"):
                    activate_camera = st.checkbox("Activate camera")
                    if activate_camera:
                        st.camera_input(
                            "Take a picture", 
                            key="camera_img",
                            on_change=add_image_to_messages,
                        )
        st.write("#")
        st.write(f"### **🎤 Add an audio")

        audio_prompt = None
        if "prev_speech_hash" not in st.session_state:
            st.session_state.prev_speech_hash = None

        speech_input = audio_recorder("Press to talk:", icon_size="3x", neutral_color="#6ca395", )
        if speech_input and st.session_state.prev_speech_hash != hash(speech_input):
            st.session_state.prev_speech_hash = hash(speech_input)
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=("audio.wav", speech_input),
            )

            audio_prompt = transcript.text

    # Chat input
    if prompt := st.chat_input("Hi! Ask me anything...")or audio_prompt:
        st.session_state.messages.append(
            {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": prompt or audio_prompt,
                }]
            }
        )
            
        # Display the new messages
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            st.write_stream(
                stream_llm_response(client, model_params)
            )

        if audio_response:
            response =  client.audio.speech.create(
                model=tts_model,
                voice=tts_voice,
                input=st.session_state.messages[-1]["content"][0]["text"],
            )
            audio_base64 = base64.b64encode(response.content).decode('utf-8')
            audio_html = f"""
            <audio controls autoplay>
                <source src="data:audio/wav;base64,{audio_base64}" type="audio/mp3">
            </audio>
            """
            st.html(audio_html)

if __name__=="__main__":
    main()

# new environment python -m venv venv
# venv/Scripts/activate
#git config --global user.email "yoikoko@gmail.com"
#git config --global user.name "Jude-Kang"
# streamlit run app.py
# git reset
# git freeze > requirements.txt
# git add .
# git commit -m "Project Completed"
#(venv) PS D:\DataVoro_Project1> git remote remove origin
#(venv) PS D:\DataVoro_Project1> git remote add origin https://github.com/dongwan-kang/datavoro.git
# git branch -M main
#(venv) PS D:\DataVoro_Project1> git push -u origin main # 에러시에 git pull