import streamlit as st
import pandas as pd
import io
from datetime import datetime
from google import genai
from google.genai.errors import APIError

# --- 1. 환경 설정 및 상수 정의 ---
APP_TITLE = "친절한 고객 응대 AI 챗봇 (결제 불편 접수)"
SESSION_ID = f"session-{datetime.now().strftime('%Y%m%d%H%M%S')}"

# 지원되는 모델 목록 정의 (gemini-2.0-flash를 기본으로 설정)
# -exp 모델은 제외합니다.
AVAILABLE_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-pro",
    "gemini-2.0-pro",
    "gemini-2.0-ultra",
]

# 한국어 시스템 프롬프트 (요구사항에 맞춰 상세히 작성)
SYSTEM_PROMPT = """
당신은 고객의 불편/불만 사항을 접수하는 친절한 게임 서비스 담당 AI입니다.
사용자는 게임 결제 과정에서 겪은 불편이나 불만을 언급하고 있습니다.

당신의 응답 원칙:
1.  **태도:** 정중하고 공감 어린 말투를 사용하며, 고객의 불편에 대해 진심으로 죄송함을 표현합니다.
2.  **정보 수집:** 사용자가 언급한 불편 사항을 구체적으로 '무엇이, 언제, 어디서, 어떻게' 발생했는지 간결하게 정리하여 사용자에게 다시 안내하고, 이 내용을 고객 응대 담당자에게 정확히 전달하겠다는 취지로 안내합니다.
3.  **회신 요청:** 담당자 확인 후 회신을 위해 반드시 이메일 주소를 요청해야 합니다.
4.  **연락 거부 처리:** 만일 사용자가 이메일 주소 제공을 거부하면, "죄송하지만, 연락처 정보를 받지 못하여 담당자의 검토 내용을 받으실 수 없어요. 대신, 관련 내용을 정리하여 접수해 드릴 수는 있습니다."라고 정중히 안내하고 대화를 마무리합니다.
"""

# --- 2. 초기 상태 설정 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []
if "session_id" not in st.session_state:
    st.session_state.session_id = SESSION_ID
if "chat" not in st.session_state:
    st.session_state.chat = None
if "api_key" not in st.session_state:
    st.session_state.api_key = None
if "model_name" not in st.session_state:
    st.session_state.model_name = AVAILABLE_MODELS[0]

# --- 3. API 키 설정 및 클라이언트 초기화 ---
def get_api_key():
    """Streamlit Secrets에서 API 키를 가져오거나 사용자 입력을 요청합니다."""
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.session_state.api_key = api_key
    except:
        if not st.session_state.api_key:
            with st.sidebar:
                st.warning("`secrets.toml`에 API 키가 없습니다. 임시로 입력해주세요.")
                key_input = st.text_input("Gemini API Key 입력:", type="password", key="api_key_input")
                if key_input:
                    st.session_state.api_key = key_input

    return st.session_state.api_key

def initialize_client_and_chat(api_key, model_name):
    """Gemini 클라이언트와 채팅 세션을 초기화합니다."""
    if st.session_state.chat and st.session_state.model_name == model_name:
        return st.session_state.chat

    try:
        client = genai.Client(api_key=api_key)
        
        # 채팅 세션 생성
        st.session_state.chat = client.chats.create(
            model=model_name,
            config={"system_instruction": SYSTEM_PROMPT}
        )
        st.session_state.model_name = model_name
        st.info(f"선택된 모델: {model_name} (새 세션 시작)")
        return st.session_state.chat
    except Exception as e:
        st.error(f"클라이언트 초기화 오류: {e}")
        st.session_state.chat = None
        return None

# --- 4. 대화 기록 관리 함수 ---
def log_message(role, content):
    """대화 로그와 CSV 로깅 리스트에 메시지를 기록합니다."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # 일반 대화 히스토리 업데이트
    st.session_state.messages.append({"role": role, "content": content})
    # CSV 로깅 리스트 업데이트
    st.session_state.chat_log.append({
        "timestamp": timestamp,
        "session_id": st.session_state.session_id,
        "role": role,
        "content": content,
        "model": st.session_state.model_name
    })

def create_csv_download():
    """현재 대화 로그를 CSV 파일로 생성합니다."""
    df = pd.DataFrame(st.session_state.chat_log)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    
    return csv_buffer.getvalue().encode('utf-8')

def clear_conversation():
    """모든 대화 상태를 초기화합니다."""
    st.session_state.messages = []
    st.session_state.chat_log = []
    st.session_state.session_id = f"session-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    st.session_state.chat = None # 채팅 세션을 재초기화하도록 설정
    st.rerun()

# --- 5. Streamlit UI 및 메인 로직 ---
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# 5.1. 사이드바 (설정 및 도구)
with st.sidebar:
    st.header("설정 및 도구")

    # 모델 선택
    st.session_state.model_name = st.selectbox(
        "사용할 모델 선택",
        AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(st.session_state.model_name),
        on_change=lambda: st.session_state.__setitem__('chat', None) # 모델 변경 시 채팅 세션 초기화
    )

    # 대화 정보 표시
    st.markdown("---")
    st.subheader("대화 정보")
    st.text(f"세션 ID: {st.session_state.session_id}")
    st.text(f"메시지 수: {len(st.session_state.messages)}")

    # 대화 초기화 버튼
    if st.button("💬 대화 초기화", use_container_width=True):
        clear_conversation()
        st.success("대화가 초기화되었습니다.")

    # CSV 로깅 옵션
    st.markdown("---")
    st.subheader("로그 기록")
    log_csv = st.checkbox("CSV 자동 기록", value=True, help="모든 대화 내용을 CSV 로깅 목록에 기록합니다.")
    
    # 로그 다운로드 버튼
    if st.session_state.chat_log:
        st.download_button(
            label="⬇️ 로그 다운로드 (CSV)",
            data=create_csv_download(),
            file_name=f"chatbot_log_{st.session_state.session_id}.csv",
            mime="text/csv",
            use_container_width=True
        )

# 5.2. API 키 확인 및 클라이언트 초기화
api_key = get_api_key()
if not api_key:
    st.warning("계속하려면 Gemini API 키를 입력하거나 `secrets.toml`에 설정해야 합니다.")
    st.stop()

chat_session = initialize_client_and_chat(api_key, st.session_state.model_name)

if not chat_session:
    st.error("채팅 세션을 시작할 수 없습니다. API 키와 모델 설정을 확인해주세요.")
    st.stop()

# 5.3. 대화 히스토리 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5.4. 사용자 입력 및 응답 생성 로직
if prompt := st.chat_input("불편 사항을 말씀해주세요."):
    # 1. 사용자 메시지 기록 및 표시
    log_message("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 모델 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("답변을 생성 중입니다..."):
            try:
                # 429 에러는 SDK 내부에서 지수 백오프 방식으로 자동 재시도합니다.
                response = chat_session.send_message(prompt)
                ai_response = response.text
                st.markdown(ai_response)
                
                # 3. AI 응답 기록
                if log_csv:
                    log_message("assistant", ai_response)

            except APIError as e:
                # API 오류 발생 시 처리 (예: Rate limit 외 다른 문제)
                st.error(f"API 호출 중 오류가 발생했습니다: {e}")
                # 이 에러 메시지를 사용자에게 표시할 필요는 없으므로, 대신 로그에 기록
                if log_csv:
                    log_message("assistant", f"API Error: {e}")
            except Exception as e:
                st.error(f"알 수 없는 오류가 발생했습니다: {e}")
                if log_csv:
                    log_message("assistant", f"Unknown Error: {e}")

# 참고: 대화 히스토리 관리는 chat_session 객체가 담당하며,
# 메모리 절약을 위해 history를 6턴으로 제한하는 등의 고급 기능은
# 필요시 chat_session.get_history()를 사용하여 수동으로 구현할 수 있습니다.
# 본 앱은 SDK의 기본 Chat 세션 기능을 활용합니다.
