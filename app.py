import streamlit as st
from google import genai
from google.genai import types
from google.genai.errors import APIError
import time
import uuid
import csv
import io

# --- 설정 및 초기화 ---

# Gemini API 키 가져오기
def get_api_key():
    # 1. Streamlit Secrets에서 API 키를 시도합니다.
    try:
        if st.secrets["GEMINI_API_KEY"]:
            return st.secrets["GEMINI_API_KEY"]
    except KeyError:
        pass
    except Exception:
        pass

    # 2. Secrets에 없으면 사용자에게 임시 입력 창을 제공합니다.
    with st.sidebar:
        st.warning("`st.secrets['GEMINI_API_KEY']`가 설정되지 않았습니다.")
        api_key = st.text_input("Gemini API Key를 입력하세요:", type="password")
        return api_key

# 시스템 프롬프트 정의 (고객 응대 스펙 준수)
SYSTEM_PROMPT = """
당신은 쇼핑몰 고객 서비스 AI 챗봇입니다.

1.  **공감 및 말투**: 사용자는 쇼핑몰 구매 과정에서 겪은 불편/불만을 언급합니다. 이들의 불편함에 깊이 공감하며, 매우 정중하고 친절한 말투로 응답해야 합니다.
2.  **정보 수집 및 전달 안내**: 사용자가 언급한 불편 사항을 구체적으로 정리하여 (무엇이, 언제, 어디서, 어떻게 발생했는지) 수집하세요. 수집한 내용을 바탕으로 "이 내용을 고객 응대 담당자에게 전달하여 구체적인 해결 방안을 모색하겠다"는 취지로 명확히 안내해야 합니다.
3.  **연락처 요청**: 담당자 확인 후 회신을 위해 대화의 마지막 부분에는 반드시 **이메일 주소**를 요청해야 합니다.
4.  **연락 거부 처리**: 만일 사용자가 이메일 주소 제공을 거부할 경우, 다음 문장만을 사용하여 정중하게 안내합니다: "죄송하지만, 연락처 정보를 받지 못하여 담당자의 검토 내용을 받으실 수 없어요. 불편을 드려 다시 한번 사과드립니다."
"""

# 세션 상태 초기화 함수
def initialize_session_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        # Gemini API 형식에 맞는 대화 목록 (첫 번째는 시스템 설정)
        st.session_state.messages = [
            types.Content(role="system", parts=[types.Part.from_text(SYSTEM_PROMPT)])
        ]
    if "history" not in st.session_state:
        # Streamlit 표시 및 CSV 로깅을 위한 전체 대화 기록
        st.session_state.history = []
    if "model_name" not in st.session_state:
        st.session_state.model_name = "gemini-2.5-flash"
    if "log_to_csv" not in st.session_state:
        st.session_state.log_to_csv = False

# 대화 초기화
def reset_conversation():
    keys_to_reset = ["messages", "history", "session_id"]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    initialize_session_state()

# 대화 기록을 CSV 형식으로 변환
def convert_history_to_csv(history):
    output = io.StringIO()
    # CSV Writer 설정: 'text' 필드는 줄바꿈이 있을 수 있으므로 quotechar를 사용
    writer = csv.DictWriter(output, fieldnames=["SessionID", "Timestamp", "Role", "Text"], quoting=csv.QUOTE_MINIMAL)
    writer.writeheader()
    
    for entry in history:
        writer.writerow({
            "SessionID": st.session_state.session_id,
            "Timestamp": entry.get("timestamp"),
            "Role": entry.get("role"),
            "Text": entry.get("text")
        })
    return output.getvalue().encode('utf-8')


# --- Streamlit UI 및 로직 ---

st.set_page_config(
    page_title="Gemini 고객 불편 응대 챗봇", 
    layout="wide", 
    initial_sidebar_state="expanded"
)
st.title("🛍️ 쇼핑몰 고객 불편 응대 AI 챗봇")
st.caption("불편을 드려 죄송합니다. 담당자 전달을 위해 구체적인 내용을 말씀해 주세요.")

initialize_session_state()
API_KEY = get_api_key()

# 사이드바 설정
with st.sidebar:
    st.subheader("🤖 챗봇 설정")
    
    # 모델 선택
    available_models = ["gemini-2.5-flash", "gemini-2.5-pro"]
    st.session_state.model_name = st.selectbox(
        "사용할 모델 선택:",
        options=available_models,
        index=available_models.index(st.session_state.model_name)
    )

    # CSV 로깅 옵션
    st.session_state.log_to_csv = st.checkbox("대화 내용 CSV 자동 기록", value=st.session_state.log_to_csv)

    # 대화 초기화 버튼
    if st.button("🔄 대화 초기화 및 새 세션 시작"):
        reset_conversation()
        st.experimental_rerun()
    
    st.markdown("---")
    st.info(f"**세션 ID:** `{st.session_state.session_id}`\n\n**선택 모델:** `{st.session_state.model_name}`")

# API 키 유효성 검사
if not API_KEY:
    st.error("Gemini API 키를 입력하거나 `st.secrets`에 설정해야 챗봇을 사용할 수 있습니다.")
else:
    try:
        client = genai.Client(api_key=API_KEY)
    except Exception as e:
        st.error(f"Gemini 클라이언트 초기화 중 오류 발생: {e}")
        st.stop()


# 기존 대화 표시
for entry in st.session_state.history:
    with st.chat_message(entry["role"], avatar="🙋‍♂️" if entry["role"] == "user" else "🤖"):
        st.markdown(entry["text"])

# 사용자 입력 처리
if prompt := st.chat_input("여기에 불편 사항을 입력해 주세요..."):
    # 사용자 메시지 기록
    st.session_state.history.append({"timestamp": time.time(), "role": "user", "text": prompt})
    st.session_state.messages.append(types.Content(role="user", parts=[types.Part.from_text(prompt)]))

    # 사용자 메시지 표시
    with st.chat_message("user", avatar="🙋‍♂️"):
        st.markdown(prompt)

    # --- API 호출 및 히스토리 관리 ---
    
    # API 요청에 보낼 메시지 (시스템 프롬프트 + 최근 6턴 유지)
    # 메시지 리스트는 [system_prompt, user_1, model_1, user_2, model_2, ...] 순서이므로,
    # 1(system) + 6(user/model pairs) = 최대 7개 메시지를 유지
    if len(st.session_state.messages) > 7:
        # 시스템 프롬프트는 항상 유지하고, 그 이후의 메시지 중 가장 최근 6개만 사용
        api_messages = [st.session_state.messages[0]] + st.session_state.messages[-6:]
    else:
        api_messages = st.session_state.messages
    
    # 모델 호출
    with st.chat_message("model", avatar="🤖"):
        with st.spinner("담당자에게 전달할 내용을 검토하고 있습니다..."):
            response_text = ""
            max_retries = 3
            
            for attempt in range(max_retries):
                try:
                    response = client.models.generate_content(
                        model=st.session_state.model_name,
                        contents=api_messages,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT # 안정성을 위해 다시 전달
                        )
                    )
                    response_text = response.text
                    break # 성공하면 루프 탈출
                
                except APIError as e:
                    # 429 Rate Limit 오류 처리
                    if e.status_code == 429 and attempt < max_retries - 1:
                        st.warning(f"API 호출 제한에 도달했습니다. {attempt + 1}초 후 재시도합니다...")
                        time.sleep(1 * (attempt + 1)) # 지수 백오프 대신 단순 증가
                    elif e.status_code == 429:
                        st.error("API 호출 제한을 초과하여 응답을 받을 수 없습니다. 잠시 후 다시 시도해 주세요.")
                        response_text = "서비스 이용량이 많아 지금은 응답이 어렵습니다. 잠시 후 다시 시도해 주시면 감사하겠습니다."
                        break
                    else:
                        st.error(f"API 오류 발생: {e}")
                        response_text = "죄송합니다. 서비스 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
                        break
                
                except Exception as e:
                    st.error(f"예상치 못한 오류 발생: {e}")
                    response_text = "죄송합니다. 예상치 못한 문제로 응답이 어렵습니다."
                    break

            # AI 응답 표시
            st.markdown(response_text)
            
            # AI 응답 기록 및 세션 상태 업데이트
            if response_text:
                st.session_state.history.append({"timestamp": time.time(), "role": "model", "text": response_text})
                # API 메시지 목록에도 추가
                st.session_state.messages.append(types.Content(role="model", parts=[types.Part.from_text(response_text)]))
            
            # CSV 자동 기록 옵션이 켜져 있을 경우
            if st.session_state.log_to_csv:
                st.toast("대화 기록이 CSV에 자동 기록되었습니다.", icon="📝")
            
            # 스크롤 최하단으로 이동
            st.experimental_rerun()


# --- 로그 다운로드 버튼 (대화 창 하단) ---
if st.session_state.history:
    st.markdown("---")
    csv_data = convert_history_to_csv(st.session_state.history)
    st.download_button(
        label="📄 전체 대화 기록 다운로드 (CSV)",
        data=csv_data,
        file_name=f"chatbot_log_{st.session_state.session_id}.csv",
        mime="text/csv"
    )
