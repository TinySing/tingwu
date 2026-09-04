import json
import os
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from aliyunsdkcore.auth.credentials import AccessKeyCredential
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest
from flask import Flask, Response, flash, jsonify, redirect, render_template, request, send_from_directory, stream_with_context, url_for
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
API_DOMAIN = "tingwu.cn-beijing.aliyuncs.com"
API_VERSION = "2023-09-30"
DEFAULT_TINGWU_PARAMETERS = {
    "TargetAudioFormat": "mp3",
    "Transcription": {
        "OutputLevel": 2,
        "DiarizationEnabled": True,
        "Diarization": {"SpeakerCount": 0},
    },
    "TranslationEnabled": False,
    "AutoChaptersEnabled": False,
    "MeetingAssistanceEnabled": False,
    "SummarizationEnabled": True,
    "Summarization": {"Types": ["QuestionsAnswering"]},
    "PptExtractionEnabled": False,
    "TextPolishEnabled": False,
    "ServiceInspectionEnabled": False,
    "ContentExtractionEnabled": False,
    "IdentityRecognitionEnabled": False,
    "CustomPromptEnabled": True,
}

DEFAULT_TINGWU_MEETING_CONFIG = {
    "type": "realtime",
    "sourceLanguage": "cn",
    "format": "pcm",
    "sampleRate": 16000,
    "parameters": json.loads(json.dumps(DEFAULT_TINGWU_PARAMETERS)),
}

DEFAULT_TINGWU_OFFLINE_CONFIG = {
    "type": "offline",
    "sourceLanguage": "cn",
    "parameters": json.loads(json.dumps(DEFAULT_TINGWU_PARAMETERS)),
}

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {"mp3", "mp4", "wav", "flac", "ogg", "aac", "m4a", "webm", "mkv", "avi", "mov", "pcm"}

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "tingwu-local-console")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB


def load_settings():
    settings = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                settings[key.strip()] = value.strip()
    return settings


def masked(value):
    if not value:
        return "未配置"
    return f"{value[:4]}{'*' * max(4, len(value) - 8)}{value[-4:]}" if len(value) > 8 else "已配置"


def save_settings(values):
    content = "\n".join(f"{key}={values.get(key, '')}" for key in (
        "TINGWU_ACCESS_KEY_ID", "TINGWU_ACCESS_KEY_SECRET", "TINGWU_APP_KEY", "DEEPSEEK_API_KEY"
    )) + "\n"
    descriptor, temporary_path = tempfile.mkstemp(dir=BASE_DIR, prefix=".tingwu-", text=True)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(content)
    os.chmod(temporary_path, 0o600)
    os.replace(temporary_path, ENV_FILE)


def client():
    settings = load_settings()
    key_id = settings.get("TINGWU_ACCESS_KEY_ID")
    key_secret = settings.get("TINGWU_ACCESS_KEY_SECRET")
    if not key_id or not key_secret:
        raise ValueError("请先在配置中心填写 AccessKey ID 和 AccessKey Secret。")
    return AcsClient("cn-beijing", credential=AccessKeyCredential(key_id, key_secret))


_PROXY_ENV_KEYS = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy",
    "SOCKS_PROXY", "SOCKS5_PROXY", "socks_proxy", "socks5_proxy",
)


@contextmanager
def without_proxy_env():
    """临时清掉代理环境变量，避免本机代理导致听悟 API 403。"""
    saved = {key: os.environ.pop(key) for key in _PROXY_ENV_KEYS if key in os.environ}
    try:
        yield
    finally:
        os.environ.update(saved)


def tingwu_request(method, path, body=None, query=None):
    api_request = CommonRequest()
    api_request.set_accept_format("json")
    api_request.set_domain(API_DOMAIN)
    api_request.set_version(API_VERSION)
    api_request.set_protocol_type("https")
    api_request.set_method(method)
    api_request.set_uri_pattern(path)
    api_request.add_header("Content-Type", "application/json")
    for key, value in (query or {}).items():
        api_request.add_query_param(key, str(value))
    if body is not None:
        api_request.set_content(json.dumps(body, ensure_ascii=False).encode("utf-8"))
    with without_proxy_env():
        response = client().do_action_with_exception(api_request)
    return json.loads(response.decode("utf-8"))


def api_call(callback):
    try:
        return callback(), None
    except Exception as error:  # SDK errors include request details useful to the operator.
        return None, str(error)


def task_result_links(response):
    """Extract direct-download URLs without JSON's HTML-safe escaping."""
    data = (response or {}).get("Data", {})
    links = []
    output_names = {
        "OutputMp3Path": "转码音频", "OutputMp4Path": "转码视频",
        "OutputThumbnailPath": "视频缩略图", "OutputSpectrumPath": "音频波形",
    }
    for field, label in output_names.items():
        if data.get(field):
            links.append((label, data[field]))
    for name, value in data.get("Result", {}).items():
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            links.append((name, value))
    return links


def pretty_json(response):
    return json.dumps(response, ensure_ascii=False, indent=2)


def fetch_url_text(url, timeout=30):
    import urllib.request

    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def format_ms(ms):
    if ms is None:
        return "00:00"
    sec = max(0, int(ms) // 1000)
    return f"{sec // 60:02d}:{sec % 60:02d}"


def assemble_transcription_text(transcription_data):
    """
    将听悟 Transcription JSON 拼成可读文本。
    官方结构：Transcription.Paragraphs[].Words[]，同 SentenceId 的 Word 组成一句。
    """
    if not transcription_data:
        return ""

    root = transcription_data
    if isinstance(root, str):
        try:
            root = json.loads(root)
        except json.JSONDecodeError:
            return root.strip()

    if isinstance(root, dict) and "Transcription" in root:
        root = root["Transcription"]

    paragraphs = []
    if isinstance(root, dict):
        paragraphs = root.get("Paragraphs") or []
    elif isinstance(root, list):
        paragraphs = root

    lines = []

    def words_to_sentences(words):
        by_sentence = {}
        order = []
        for word in words or []:
            if not isinstance(word, dict):
                continue
            sid = word.get("SentenceId", 0)
            if sid not in by_sentence:
                by_sentence[sid] = []
                order.append(sid)
            by_sentence[sid].append(word)
        sentences = []
        for sid in sorted(order, key=lambda x: (by_sentence[x][0].get("Start") or 0, x)):
            chunk = sorted(by_sentence[sid], key=lambda w: (w.get("Start") or 0, w.get("Id") or 0))
            text = "".join(w.get("Text", "") for w in chunk).strip()
            if not text:
                continue
            start = chunk[0].get("Start")
            sentences.append((start, text))
        return sentences

    if paragraphs:
        for para in paragraphs:
            if not isinstance(para, dict):
                continue
            speaker = para.get("SpeakerId")
            spk_label = f"发言人 {speaker}" if speaker not in (None, "") else "发言人"
            sentences = words_to_sentences(para.get("Words"))
            if not sentences:
                continue
            lines.append(f"【{spk_label}】")
            for start, text in sentences:
                lines.append(f"[{format_ms(start)}] {text}")
            lines.append("")
    else:
        # 兜底：顶层 Words 列表
        words = root.get("Words") if isinstance(root, dict) else []
        for start, text in words_to_sentences(words):
            lines.append(f"[{format_ms(start)}] {text}")

    return "\n".join(lines).strip()


def get_task_transcription_text(task_response):
    """从 GetTaskInfo 响应中获取并拼接转写全文。"""
    data = (task_response or {}).get("Data") or {}
    result = data.get("Result") or {}
    transcription_ref = result.get("Transcription")
    if not transcription_ref:
        return ""

    transcription_data = None
    if isinstance(transcription_ref, str):
        if transcription_ref.startswith(("http://", "https://")):
            try:
                raw = fetch_url_text(transcription_ref)
                transcription_data = json.loads(raw)
            except Exception as error:
                return f"[转写结果下载或解析失败: {error}]"
        else:
            try:
                transcription_data = json.loads(transcription_ref)
            except json.JSONDecodeError:
                return transcription_ref
    elif isinstance(transcription_ref, dict):
        transcription_data = transcription_ref

    return assemble_transcription_text(transcription_data)


def load_result_payload(ref):
    """下载并解析 Result 里的 URL / JSON 文本。"""
    if not ref:
        return None
    if isinstance(ref, (dict, list)):
        return ref
    if not isinstance(ref, str):
        return None
    text = ref.strip()
    if text.startswith(("http://", "https://")):
        try:
            text = fetch_url_text(text)
        except Exception as error:
            return f"[结果下载失败: {error}]"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def summarization_to_markdown(data):
    """对齐官方 Summarization JSON，转成 Markdown。"""
    if not data:
        return ""
    if isinstance(data, str):
        return data.strip()
    root = data.get("Summarization", data) if isinstance(data, dict) else {}
    if not isinstance(root, dict):
        return json.dumps(data, ensure_ascii=False, indent=2)

    parts = []
    paragraph = (root.get("ParagraphSummary") or "").strip()
    if paragraph:
        parts.append("## 全文摘要\n\n" + paragraph)

    conversational = root.get("ConversationalSummary") or []
    if conversational:
        parts.append("## 发言总结")
        for item in conversational:
            if not isinstance(item, dict):
                continue
            name = item.get("SpeakerName") or (
                f"发言人 {item.get('SpeakerId')}" if item.get("SpeakerId") not in (None, "") else "发言人"
            )
            summary = (item.get("Summary") or "").strip()
            parts.append(f"### {name}\n\n{summary}")

    qa_list = root.get("QuestionsAnsweringSummary") or []
    if qa_list:
        parts.append("## 问答回顾")
        for index, item in enumerate(qa_list, 1):
            if not isinstance(item, dict):
                continue
            question = (item.get("Question") or "").strip()
            answer = (item.get("Answer") or "").strip()
            parts.append(f"**Q{index}.** {question}\n\n**A{index}.** {answer}")

    mind = root.get("MindMapSummary") or root.get("MindMap") or root.get("MindMapTree")
    if mind:
        mind_text = mind if isinstance(mind, str) else json.dumps(mind, ensure_ascii=False, indent=2)
        parts.append("## 思维导图\n\n```json\n" + mind_text + "\n```")

    return "\n\n".join(parts).strip()


def custom_prompt_items(data):
    """把 CustomPrompt 载荷归一成条目列表。"""
    if not data:
        return []
    if isinstance(data, str):
        parsed = parse_jsonish_object(data)
        return custom_prompt_items(parsed) if parsed else []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        payload = data.get("CustomPrompt", data)
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            if "Result" in payload or "Name" in payload:
                return [payload]
            items = payload.get("Contents") or payload.get("items") or []
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
            return [payload]
    return []


def parse_jsonish_object(text):
    """尽量从模型输出里解析 JSON 对象，兼容 ```json 代码块包裹。"""
    if not text or not isinstance(text, str):
        return None
    s = text.strip()
    if not s:
        return None
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    try:
        parsed = json.loads(s)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        start = s.find("{")
        end = s.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(s[start:end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


def custom_prompt_to_markdown(data):
    """对齐官方 CustomPrompt JSON：Name + Result（Result 常为 Markdown）。"""
    if not data:
        return ""
    if isinstance(data, str):
        return data.strip()

    parts = []
    for item in custom_prompt_items(data):
        name = item.get("Name") or "自定义总结"
        result = item.get("Result") or ""
        if isinstance(result, (dict, list)):
            result = json.dumps(result, ensure_ascii=False, indent=2)
        result = str(result).strip()
        if not result:
            continue
        header = f"## {name}"
        if item.get("Truncated"):
            header += "\n\n> 内容已截断（超过模型 token 限制）"
        parts.append(f"{header}\n\n{result}")
    return "\n\n".join(parts).strip()


def get_task_summary_blocks(task_response):
    """提取自定义 Prompt 总结和大模型摘要，供页面按 Markdown 渲染。"""
    result = ((task_response or {}).get("Data") or {}).get("Result") or {}
    blocks = []
    custom_payload = load_result_payload(result.get("CustomPrompt"))

    for index, item in enumerate(custom_prompt_items(custom_payload)):
        raw_result = item.get("Result")
        if isinstance(raw_result, dict):
            parsed = raw_result
        elif isinstance(raw_result, str):
            parsed = parse_jsonish_object(raw_result)
        else:
            parsed = None
        if not parsed:
            continue
        summary_md = (parsed.get("summaryMarkdown") or "").strip()
        if not summary_md:
            continue
        topic = (parsed.get("topic") or item.get("Name") or "会议纪要").strip()
        preview = (parsed.get("summaryPreview") or "").strip()
        blocks.append({
            "id": f"summary-markdown-{index}",
            "title": f"summaryMarkdown · {topic}",
            "markdown": summary_md,
            "topic": topic,
            "summary_preview": preview,
        })

    custom_md = custom_prompt_to_markdown(custom_payload)
    if custom_md:
        blocks.append({"id": "custom-prompt", "title": "自定义 Prompt 原始输出", "markdown": custom_md})

    summary_md = summarization_to_markdown(load_result_payload(result.get("Summarization")))
    if summary_md:
        blocks.append({"id": "summarization", "title": "大模型摘要", "markdown": summary_md})

    return blocks


def object_json(value, label):
    if not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label}不是有效的 JSON：{error.msg}") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{label}必须是 JSON 对象。")
    return parsed


def load_app_key():
    app_key = load_settings().get("TINGWU_APP_KEY", "")
    if not app_key:
        raise ValueError("请先在配置中心填写 AppKey。")
    return app_key


def config_get(config, *keys, default=None):
    for key in keys:
        value = config.get(key)
        if value not in (None, ""):
            return value
    return default


def meeting_prompt_parameters(form):
    prompt = form.get("custom_prompt", "").strip()
    if not prompt:
        raise ValueError("请输入 MEETING_CUSTOM_PROMPT_CONTENT。")
    if "{Transcription}" not in prompt:
        raise ValueError("MEETING_CUSTOM_PROMPT_CONTENT 必须包含 {Transcription}。")
    config = object_json(form.get("meeting_config", ""), "buildTingwuMeetingCreatePayload 配置")
    parameters = config_get(config, "parameters", "Parameters", default={})
    if not isinstance(parameters, dict):
        raise ValueError("配置中的 parameters 必须是 JSON 对象。")
    parameters = json.loads(json.dumps(parameters))
    parameters["CustomPromptEnabled"] = True
    parameters["CustomPrompt"] = {
        "Contents": [{
            "Name": "summary",
            "Prompt": prompt,
            "Model": "tingwu-plus",
            "TransType": "sentence-chat",
        }]
    }
    return config, parameters


def create_task_payload(form):
    app_key = load_app_key()
    config, parameters = meeting_prompt_parameters(form)
    task_mode = form.get("task_mode") or config_get(config, "type", default="offline")
    if task_mode not in ("offline", "realtime"):
        raise ValueError("调用模式只支持离线音视频转写或实时会议转写。")

    input_data = {
        "SourceLanguage": config_get(config, "sourceLanguage", "SourceLanguage", default="cn"),
    }
    optional_input = (
        ("TaskKey", "taskKey", "TaskKey"),
        ("OutputPath", "outputPath", "OutputPath"),
        ("AudioChannelMode", "audioChannelMode", "AudioChannelMode"),
    )
    for api_key, camel_key, pascal_key in optional_input:
        value = config_get(config, camel_key, pascal_key)
        if value:
            input_data[api_key] = value
    language_hints = config_get(config, "languageHints", "LanguageHints")
    if language_hints:
        input_data["LanguageHints"] = language_hints
    if config_get(config, "progressiveCallbacksEnabled", "ProgressiveCallbacksEnabled"):
        input_data["ProgressiveCallbacksEnabled"] = True
    if config_get(config, "multipleStreamsEnabled", "MultipleStreamsEnabled"):
        input_data["MultipleStreamsEnabled"] = True

    if task_mode == "offline":
        file_url = form.get("file_url", "").strip() or config_get(config, "fileUrl", "FileUrl", default="")
        if not file_url:
            raise ValueError("离线任务必须填写音视频 URL。")
        lowered = file_url.lower()
        if any(host in lowered for host in ("://127.0.0.1", "://localhost", "://0.0.0.0", "://[::1]")):
            raise ValueError("离线任务的音视频 URL 必须是听悟云端可访问的公网地址；本机 127.0.0.1/localhost 上传地址无效，请改用 OSS 等公网链接。")
        input_data["FileUrl"] = file_url
    else:
        input_data["Format"] = config_get(config, "format", "Format", default="pcm")
        input_data["SampleRate"] = int(config_get(config, "sampleRate", "SampleRate", default=16000))

    return {
        "AppKey": app_key,
        "Input": input_data,
        "Parameters": parameters,
    }, {"type": task_mode}


def rerun_meeting_payload(form):
    task_id = form.get("task_id", "").strip()
    if not task_id:
        raise ValueError("请输入要重新生成的会议 TaskId。")
    app_key = load_app_key()
    config, parameters = meeting_prompt_parameters(form)
    # 重跑复用已有转写结果，官方不支持再次转码或再次运行语音转写。
    parameters.pop("TargetAudioFormat", None)
    parameters.pop("Transcoding", None)
    parameters.pop("Transcription", None)
    return {
        "AppKey": app_key,
        "Input": {
            "TaskId": task_id,
            "SourceLanguage": config_get(config, "sourceLanguage", "SourceLanguage", default="cn"),
        },
        "Parameters": parameters,
    }


@app.route("/")
def index():
    return redirect(url_for("create_task"))


@app.route("/settings", methods=["GET", "POST"])
def settings():
    current = load_settings()
    if request.method == "POST":
        updated = current.copy()
        for key in ("TINGWU_ACCESS_KEY_ID", "TINGWU_ACCESS_KEY_SECRET", "TINGWU_APP_KEY", "DEEPSEEK_API_KEY"):
            value = request.form.get(key, "").strip()
            if value:
                updated[key] = value
        save_settings(updated)
        flash("配置已保存到本机 .env 文件。", "success")
        return redirect(url_for("settings"))
    return render_template("settings.html", settings=current, masked=masked)


@app.route("/create-task", methods=["GET", "POST"])
def create_task():
    result = None
    meeting_join_url = ""
    task_id = ""
    form_data = request.form if request.method == "POST" else {}
    if request.method == "POST":
        try:
            payload, query = create_task_payload(request.form)
            result, error = api_call(lambda: tingwu_request("PUT", "/openapi/tingwu/v2/tasks", payload, query))
            if error:
                flash(error, "error")
            elif result and query.get("type") == "realtime":
                data = result.get("Data") or {}
                meeting_join_url = data.get("MeetingJoinUrl", "")
                task_id = data.get("TaskId", "")
        except ValueError as error:
            flash(str(error), "error")
    return render_template(
        "create_task.html",
        default_realtime_config=pretty_json(DEFAULT_TINGWU_MEETING_CONFIG),
        default_offline_config=pretty_json(DEFAULT_TINGWU_OFFLINE_CONFIG),
        form_data=form_data,
        result=result,
        result_text=pretty_json(result) if result else "",
        meeting_join_url=meeting_join_url,
        task_id=task_id,
    )


@app.route("/rerun-meeting", methods=["GET", "POST"])
def rerun_meeting():
    result = None
    rerun_task_id = ""
    form_data = request.form if request.method == "POST" else {}
    if request.method == "POST":
        try:
            payload = rerun_meeting_payload(request.form)
            result, error = api_call(lambda: tingwu_request(
                "PUT", "/openapi/tingwu/v2/tasks", payload, {"type": "offline"}
            ))
            if error:
                flash(error, "error")
            elif str(result.get("Code")) == "0":
                rerun_task_id = (result.get("Data") or {}).get("TaskId", "")
                flash("重跑提交成功，正在等待结果…", "success")
            else:
                flash(result.get("Message", "重跑提交失败。"), "error")
        except ValueError as error:
            flash(str(error), "error")
    return render_template(
        "rerun_meeting.html",
        default_config=pretty_json(DEFAULT_TINGWU_MEETING_CONFIG),
        form_data=form_data,
        rerun_task_id=rerun_task_id,
    )


@app.route("/get-task-info", methods=["GET", "POST"])
def get_task_info():
    if request.method == "GET":
        return render_template("get_task_info.html", summary_blocks=[])
    task_id = request.form.get("task_id", "").strip()
    if not task_id:
        flash("请输入 TaskId。", "error")
        return redirect(url_for("get_task_info"))
    result, error = api_call(lambda: tingwu_request("GET", f"/openapi/tingwu/v2/tasks/{task_id}"))
    if error:
        flash(error, "error")
        return redirect(url_for("get_task_info"))
    transcription_text = get_task_transcription_text(result)
    summary_blocks = get_task_summary_blocks(result)
    task_status = ((result or {}).get("Data") or {}).get("TaskStatus", "")
    return render_template(
        "get_task_info.html",
        query_result=result,
        query_result_text=pretty_json(result),
        transcription_text=transcription_text,
        summary_blocks=summary_blocks,
        queried_task_id=task_id,
        query_result_links=task_result_links(result),
        task_status=task_status,
    )


@app.route("/phrases")
def phrases():
    result, error = api_call(lambda: tingwu_request("GET", "/openapi/tingwu/v2/resources/phrases"))
    if error:
        flash(error, "error")
    return render_template("phrases.html", phrases=(result or {}).get("Data", {}).get("Phrases", []))


def phrase_payload(form):
    word_weights = form.get("word_weights", "").strip()
    try:
        parsed = json.loads(word_weights)
        if not isinstance(parsed, dict) or not parsed:
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        raise ValueError("热词权重必须是非空 JSON 对象，例如：{\"听悟\": 3}。")
    return {
        "Name": form.get("name", "").strip(),
        "Description": form.get("description", "").strip(),
        "WordWeights": json.dumps(parsed, ensure_ascii=False),
    }


@app.post("/phrases/create")
def create_phrase():
    try:
        payload = phrase_payload(request.form)
        if not payload["Name"]:
            raise ValueError("请填写词表名称。")
        _, error = api_call(lambda: tingwu_request("POST", "/openapi/tingwu/v2/resources/phrases", payload))
        flash(error or "热词词表已创建。", "error" if error else "success")
    except ValueError as error:
        flash(str(error), "error")
    return redirect(url_for("phrases"))


@app.post("/phrases/<phrase_id>/update")
def update_phrase(phrase_id):
    try:
        payload = phrase_payload(request.form)
        if not payload["Name"]:
            raise ValueError("请填写词表名称。")
        _, error = api_call(lambda: tingwu_request("PUT", f"/openapi/tingwu/v2/resources/phrases/{phrase_id}", payload))
        flash(error or "热词词表已更新。", "error" if error else "success")
    except ValueError as error:
        flash(str(error), "error")
    return redirect(url_for("phrases"))


@app.post("/phrases/<phrase_id>/delete")
def delete_phrase(phrase_id):
    _, error = api_call(lambda: tingwu_request("DELETE", f"/openapi/tingwu/v2/resources/phrases/{phrase_id}"))
    flash(error or "热词词表已删除。", "error" if error else "success")
    return redirect(url_for("phrases"))


@app.get("/phrases/<phrase_id>")
def phrase_detail(phrase_id):
    result, error = api_call(lambda: tingwu_request("GET", f"/openapi/tingwu/v2/resources/phrases/{phrase_id}"))
    if error:
        flash(error, "error")
        return redirect(url_for("phrases"))
    return render_template("phrase_detail.html", phrase=(result or {}).get("Data", {}))


@app.get("/api/phrases/<phrase_id>")
def api_phrase_detail(phrase_id):
    result, error = api_call(lambda: tingwu_request("GET", f"/openapi/tingwu/v2/resources/phrases/{phrase_id}"))
    if error:
        return jsonify({"ok": False, "error": error}), 500
    return jsonify({"ok": True, "data": (result or {}).get("Data", {})})


@app.route("/realtime")
def realtime():
    return render_template("realtime.html")


@app.post("/api/realtime/create")
def api_realtime_create():
    data = request.get_json(silent=True) or {}
    app_key = load_settings().get("TINGWU_APP_KEY", "")
    if not app_key:
        return jsonify({"ok": False, "error": "请先在配置中心填写 AppKey。"}), 400

    input_data = {
        "Format": data.get("format", "pcm"),
        "SampleRate": int(data.get("sampleRate", 16000)),
    }
    if data.get("sourceLanguage"):
        input_data["SourceLanguage"] = data["sourceLanguage"]
    if data.get("phraseId"):
        input_data["PhraseId"] = data["phraseId"]

    parameters = {}
    transcription = {}
    if data.get("diarizationEnabled"):
        transcription["DiarizationEnabled"] = True
        transcription["Diarization"] = {"SpeakerCount": int(data.get("speakerCount", 0))}
    if data.get("outputLevel"):
        transcription["OutputLevel"] = int(data["outputLevel"])
    if transcription:
        parameters["Transcription"] = transcription
    if data.get("translationLanguages"):
        parameters["TranslationEnabled"] = True
        parameters["Translation"] = {"TargetLanguages": data["translationLanguages"]}
    if data.get("summarizationEnabled"):
        parameters["SummarizationEnabled"] = True
        parameters["Summarization"] = {"Types": data.get("summaryTypes", ["QuestionsAnswering"])}

    payload = {"AppKey": app_key, "Input": input_data}
    if parameters:
        payload["Parameters"] = parameters
    result, error = api_call(lambda: tingwu_request(
        "PUT", "/openapi/tingwu/v2/tasks", payload, {"type": "realtime"}
    ))
    if error:
        return jsonify({"ok": False, "error": error}), 500
    task_id = (result.get("Data") or {}).get("TaskId", "")
    meeting_join_url = (result.get("Data") or {}).get("MeetingJoinUrl", "")
    return jsonify({
        "ok": True,
        "taskId": task_id,
        "meetingJoinUrl": meeting_join_url,
        "raw": result,
    })


@app.post("/api/realtime/stop")
def api_realtime_stop():
    data = request.get_json(silent=True) or {}
    task_id = data.get("taskId", "").strip()
    if not task_id:
        return jsonify({"ok": False, "error": "缺少 taskId"}), 400
    app_key = load_settings().get("TINGWU_APP_KEY", "")
    if not app_key:
        return jsonify({"ok": False, "error": "请先在配置中心填写 AppKey。"}), 400
    payload = {"AppKey": app_key, "Input": {"TaskId": task_id}}
    result, error = api_call(lambda: tingwu_request(
        "PUT", "/openapi/tingwu/v2/tasks", payload, {"type": "realtime", "operation": "stop"}
    ))
    if error:
        return jsonify({"ok": False, "error": error}), 500
    return jsonify({"ok": True, "raw": result})


@app.post("/api/upload")
def api_upload():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "未选择文件"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"ok": False, "error": "未选择文件"}), 400
    ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"ok": False, "error": f"不支持的文件格式: .{ext}"}), 400
    safe_name = f"{uuid.uuid4().hex[:12]}_{secure_filename(f.filename)}"
    f.save(UPLOAD_DIR / safe_name)
    file_url = request.host_url.rstrip("/") + f"/files/{safe_name}"
    return jsonify({"ok": True, "url": file_url, "filename": safe_name})


@app.get("/files/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


def extract_customprompt_content(result_data):
    """
    从任务 Result 中尽量提取“自定义 Prompt 输出”的可读文本。
    若 Result 是含 summaryMarkdown 的 JSON，优先返回该 Markdown。
    """
    if not isinstance(result_data, dict):
        return json.dumps(result_data, ensure_ascii=False, indent=2)

    def try_val(val):
        if isinstance(val, str):
            s = val.strip()
            if not s:
                return None
            if s.startswith(("http://", "https://")):
                try:
                    import urllib.request

                    with urllib.request.urlopen(s, timeout=15) as resp:
                        text = resp.read().decode("utf-8", errors="replace")
                        try:
                            parsed = json.loads(text)
                            md = extract_summary_markdown_text(parsed)
                            if md:
                                return md
                            return json.dumps(parsed, ensure_ascii=False, indent=2)
                        except Exception:
                            return text
                except Exception:
                    return f"[无法下载: {s}]"
            if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")) or s.startswith("```"):
                parsed = parse_jsonish_object(s)
                if parsed:
                    md = extract_summary_markdown_text(parsed)
                    if md:
                        return md
                    return json.dumps(parsed, ensure_ascii=False, indent=2)
                if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                    try:
                        parsed = json.loads(s)
                        return json.dumps(parsed, ensure_ascii=False, indent=2)
                    except Exception:
                        pass
            return s
        if isinstance(val, (dict, list)):
            md = extract_summary_markdown_text(val)
            if md:
                return md
            return json.dumps(val, ensure_ascii=False, indent=2)
        return None

    # 1) 明确优先字段（对齐官方 GetTaskInfo：Result 里主要是 CustomPrompt / Summarization 等）
    for key in ("CustomPrompt", "CustomPromptResult", "Summarization", "summary"):
        if key in result_data:
            extracted = try_val(result_data.get(key))
            if extracted is not None:
                return extracted

    # 2) 模糊匹配（尽量避免 mp3/mp4 等大文件）
    fuzzy_substrings = ("customprompt", "summarization", "summary", "prompt")
    for key, val in result_data.items():
        k = str(key).lower()
        if any(sub in k for sub in fuzzy_substrings):
            extracted = try_val(val)
            if extracted:
                return extracted

    # 3) 兜底：返回非空字符串/结构体（不主动拉 mp3/mp4）
    skip_substrings = ("mp3", "mp4", "wav", "m4a", "ogg", "mov", "avi")
    for key, val in result_data.items():
        k = str(key).lower()
        if any(sub in k for sub in skip_substrings):
            continue
        if isinstance(val, str) and val.strip():
            extracted = try_val(val)
            if extracted:
                return extracted

    return json.dumps(result_data, ensure_ascii=False, indent=2)


def extract_summary_markdown_text(payload):
    """从 CustomPrompt 载荷或单条 Result JSON 中提取 summaryMarkdown。"""
    if isinstance(payload, dict) and payload.get("summaryMarkdown"):
        return str(payload.get("summaryMarkdown") or "").strip()
    parts = []
    for item in custom_prompt_items(payload):
        raw = item.get("Result")
        parsed = raw if isinstance(raw, dict) else parse_jsonish_object(raw if isinstance(raw, str) else "")
        if not parsed:
            continue
        md = (parsed.get("summaryMarkdown") or "").strip()
        if md:
            parts.append(md)
    return "\n\n".join(parts).strip()


def extract_summarization_content(result_data):
    """
    从任务 Result 中尽量提取“大模型摘要/摘要类”的可读文本。
    官方文档中主要字段名是：Result.Summarization（以及若干同义字段）。
    """
    if not isinstance(result_data, dict):
        return json.dumps(result_data, ensure_ascii=False, indent=2)

    def try_val(val):
        if isinstance(val, str):
            s = val.strip()
            if not s:
                return None
            if s.startswith(("http://", "https://")):
                try:
                    import urllib.request

                    with urllib.request.urlopen(s, timeout=15) as resp:
                        text = resp.read().decode("utf-8", errors="replace")
                        try:
                            parsed = json.loads(text)
                            return json.dumps(parsed, ensure_ascii=False, indent=2)
                        except Exception:
                            return text
                except Exception:
                    return f"[无法下载: {s}]"
            if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                try:
                    parsed = json.loads(s)
                    return json.dumps(parsed, ensure_ascii=False, indent=2)
                except Exception:
                    pass
            return s
        if isinstance(val, (dict, list)):
            return json.dumps(val, ensure_ascii=False, indent=2)
        return None

    # 明确优先字段（对齐官方 GetTaskInfo：Result.Summarization）
    for key in ("Summarization", "summary"):
        if key in result_data:
            extracted = try_val(result_data.get(key))
            if extracted is not None:
                return extracted

    # 兜底模糊匹配
    fuzzy_substrings = ("summarization", "summary", "prompt")
    skip_substrings = ("mp3", "mp4", "wav", "m4a", "ogg", "mov", "avi")
    for key, val in result_data.items():
        k = str(key).lower()
        if any(sub in k for sub in skip_substrings):
            continue
        if any(sub in k for sub in fuzzy_substrings):
            extracted = try_val(val)
            if extracted:
                return extracted

    return json.dumps(result_data, ensure_ascii=False, indent=2)


@app.get("/api/task-poll/<task_id>")
def api_task_poll(task_id):
    result, error = api_call(lambda: tingwu_request("GET", f"/openapi/tingwu/v2/tasks/{task_id}"))
    if error:
        return jsonify({"ok": False, "error": error}), 500
    data = result.get("Data") or {}
    status = data.get("TaskStatus", "")
    out = {"ok": True, "status": status, "raw": result}
    if status == "COMPLETED":
        result_data = data.get("Result", {})
        out["customPromptContent"] = extract_customprompt_content(result_data)
        out["summarizationContent"] = extract_summarization_content(result_data)
        out["content"] = out["customPromptContent"] or out["summarizationContent"]
    elif status in ("FAILED", "ERROR", "INVALID"):
        # 失败时把官方 ErrorCode/ErrorMessage 回传给前端，便于排查
        out["errorCode"] = data.get("ErrorCode", "")
        out["errorMessage"] = data.get("ErrorMessage", "")
    return jsonify(out)


@app.route("/prompt-tuner")
def prompt_tuner():
    return render_template("prompt_tuner.html")


def deepseek_chat(messages, api_key):
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.7,
        max_tokens=4096,
    )
    return resp.choices[0].message.content.strip()


def poll_task_result(task_id, timeout=180):
    """Poll GetTaskInfo until completed or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        result, error = api_call(lambda: tingwu_request("GET", f"/openapi/tingwu/v2/tasks/{task_id}"))
        if error:
            return None, error
        data = (result.get("Data") or {})
        status = data.get("TaskStatus", "")
        if status == "COMPLETED":
            result_data = data.get("Result", {})
            custom_result = extract_customprompt_content(result_data)
            return custom_result, None
        elif status in ("FAILED", "ERROR", "INVALID"):
            # 失败时返回官方错误信息（优先 ErrorMessage）
            err_msg = data.get("ErrorMessage") or data.get("ErrorCode") or ""
            return None, f"任务失败: {status}{(' - ' + err_msg) if err_msg else ''}"
        time.sleep(5)
    return None, "轮询超时"


def sse_event(event, data):
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


@app.post("/api/prompt-tuner/run")
def api_prompt_tuner_run():
    req = request.get_json(silent=True) or {}
    source_task_id = req.get("taskId", "").strip()
    goal = req.get("goal", "").strip()
    max_rounds = min(int(req.get("maxRounds", 5)), 10)
    initial_prompt = req.get("initialPrompt", "").strip()
    feedback = req.get("feedback", "").strip()
    last_result = req.get("lastResult", "").strip()
    try:
        round_offset = max(0, int(req.get("roundOffset", 0) or 0))
    except (TypeError, ValueError):
        round_offset = 0

    if not source_task_id:
        return jsonify({"ok": False, "error": "请输入源会议 TaskId"}), 400
    if not goal and not feedback:
        return jsonify({"ok": False, "error": "请描述期望的输出结果，或填写追问反馈"}), 400

    settings = load_settings()
    deepseek_key = settings.get("DEEPSEEK_API_KEY", "")
    app_key = settings.get("TINGWU_APP_KEY", "")
    if not deepseek_key:
        return jsonify({"ok": False, "error": "请先在配置中心填写 DeepSeek API Key"}), 400
    if not app_key:
        return jsonify({"ok": False, "error": "请先在配置中心填写 AppKey"}), 400

    effective_goal = goal
    if feedback:
        effective_goal = (goal + "\n\n用户补充反馈：\n" + feedback).strip() if goal else ("用户补充反馈：\n" + feedback)

    def generate():
        current_prompt = initial_prompt
        history = []
        actual_output = last_result

        for local_round in range(1, max_rounds + 1):
            round_num = round_offset + local_round
            yield sse_event("round_start", {"round": round_num, "maxRounds": round_offset + max_rounds})

            # Step 1: Generate or refine the prompt via DeepSeek
            yield sse_event("step", {"round": round_num, "step": "generating_prompt", "message": "正在生成/优化提示词…"})
            try:
                if local_round == 1 and not current_prompt:
                    messages = [
                        {"role": "system", "content": "你是一个提示词工程专家。用户会给出他们想要从会议转写文本中提取的信息或输出格式。你需要生成一个用于听悟（通义听悟）CustomPrompt 的提示词。\n\n要求：\n1. 提示词中必须包含 {Transcription} 占位符，听悟会自动将转写结果填入\n2. 提示词应该清晰、具体，指导模型输出用户期望的格式和内容\n3. 只输出提示词本身，不要有任何解释或前缀，不要向用户追问"},
                        {"role": "user", "content": f"我的需求：{effective_goal}"},
                    ]
                elif local_round == 1 and current_prompt:
                    user_parts = [
                        f"改进目标：{effective_goal}",
                        f"当前 CustomPrompt 全文：\n{current_prompt}",
                    ]
                    if last_result:
                        user_parts.append(f"上一轮听悟实际输出：\n{last_result[:4000]}")
                    messages = [
                        {"role": "system", "content": "你是一个提示词工程专家。用户会提供一份已有的听悟 CustomPrompt、改进目标，以及可能的上一轮实际输出。请直接基于现有提示词做针对性修改。\n\n要求：\n1. 提示词中必须包含 {Transcription} 占位符\n2. 保留原提示词里仍然有效的结构与约束，只改与目标/反馈相关的部分\n3. 只输出改进后的完整提示词本身，不要解释，不要向用户追问或索要更多材料"},
                        {"role": "user", "content": "\n\n".join(user_parts)},
                    ]
                else:
                    feedback_text = ""
                    for h in history:
                        feedback_text += f"\n--- 第 {h['round']} 轮 ---\n提示词：\n{h['prompt']}\n\n实际输出：\n{h['result']}\n"
                    messages = [
                        {"role": "system", "content": "你是一个提示词工程专家。用户正在迭代优化听悟（通义听悟）CustomPrompt 的提示词。下面是之前的尝试和结果。请基于用户的目标和之前的输出，生成改进后的提示词。\n\n要求：\n1. 提示词中必须包含 {Transcription} 占位符\n2. 分析之前输出与期望的差距，针对性优化\n3. 只输出新的提示词本身，不要有任何解释，不要向用户追问"},
                        {"role": "user", "content": f"我的目标：{effective_goal}\n\n之前的迭代记录：{feedback_text}"},
                    ]
                current_prompt = deepseek_chat(messages, deepseek_key)
                if "{Transcription}" not in current_prompt:
                    # 只补齐占位符，避免把“提示词正文”污染成额外说明文本
                    current_prompt = current_prompt.rstrip()
                    current_prompt += "\n\n{Transcription}"
            except Exception as e:
                yield sse_event("error", {"round": round_num, "message": f"DeepSeek 调用失败: {e}"})
                return

            yield sse_event("prompt_generated", {"round": round_num, "prompt": current_prompt})

            # Step 2: Submit rerun to Tingwu
            yield sse_event("step", {"round": round_num, "step": "submitting_rerun", "message": "正在提交听悟重跑任务…"})
            try:
                payload = {
                    "AppKey": app_key,
                    "Input": {"TaskId": source_task_id, "SourceLanguage": "cn"},
                    "Parameters": {
                        "CustomPromptEnabled": True,
                        "CustomPrompt": {
                            "Contents": [{
                                "Name": "summary",
                                "Prompt": current_prompt,
                                "Model": "tingwu-plus",
                                "TransType": "sentence-chat",
                            }]
                        },
                    },
                }
                result, error = api_call(lambda: tingwu_request(
                    "PUT", "/openapi/tingwu/v2/tasks", payload, {"type": "offline"}
                ))
                if error:
                    yield sse_event("error", {"round": round_num, "message": f"重跑提交失败: {error}"})
                    return
                rerun_task_id = (result.get("Data") or {}).get("TaskId", "")
                if not rerun_task_id:
                    yield sse_event("error", {"round": round_num, "message": "重跑返回无 TaskId"})
                    return
            except Exception as e:
                yield sse_event("error", {"round": round_num, "message": f"重跑异常: {e}"})
                return

            yield sse_event("rerun_submitted", {"round": round_num, "rerunTaskId": rerun_task_id})

            # Step 3: Poll for result
            yield sse_event("step", {"round": round_num, "step": "polling_result", "message": "等待听悟处理完成…"})
            actual_output, poll_error = poll_task_result(rerun_task_id, timeout=600)
            if poll_error:
                yield sse_event("error", {"round": round_num, "message": f"结果获取失败: {poll_error}"})
                return

            yield sse_event("result_ready", {"round": round_num, "result": actual_output})
            history.append({"round": round_num, "prompt": current_prompt, "result": actual_output})

            # Step 4: Evaluate
            yield sse_event("step", {"round": round_num, "step": "evaluating", "message": "正在评估输出质量…"})
            try:
                eval_messages = [
                    {"role": "system", "content": "你是一个输出质量评估专家。请评估听悟的实际输出是否满足用户的期望目标。\n\n请严格按以下 JSON 格式回复（不要有其他内容）：\n{\"score\": 1-10, \"satisfied\": true/false, \"analysis\": \"简要分析\"}"},
                    {"role": "user", "content": f"用户目标：{effective_goal}\n\n实际输出：\n{actual_output[:3000]}"},
                ]
                eval_raw = deepseek_chat(eval_messages, deepseek_key)
                eval_raw = eval_raw.strip()
                if eval_raw.startswith("```"):
                    eval_raw = eval_raw.split("\n", 1)[-1].rsplit("```", 1)[0]
                evaluation = json.loads(eval_raw)
            except Exception:
                evaluation = {"score": 5, "satisfied": False, "analysis": "评估解析失败，继续优化"}

            yield sse_event("evaluation", {
                "round": round_num,
                "score": evaluation.get("score", 0),
                "satisfied": evaluation.get("satisfied", False),
                "analysis": evaluation.get("analysis", ""),
            })

            if evaluation.get("satisfied"):
                yield sse_event("done", {
                    "round": round_num,
                    "message": "提示词优化完成！若不满意可继续追问修改。",
                    "finalPrompt": current_prompt,
                    "finalResult": actual_output,
                })
                return

        yield sse_event("done", {
            "round": round_offset + max_rounds,
            "message": f"已完成本次 {max_rounds} 轮迭代。若不满意可继续追问修改。",
            "finalPrompt": current_prompt,
            "finalResult": actual_output or "",
        })

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    # macOS 隔空播放常占用 5000，开发默认改用 5001
    app.run(debug=True, port=5001)
