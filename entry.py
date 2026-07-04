from __future__ import annotations

import base64
import json
import os
import sys
from typing import Any, Dict, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tongflow.node_slots import NodeSlots
from tongflow.slots import node_slot
from tongflow.protocol import asset, prompt_media_to_bytes
from tongflow.models.gen_text import GenTextInput, GenTextOutput
from tongflow.models.audio_describe import (
    AudioDescribeInput,
    AudioDescribeOutput,
)
from tongflow.models.split_text import SplitTextInput, SplitTextOutput
from tongflow.models.image_gen import ImageGenInput, ImageGenOutput
from tongflow.models.image_edit import ImageEditInput, ImageEditOutput
from tongflow.models.image_fusion import ImageFusionInput, ImageFusionOutput
from tongflow.models.drop_video import DropVideoInput, DropVideoOutput
from tongflow.models.arrange_group import ArrangeGroupInput, ArrangeGroupOutput
from tongflow.llm_batch_handlers import arrange_group_output, drop_video_output


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_IMAGE_MODEL = "gpt-image-2"
# Chat-completions audio-in model (gpt-audio is the GA successor of
# gpt-4o-audio-preview); text-only chat models reject input_audio parts.
DEFAULT_AUDIO_MODEL = "gpt-audio"


def _resolve_base_url() -> str:
    # Any OpenAI-compatible endpoint (official, Azure, a relay, or a local
    # server like vLLM / Ollama / LM Studio) can be selected via env.
    return (
        os.environ.get("OPENAI_BASE_URL") or ""
    ).strip().rstrip("/") or DEFAULT_BASE_URL


def _require_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return api_key


def _resolve_chat_model() -> str:
    return (os.environ.get("OPENAI_CHAT_MODEL") or "").strip() or DEFAULT_MODEL


def _resolve_audio_model() -> str:
    return (os.environ.get("OPENAI_AUDIO_MODEL") or "").strip() or DEFAULT_AUDIO_MODEL


def _resolve_image_model() -> str:
    return (os.environ.get("OPENAI_IMAGE_MODEL") or "").strip() or DEFAULT_IMAGE_MODEL


def _resolve_size(width: int | None, height: int | None) -> str | None:
    # OpenAI image models accept a `WxH` string (or "auto"). An explicit
    # override wins; otherwise derive from the ABI width/height when both given.
    override = (os.environ.get("OPENAI_IMAGE_SIZE") or "").strip()
    if override:
        return override
    if width and height:
        return f"{width}x{height}"
    return None


def _request(
    url: str, *, data: bytes, headers: Dict[str, str], timeout: int = 180
) -> bytes:
    req = Request(url, data=data, headers=headers, method="POST")
    try:
        resp = urlopen(req, timeout=timeout)  # noqa: S310
    except HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError(
            f"HTTP {e.code} from OpenAI ({url}): {err_body or e.reason}"
        ) from e
    except URLError as e:
        raise RuntimeError(f"Network error contacting {url}: {e.reason}") from e
    return resp.read()


def _chat_openai(
    *, api_key: str, model: str, user_message: str, json_mode: bool = False
) -> str:
    headers: Dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "temperature": 1,
        "messages": [
            {
                "role": "system",
                "content": "你是一个根据用户要求进行文本生成的万能助手，请严格按照用户的要求进行文本生成。",
            },
            {"role": "user", "content": user_message},
        ],
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    body = _request(
        f"{_resolve_base_url()}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    ).decode("utf-8", errors="replace")
    obj = json.loads(body)
    choices = obj.get("choices") or []
    if not choices:
        raise RuntimeError("OpenAI response missing choices")
    message = (choices[0] or {}).get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError("OpenAI response missing message.content")
    return content.strip()


def _sniff_image(data: bytes) -> Tuple[str, str]:
    # Declare a sensible content-type/extension for multipart upload parts.
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png", "png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg", "jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp", "webp"
    return "image/png", "png"


def _multipart(
    fields: Dict[str, str], files: List[Tuple[str, str, str, bytes]]
) -> Tuple[bytes, str]:
    boundary = "----tongflow" + os.urandom(16).hex()
    line = boundary.encode()
    parts: List[bytes] = []
    for name, value in fields.items():
        parts.append(b"--" + line)
        parts.append(f'Content-Disposition: form-data; name="{name}"'.encode())
        parts.append(b"")
        parts.append(value.encode("utf-8"))
    for field, filename, mime, content in files:
        parts.append(b"--" + line)
        parts.append(
            f'Content-Disposition: form-data; name="{field}"; filename="{filename}"'.encode()
        )
        parts.append(f"Content-Type: {mime}".encode())
        parts.append(b"")
        parts.append(content)
    parts.append(b"--" + line + b"--")
    parts.append(b"")
    return b"\r\n".join(parts), f"multipart/form-data; boundary={boundary}"


def _download(url: str) -> Tuple[bytes, str]:
    try:
        resp = urlopen(url, timeout=180)  # noqa: S310
    except HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} downloading image from {url}") from e
    except URLError as e:
        raise RuntimeError(f"Network error downloading image: {e.reason}") from e
    mime = resp.headers.get_content_type() or "image/png"
    return resp.read(), mime


def _asset_from_images_response(body: bytes):
    # The OpenAI Images API returns data[0].b64_json (GPT image models) or, on
    # some endpoints / DALL·E, data[0].url. Support both.
    obj = json.loads(body.decode("utf-8", errors="replace"))
    items = obj.get("data") or []
    if not items:
        raise RuntimeError("image API response missing 'data'")
    item = items[0] or {}
    b64 = item.get("b64_json")
    if isinstance(b64, str) and b64:
        return asset(base64.b64decode(b64), mime="image/png")
    url = item.get("url")
    if isinstance(url, str) and url:
        raw, mime = _download(url)
        return asset(raw, mime=mime)
    raise RuntimeError("image API response missing b64_json/url")


def _generate_image(*, api_key: str, model: str, prompt: str, size: str | None):
    payload: Dict[str, Any] = {"model": model, "prompt": prompt, "n": 1}
    if size:
        payload["size"] = size
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = _request(
        f"{_resolve_base_url()}/images/generations",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        timeout=300,
    )
    return _asset_from_images_response(body)


def _edit_image(
    *, api_key: str, model: str, prompt: str, images: List[bytes], size: str | None
):
    fields: Dict[str, str] = {"model": model, "prompt": prompt, "n": "1"}
    if size:
        fields["size"] = size
    # A single image uses the `image` part; multiple reference images use the
    # repeated `image[]` part (GPT image models accept an array).
    field_name = "image" if len(images) == 1 else "image[]"
    files: List[Tuple[str, str, str, bytes]] = []
    for i, blob in enumerate(images):
        mime, ext = _sniff_image(blob)
        files.append((field_name, f"image_{i}.{ext}", mime, blob))
    data, content_type = _multipart(fields, files)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": content_type}
    body = _request(
        f"{_resolve_base_url()}/images/edits",
        data=data,
        headers=headers,
        timeout=300,
    )
    return _asset_from_images_response(body)


def _parse_split_texts(raw: str) -> list[str]:
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if "\n" in s:
            _, _, s = s.partition("\n")
        s = s.strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    obj = json.loads(s)
    if isinstance(obj, list):
        items = obj
    elif isinstance(obj, dict):
        raw_items = obj.get("texts")
        items = raw_items if isinstance(raw_items, list) else None
    else:
        items = None
    if items is None or not all(isinstance(x, str) for x in items):
        raise ValueError("LLM did not return a JSON array of strings")
    cleaned = [x.strip() for x in items if x.strip()]
    if not cleaned:
        raise ValueError("LLM returned an empty split")
    return cleaned


def _build_split_user_message(input: SplitTextInput) -> str:
    instruction = (input.userPrompt or "").strip() or "Split into natural, coherent segments."
    return (
        f"Split the following text into multiple segments according to this instruction:\n"
        f"{instruction}\n\n"
        f'Return ONLY a JSON object of the form {{"texts": ["segment 1", "segment 2", ...]}} '
        f"— no prose, no markdown, no code fences. "
        f"Each array element is one segment. Preserve the original wording; do not summarize.\n\n"
        f"TEXT:\n{input.text}"
    )


@node_slot(NodeSlots.GEN_TEXT)
def gen_text(input: GenTextInput) -> GenTextOutput:
    user_message = (
        f"{input.userPrompt or ''}\n\n用户输入：{input.text}\n\n"
        "注意：除了明确的答案本身，不要生成任何其他多余内容。"
    )
    answer = _chat_openai(
        api_key=_require_api_key(),
        model=_resolve_chat_model(),
        user_message=user_message,
    )
    return GenTextOutput(success=True, text=answer)


@node_slot(NodeSlots.SPLIT_TEXT)
def split_text(input: SplitTextInput) -> SplitTextOutput:
    raw = _chat_openai(
        api_key=_require_api_key(),
        model=_resolve_chat_model(),
        user_message=_build_split_user_message(input),
        json_mode=True,
    )
    try:
        texts = _parse_split_texts(raw)
    except (ValueError, json.JSONDecodeError) as e:
        return SplitTextOutput(success=False, error=str(e))
    return SplitTextOutput(success=True, texts=texts)


@node_slot(NodeSlots.IMAGE_GEN)
def image_gen(input: ImageGenInput) -> ImageGenOutput:
    prompt = (input.text or "").strip()
    if not prompt:
        return ImageGenOutput(success=False, error="image-gen requires a text prompt")
    image = _generate_image(
        api_key=_require_api_key(),
        model=_resolve_image_model(),
        prompt=prompt,
        size=_resolve_size(input.width, input.height),
    )
    return ImageGenOutput(success=True, image=image)


@node_slot(NodeSlots.IMAGE_EDIT)
def image_edit(input: ImageEditInput) -> ImageEditOutput:
    image = _edit_image(
        api_key=_require_api_key(),
        model=_resolve_image_model(),
        prompt=input.text,
        images=[prompt_media_to_bytes(input.image)],
        size=_resolve_size(input.width, input.height),
    )
    return ImageEditOutput(success=True, image=image)


@node_slot(NodeSlots.IMAGE_FUSION)
def image_fusion(input: ImageFusionInput) -> ImageFusionOutput:
    images = [prompt_media_to_bytes(x) for x in (input.images or [])]
    if not images:
        return ImageFusionOutput(
            success=False, error="image-fusion requires at least one input image"
        )
    image = _edit_image(
        api_key=_require_api_key(),
        model=_resolve_image_model(),
        prompt=input.text,
        images=images,
        size=_resolve_size(input.width, input.height),
    )
    return ImageFusionOutput(success=True, image=image)


@node_slot(NodeSlots.DROP_VIDEO)
def drop_video(input: DropVideoInput) -> DropVideoOutput:
    # Deterministic LLM-runner helper operating on the wire dict shape.
    result = drop_video_output(input.model_dump())
    return DropVideoOutput.model_construct(**result)


@node_slot(NodeSlots.ARRANGE_GROUP)
def arrange_group(input: ArrangeGroupInput) -> ArrangeGroupOutput:
    result = arrange_group_output(input.model_dump())
    return ArrangeGroupOutput.model_construct(**result)


# Runtime dispatcher. The @node_slot wrapper accepts a raw dict at this
# I/O boundary (it deep-constructs the typed BaseModel internally) and dumps
# the BaseModel return to a dict. `Any` reflects the boundary, not the
# plugin-facing contract above.
# input_audio accepts only short format tokens; wav/mp3 are the documented set.
_AUDIO_FORMATS = {"audio/mpeg": "mp3", "audio/mp3": "mp3", "audio/x-wav": "wav"}


@node_slot(NodeSlots.AUDIO_DESCRIBE)
def audio_describe(input: AudioDescribeInput) -> AudioDescribeOutput:
    instruction = (
        (input.userPrompt or "").strip()
        or (input.text or "").strip()
        or "Describe the clip in detail: whether it is music, speech, or "
        "ambient sound; genre or content; mood; instruments or voice "
        "characteristics; notable events."
    )
    mime = (input.audio.mime or "audio/wav").strip().lower()
    fmt = _AUDIO_FORMATS.get(mime) or mime.removeprefix("audio/") or "wav"
    # gpt-audio is tuned for voice chat: a user turn that pairs text with
    # input_audio makes it treat the clip as "not yet played" and it answers
    # with filler instead of listening. Putting the task in the system role
    # and sending the audio as the sole user content is the reliable shape.
    payload: Dict[str, Any] = {
        "model": _resolve_audio_model(),
        "modalities": ["text"],
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an audio analysis assistant. The user's message "
                    "IS an audio clip. Do not reply conversationally and never "
                    "ask for the audio; it is already provided. "
                    f"Task: {instruction}"
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": input.audio.bytesBase64,
                            "format": fmt,
                        },
                    },
                ],
            },
        ],
    }
    body = _request(
        f"{_resolve_base_url()}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {_require_api_key()}",
            "Content-Type": "application/json",
        },
    ).decode("utf-8", errors="replace")
    obj = json.loads(body)
    choices = obj.get("choices") or []
    if not choices:
        raise RuntimeError("OpenAI response missing choices")
    content = ((choices[0] or {}).get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("OpenAI response missing message.content")
    return AudioDescribeOutput(success=True, text=content.strip())


_SLOT_HANDLERS: Dict[str, Any] = {
    NodeSlots.GEN_TEXT: gen_text,
    NodeSlots.AUDIO_DESCRIBE: audio_describe,
    NodeSlots.SPLIT_TEXT: split_text,
    NodeSlots.IMAGE_GEN: image_gen,
    NodeSlots.IMAGE_EDIT: image_edit,
    NodeSlots.IMAGE_FUSION: image_fusion,
    NodeSlots.DROP_VIDEO: drop_video,
    NodeSlots.ARRANGE_GROUP: arrange_group,
}


def _write(out: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(out, ensure_ascii=False))
    sys.stdout.flush()


def main() -> int:
    try:
        raw = sys.stdin.read()
        req = json.loads(raw) if raw.strip() else {}
        prompt = req.get("prompt") if isinstance(req, dict) else {}
        if not isinstance(prompt, dict):
            prompt = {}
        slot = str(req.get("nodeSlot") or "") if isinstance(req, dict) else ""

        handler = _SLOT_HANDLERS.get(slot)
        if handler is None:
            raise RuntimeError(f"unsupported nodeSlot: {slot!r}")
        out = handler(prompt)
    except Exception as e:  # noqa: BLE001 — surfaced as ABI failure
        _write({"success": False, "error": str(e)})
        return 1

    _write(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
