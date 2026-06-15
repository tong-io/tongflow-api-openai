from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict
from urllib.request import Request, urlopen

from tongflow.node_slots import NodeSlots
from tongflow.slots import node_slot
from tongflow.models.gen_text import GenTextInput, GenTextOutput
from tongflow.models.split_text import SplitTextInput, SplitTextOutput
from tongflow.models.drop_video import DropVideoInput, DropVideoOutput
from tongflow.models.arrange_group import ArrangeGroupInput, ArrangeGroupOutput
from tongflow.llm_batch_handlers import arrange_group_output, drop_video_output


OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


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

    req = Request(
        f"{OPENAI_BASE_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    resp = urlopen(req, timeout=180)  # noqa: S310
    if getattr(resp, "status", 200) >= 400:
        raise RuntimeError(f"HTTP {resp.status}")

    body = resp.read().decode("utf-8", errors="replace")
    obj = json.loads(body)
    choices = obj.get("choices") or []
    if not choices:
        raise RuntimeError("OpenAI response missing choices")
    message = (choices[0] or {}).get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError("OpenAI response missing message.content")
    return content.strip()


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
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    model = (os.environ.get("OPENAI_CHAT_MODEL") or "").strip() or DEFAULT_MODEL
    user_message = (
        f"{input.userPrompt or ''}\n\n用户输入：{input.text}\n\n"
        "注意：除了明确的答案本身，不要生成任何其他多余内容。"
    )
    answer = _chat_openai(api_key=api_key, model=model, user_message=user_message)
    return GenTextOutput(success=True, text=answer)


@node_slot(NodeSlots.SPLIT_TEXT)
def split_text(input: SplitTextInput) -> SplitTextOutput:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    model = (os.environ.get("OPENAI_CHAT_MODEL") or "").strip() or DEFAULT_MODEL
    raw = _chat_openai(
        api_key=api_key,
        model=model,
        user_message=_build_split_user_message(input),
        json_mode=True,
    )
    try:
        texts = _parse_split_texts(raw)
    except (ValueError, json.JSONDecodeError) as e:
        return SplitTextOutput(success=False, error=str(e))
    return SplitTextOutput(success=True, texts=texts)


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
_SLOT_HANDLERS: Dict[str, Any] = {
    NodeSlots.GEN_TEXT: gen_text,
    NodeSlots.SPLIT_TEXT: split_text,
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
