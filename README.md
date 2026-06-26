# tongflow-api-openai

Official [TongFlow](https://github.com/tong-io/tongflow) plugin. Text **and image** generation via any [OpenAI](https://openai.com)-compatible endpoint — OpenAI itself, Azure OpenAI, a third-party relay, or a local server (vLLM / Ollama / LM Studio). Point it at an endpoint with `OPENAI_BASE_URL` and supply that endpoint's key and models.

## Capabilities

Implements these ABI slots (runs locally as a Python process, no GPU):

Text — via the Chat Completions API (`/chat/completions`):

- **Generate / rewrite text** (`gen-text`) — create or edit copy from a prompt.
- **Split long text** (`split-text`) — break a long passage into chunks.
- **Arrange & batch groups** (`arrange-group`) — group and arrange text/clip batches for downstream processing.
- **Filter or drop clips** (`drop-video`) — drop unwanted clips by rule.

Image — via the Images API (`/images/generations`, `/images/edits`), using `gpt-image-2` by default:

- **Generate image** (`image-gen`) — text → image.
- **Edit image** (`image-edit`) — one image + an instruction → edited image.
- **Fuse images** (`image-fusion`) — multiple reference images + a prompt → composed image.

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `OPENAI_API_KEY` | ✅ | Key for the target endpoint. For OpenAI itself, create one at [platform.openai.com/api-keys](https://platform.openai.com/api-keys). |
| `OPENAI_BASE_URL` | optional | OpenAI-compatible endpoint. Defaults to `https://api.openai.com/v1`; set it to an Azure / relay / local (vLLM·Ollama·LM Studio) base URL to switch providers. |
| `OPENAI_CHAT_MODEL` | optional | Chat model id for the text slots (default `gpt-4o-mini`). |
| `OPENAI_IMAGE_MODEL` | optional | Image model id for the image slots (default `gpt-image-2`; e.g. `gpt-image-1`, `dall-e-3`). |
| `OPENAI_IMAGE_SIZE` | optional | Force an output size for the image slots, e.g. `1024x1024` / `1536x1024` / `auto`. By default the node's width × height is used, or the model's default when unset. |

Values are stored locally and take effect without a restart.

> **`OPENAI_BASE_URL` must be the API root, not the full path.** The plugin appends `/chat/completions`, `/images/generations`, or `/images/edits` itself, so set the base URL up to the version segment (e.g. `https://api.openai.com/v1`) — do **not** include a method path. A trailing slash is fine; it is trimmed automatically.

> Image slots require an endpoint that serves the OpenAI **Images API**. Most relays and Azure expose it; local servers (Ollama / vLLM / LM Studio) typically do **not** — use them for the text slots only.

## Pointing at a different provider

Any service that implements the OpenAI Chat Completions API works. Set the three variables to match the provider:

| Provider | `OPENAI_BASE_URL` | `OPENAI_API_KEY` | `OPENAI_CHAT_MODEL` |
| --- | --- | --- | --- |
| OpenAI (default) | *(leave empty)* | your OpenAI key | e.g. `gpt-4o-mini` |
| Third-party relay / proxy | the relay's base URL, e.g. `https://your-relay.example.com/v1` | the relay's key | a model the relay exposes |
| Azure OpenAI | `https://<resource>.openai.azure.com/openai/v1` | your Azure key | your deployment name |
| Local — Ollama | `http://localhost:11434/v1` | any non-empty string | e.g. `llama3.1` |
| Local — vLLM / LM Studio | `http://localhost:8000/v1` (vLLM) · `http://localhost:1234/v1` (LM Studio) | any non-empty string | the served model id |

> Mixing an endpoint with a key issued for a *different* endpoint is the common failure: a relay key sent to `api.openai.com` returns **401 Unauthorized**. Make sure `OPENAI_API_KEY` belongs to whatever `OPENAI_BASE_URL` points at.

## Errors

Failures surface the endpoint that was actually called plus the upstream response body, e.g. `HTTP 401 from OpenAI (https://.../v1/chat/completions): {...}`. Use it to tell apart a wrong key (401), rate limiting / quota (429), and a wrong base URL (404 / connection error).
