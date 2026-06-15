# tongflow-api-openai

Official TongFlow plugin. Text generation via the [OpenAI](https://openai.com) chat API.

## Capabilities

Implements these ABI slots (runs locally as a Python process, no GPU):

- **Generate / rewrite text** (`gen-text`) — create or edit copy from a prompt.
- **Split long text** (`split-text`) — break a long passage into chunks.
- **Arrange & batch groups** (`arrange-group`) — group and arrange text/clip batches for downstream processing.
- **Filter or drop clips** (`drop-video`) — drop unwanted clips by rule.

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `OPENAI_API_KEY` | ✅ | Create one at [platform.openai.com/api-keys](https://platform.openai.com/api-keys). |
| `OPENAI_CHAT_MODEL` | optional | Override the default chat model (e.g. `gpt-4o-mini`). |

Values are stored locally and take effect without a restart.
