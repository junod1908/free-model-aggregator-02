#!/usr/bin/env python3
"""free-model-aggregator-02: Live Gauge — the best live free-model aggregator.

One long-running server. It reads the trial brief (fourteen free OpenRouter
models with the house's sealed quality ratings, and a six-prompt workload),
registers one catalog document with the trial env, then answers the env
client's OpenAI-wire and Anthropic-wire requests — JSON and server-sent
events — by routing each one to the strongest free model that meets the
request's requirements, dialling it live on the grader gateway's Chat
Completions route (the only route these model ids have) and translating the
answer back into the client's wire. A gateway refusal (429/5xx) is never the
answer: the request retries once on the same model when it still out-rates
the next candidate, else fails over to the next candidate, and every refusal
that precedes a served answer is reported in the
x-makefun-upstream-failures header so the failover column can see it.

Every routing choice is a function of the brief, the requests and the
gateway's answers seen so far: no wall clock and no randomness anywhere, so
a replayed trial makes the same calls in the same order.

Run `python3 main.py --list` to print the catalog document and exit 0.
"""

import http.client
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

PROTOCOL_VERSION = 1
# The timestamp the providers stamp on their answers; translated answers
# reuse it so a response never carries a wall-clock reading.
FIXED_CREATED = 1767225600
ANTHROPIC_VERSION = "2023-06-01"
UPSTREAM_TIMEOUT = 8.0
POLL_SECONDS = 0.1

# Live egress: one gateway, one route, one global call budget per trial.
MAX_ATTEMPTS = 4
MAX_CALLS_PER_TRIAL = 60
MAX_TOKENS_DEFAULT = 512
MAX_TOKENS_CAP = 8192
MAX_TRAIL_ROWS = 8
MAX_TRAIL_BYTES = 8192
TOKEN_RE = re.compile(r"\b[A-Z]+-\d{4}\b")

GATEWAY_PROVIDER_ID = "openrouter"
GATEWAY_BASE_URL = "https://openrouter.ai/api/v1"
GATEWAY_KEY_ENV = "OPENROUTER_API_KEY"

# Stop reasons group into families; a translated answer reports the family's
# spelling on the client's wire.
STOP_FAMILY = {
    "stop": 0, "end_turn": 0, "STOP": 0,
    "length": 1, "max_tokens": 1, "MAX_TOKENS": 1,
    "tool_calls": 2, "function_call": 2, "tool_use": 2,
    "content_filter": 3, "refusal": 3, "SAFETY": 3,
}
STOP_ON_WIRE = {
    "openai": {0: "stop", 1: "length", 2: "tool_calls", 3: "content_filter"},
    "anthropic": {0: "end_turn", 1: "max_tokens", 2: "tool_use", 3: "refusal"},
}


def is_record(value):
    return isinstance(value, dict)


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def string_list(value):
    """A stop field as the list of strings it names (a lone string counts)."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def blocks_text(content):
    """The text a content value carries, the way every wire's parser reads it:
    a string is itself; blocks contribute their text parts joined by newlines."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return ""


# --------------------------------------------------------------------------
# The brief
# --------------------------------------------------------------------------

class Brief:
    """The two JSON documents embedded in the trial's prompt file."""

    def __init__(self, prompt_path):
        self.models = []           # model rows, in the brief's order
        self.model_by_id = {}
        self.manifest = []         # workload rows, in sending order
        self.error = None
        if not prompt_path:
            self.error = "MAKEFUN_PROMPT_FILE is not set"
            return
        try:
            with open(prompt_path, "r", encoding="utf-8") as handle:
                text = handle.read()
        except OSError as err:
            self.error = f"the brief cannot be read: {err}"
            return
        docs = self._embedded_documents(text)
        for doc in docs:
            schema = doc.get("schema", "")
            if schema.endswith("/models/v1"):
                for row in doc.get("models", []):
                    if is_record(row) and isinstance(row.get("id"), str):
                        self.models.append(row)
                        self.model_by_id[row["id"]] = row
            elif schema.endswith("/ratings/v1"):
                for row in doc.get("ratings", []):
                    if is_record(row) and isinstance(row.get("model"), str):
                        row = dict(row)
                        row["id"] = row["model"]
                        self.models.append(row)
                        self.model_by_id[row["id"]] = row
            elif schema.endswith("/prompts/v1"):
                prompts = doc.get("prompts")
                if isinstance(prompts, list):
                    self.manifest = [row for row in prompts if is_record(row)]
            elif schema.endswith("/workload/v1"):
                manifest = doc.get("manifest")
                if isinstance(manifest, list):
                    self.manifest = [row for row in manifest if is_record(row)]
        if not self.models:
            self.error = "the brief holds no free model list"

    @staticmethod
    def _embedded_documents(text):
        """Every JSON document that starts with a schema key, in file order."""
        decoder = json.JSONDecoder()
        docs = []
        at = 0
        while True:
            at = text.find('{"schema"', at)
            if at < 0:
                return docs
            try:
                doc, _end = decoder.raw_decode(text, at)
            except json.JSONDecodeError:
                at += 1
                continue
            if is_record(doc):
                docs.append(doc)
            at += 1


# --------------------------------------------------------------------------
# The catalog: exactly the fourteen free models, one provider
# --------------------------------------------------------------------------

def build_catalog(brief):
    """The one catalog document: what --list prints and what register sends.

    Models come from the brief's free model list, in the brief's order; the
    gateway's public endpoint is the base URL, never the trial gateway."""
    models = []
    seen = set()
    for row in brief.models:
        model_id = row.get("id")
        if not isinstance(model_id, str) or model_id in seen:
            continue
        if row.get("free") is not True:
            continue
        seen.add(model_id)
        context = row.get("context_length")
        models.append({
            "id": model_id,
            "free": True,
            "context_length": int(context) if is_number(context) else None,
        })
    if not models:
        return {"providers": []}
    return {"providers": [{
        "id": GATEWAY_PROVIDER_ID,
        "base_url": GATEWAY_BASE_URL,
        "key_env": GATEWAY_KEY_ENV,
        "wire": "openai",
        "models": models,
    }]}


def free_model_ids(catalog):
    """Every free model id in the catalog, once each, in catalog order."""
    ids = []
    seen = set()
    for provider in catalog["providers"]:
        for model in provider["models"]:
            if model.get("free") is True and model["id"] not in seen:
                seen.add(model["id"])
                ids.append(model["id"])
    return ids


# --------------------------------------------------------------------------
# Small HTTP client (env routes and the gateway route alike)
# --------------------------------------------------------------------------

def http_call(method, url, headers=None, payload=None, timeout=UPSTREAM_TIMEOUT):
    """One HTTP round trip; returns (status, body bytes). Raises on transport
    errors so the caller can treat an unreachable route as a failure."""
    parts = urlsplit(url)
    if parts.scheme == "http":
        conn = http.client.HTTPConnection(parts.hostname, parts.port or 80, timeout=timeout)
    elif parts.scheme == "https":
        conn = http.client.HTTPSConnection(parts.hostname, parts.port or 443, timeout=timeout)
    else:
        raise ValueError(f"not an http(s) URL the aggregator can dial: {url}")
    target = parts.path or "/"
    if parts.query:
        target = f"{target}?{parts.query}"
    body = None
    sent_headers = dict(headers or {})
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        sent_headers.setdefault("content-type", "application/json")
    try:
        conn.request(method, target, body=body, headers=sent_headers)
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Canonical requests: what a client body asks, read once, in wire-neutral
# terms, so the gateway's Chat Completions body can be built from it.
# --------------------------------------------------------------------------

def parse_image_url(url):
    """An image part from an OpenAI image_url: a data URI splits into its
    media type and base64 data; anything else stays a URL part."""
    if isinstance(url, str) and url.startswith("data:"):
        head, _, data = url[5:].partition(",")
        media = head.split(";")[0] or "image/png"
        return {"kind": "image", "media_type": media, "data": data}
    return {"kind": "image", "url": url if isinstance(url, str) else ""}


def content_parts(content, api):
    """A message's content as ordered parts: {"kind": "text", ...} and
    {"kind": "image", ...} entries, in the order the client sent them."""
    if isinstance(content, str):
        return [{"kind": "text", "text": content}] if content != "" else []
    if not isinstance(content, list):
        return []
    parts = []
    for block in content:
        if not is_record(block):
            continue
        kind = block.get("type")
        if kind == "text" and isinstance(block.get("text"), str):
            parts.append({"kind": "text", "text": block["text"]})
        elif api == "openai" and kind == "image_url" and is_record(block.get("image_url")):
            parts.append(parse_image_url(block["image_url"].get("url")))
        elif api == "anthropic" and kind == "image" and is_record(block.get("source")):
            source = block["source"]
            if source.get("type") == "base64" and isinstance(source.get("data"), str):
                parts.append({
                    "kind": "image",
                    "media_type": source.get("media_type") or "image/png",
                    "data": source["data"],
                })
            elif source.get("type") == "url" and isinstance(source.get("url"), str):
                parts.append({"kind": "image", "url": source["url"]})
    return parts


def canonical_from_openai(body):
    system_parts = []
    messages = []
    for message in body.get("messages", []):
        if not is_record(message):
            continue
        role = message.get("role")
        if role == "system":
            text = blocks_text(message.get("content"))
            if text != "":
                system_parts.append(text)
        elif role in ("user", "assistant"):
            messages.append({"role": role, "parts": content_parts(message.get("content"), "openai")})
    tools = []
    for tool in body.get("tools", []):
        if is_record(tool) and is_record(tool.get("function")) and isinstance(tool["function"].get("name"), str):
            fn = tool["function"]
            tools.append({
                "name": fn["name"],
                "description": fn.get("description"),
                "schema": fn.get("parameters"),
            })
    max_tokens = body.get("max_tokens")
    if not is_number(max_tokens):
        max_tokens = body.get("max_completion_tokens")
    fmt = body.get("response_format")
    json_mode = is_record(fmt) and fmt.get("type") in ("json_object", "json_schema")
    return {
        "system": "\n".join(system_parts),
        "messages": messages,
        "tools": tools,
        "max_tokens": int(max_tokens) if is_number(max_tokens) else None,
        "stops": string_list(body.get("stop")),
        "json_mode": json_mode,
        "response_format": fmt if json_mode else None,
        "tool_choice": body.get("tool_choice"),
        "temperature": body.get("temperature") if is_number(body.get("temperature")) else None,
        "stream": body.get("stream") is True,
    }


def canonical_from_anthropic(body):
    messages = []
    for message in body.get("messages", []):
        if not is_record(message):
            continue
        role = message.get("role")
        if role in ("user", "assistant"):
            messages.append({"role": role, "parts": content_parts(message.get("content"), "anthropic")})
    tools = []
    for tool in body.get("tools", []):
        if is_record(tool) and isinstance(tool.get("name"), str):
            tools.append({
                "name": tool["name"],
                "description": tool.get("description"),
                "schema": tool.get("input_schema"),
            })
    max_tokens = body.get("max_tokens")
    return {
        "system": blocks_text(body.get("system")),
        "messages": messages,
        "tools": tools,
        "max_tokens": int(max_tokens) if is_number(max_tokens) else None,
        "stops": string_list(body.get("stop_sequences")),
        "json_mode": False,
        "response_format": None,
        "tool_choice": None,
        "temperature": body.get("temperature") if is_number(body.get("temperature")) else None,
        "stream": body.get("stream") is True,
    }


def openai_content(parts):
    """Canonical parts as an OpenAI content value: a lone text part stays a
    plain string; anything else becomes blocks in the same order."""
    if len(parts) == 1 and parts[0]["kind"] == "text":
        return parts[0]["text"]
    blocks = []
    for part in parts:
        if part["kind"] == "text":
            blocks.append({"type": "text", "text": part["text"]})
        elif "data" in part:
            blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:{part['media_type']};base64,{part['data']}"},
            })
        else:
            blocks.append({"type": "image_url", "image_url": {"url": part.get("url", "")}})
    return blocks


def build_openai_request(canon, model):
    """The one upstream body this entry ever sends: Chat Completions, always
    non-streaming (the client's stream is synthesised locally instead)."""
    messages = []
    if canon["system"] != "":
        messages.append({"role": "system", "content": canon["system"]})
    for message in canon["messages"]:
        messages.append({"role": message["role"], "content": openai_content(message["parts"])})
    body = {"model": model, "messages": messages, "stream": False}
    if canon["tools"]:
        tools = []
        for tool in canon["tools"]:
            fn = {"name": tool["name"]}
            if isinstance(tool.get("description"), str):
                fn["description"] = tool["description"]
            if is_record(tool.get("schema")):
                fn["parameters"] = tool["schema"]
            tools.append({"type": "function", "function": fn})
        body["tools"] = tools
        if canon["tool_choice"] is not None:
            body["tool_choice"] = canon["tool_choice"]
    cap = canon["max_tokens"]
    if cap is None:
        cap = MAX_TOKENS_DEFAULT
    body["max_tokens"] = min(int(cap), MAX_TOKENS_CAP)
    if canon["stops"]:
        body["stop"] = list(canon["stops"])
    if canon["json_mode"] and canon["response_format"] is not None:
        body["response_format"] = canon["response_format"]
    if canon["temperature"] is not None:
        body["temperature"] = canon["temperature"]
    return body


def extract_answer(payload):
    """A Chat Completions answer as (text, input tokens, output tokens,
    stop, has_tool_calls): the answer's whole text, its usage counts as the
    provider reports them, and its stop reason in its own spelling."""
    choices = payload.get("choices") or []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get("message") if is_record(choice.get("message")) else {}
    content = message.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "".join(
            block["text"] for block in content
            if is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str)
        )
    else:
        text = ""
    usage = payload.get("usage") or {}
    tool_calls = message.get("tool_calls")
    return (
        text,
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        choice.get("finish_reason"),
        isinstance(tool_calls, list) and len(tool_calls) > 0,
    )


def refusal_code(raw, status):
    """The short code of a gateway/provider error body, for the trail."""
    try:
        doc = json.loads(raw.decode("utf-8"))
    except Exception:
        doc = None
    if is_record(doc) and is_record(doc.get("error")):
        for key in ("code", "type"):
            value = doc["error"].get(key)
            if isinstance(value, str) and value != "":
                return value
        message = doc["error"].get("message")
        if isinstance(message, str) and message != "":
            return message
    return f"http_{status}"


# --------------------------------------------------------------------------
# Client answers: a served (text, usage, stop) rendered in the client's
# wire, as one JSON document or as a server-sent event stream
# --------------------------------------------------------------------------

def split_text(text, size=13):
    pieces = [text[at:at + size] for at in range(0, len(text), size)]
    return pieces if pieces else [""]


def client_json(api, model, text, tokens_in, tokens_out, stop_reason, tag):
    family = STOP_FAMILY.get(stop_reason, 0)
    if api == "openai":
        return {
            "id": f"chatcmpl-{tag}",
            "object": "chat.completion",
            "created": FIXED_CREATED,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": STOP_ON_WIRE["openai"][family],
            }],
            "usage": {
                "prompt_tokens": tokens_in,
                "completion_tokens": tokens_out,
                "total_tokens": (tokens_in or 0) + (tokens_out or 0),
            },
        }
    return {
        "id": f"msg_{tag}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": STOP_ON_WIRE["anthropic"][family],
        "stop_sequence": None,
        "usage": {"input_tokens": tokens_in, "output_tokens": tokens_out},
    }


def sse_frame(event, data):
    head = f"event: {event}\n" if event is not None else ""
    return f"{head}data: {json.dumps(data)}\n\n"


def client_stream(api, model, text, tokens_in, tokens_out, stop_reason, tag):
    family = STOP_FAMILY.get(stop_reason, 0)
    frames = []
    if api == "openai":
        stop = STOP_ON_WIRE["openai"][family]

        def chunk(delta, finish):
            return {
                "id": f"chatcmpl-{tag}",
                "object": "chat.completion.chunk",
                "created": FIXED_CREATED,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }

        frames.append(sse_frame(None, chunk({"role": "assistant", "content": ""}, None)))
        for piece in split_text(text):
            frames.append(sse_frame(None, chunk({"content": piece}, None)))
        frames.append(sse_frame(None, chunk({}, stop)))
        frames.append(sse_frame(None, {
            "id": f"chatcmpl-{tag}",
            "object": "chat.completion.chunk",
            "created": FIXED_CREATED,
            "model": model,
            "choices": [],
            "usage": {
                "prompt_tokens": tokens_in,
                "completion_tokens": tokens_out,
                "total_tokens": (tokens_in or 0) + (tokens_out or 0),
            },
        }))
        frames.append("data: [DONE]\n\n")
        return "".join(frames)
    stop = STOP_ON_WIRE["anthropic"][family]
    frames.append(sse_frame("message_start", {
        "type": "message_start",
        "message": {
            "id": f"msg_{tag}",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": tokens_in, "output_tokens": 0},
        },
    }))
    frames.append(sse_frame("content_block_start", {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }))
    for piece in split_text(text):
        frames.append(sse_frame("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": piece},
        }))
    frames.append(sse_frame("content_block_stop", {"type": "content_block_stop", "index": 0}))
    frames.append(sse_frame("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop, "stop_sequence": None},
        "usage": {"output_tokens": tokens_out},
    }))
    frames.append(sse_frame("message_stop", {"type": "message_stop"}))
    return "".join(frames)


def client_error(api, status, code, message):
    if api == "anthropic":
        return {"type": "error", "error": {"type": code, "message": message}}
    return {"error": {"message": message, "type": code, "param": None, "code": None}}


def trail_header(trail):
    """The x-makefun-upstream-failures value for one prompt's refusals, or
    None when there is nothing to report. Rows are real HTTP refusals only:
    a transport failure has no HTTP status to report, so it is left out
    rather than invented (a row with a status outside 100-599 would make
    the env discard the whole trail)."""
    rows = list(trail[:MAX_TRAIL_ROWS])
    while rows:
        value = json.dumps(rows, separators=(",", ":"))
        if len(value.encode("utf-8")) <= MAX_TRAIL_BYTES:
            return value
        rows = rows[:-1]
    return None


# --------------------------------------------------------------------------
# The router: candidates by rating, one gateway, refusals failed over
# --------------------------------------------------------------------------

def meets_requirements(row, requires):
    if requires.get("tools") is True and row.get("tools") is not True:
        return False
    if requires.get("vision") is True and row.get("vision") is not True:
        return False
    if requires.get("json") is True and row.get("json") is not True:
        return False
    minimum = requires.get("min_context")
    if is_number(minimum):
        context = row.get("context_length")
        if not is_number(context) or context < minimum:
            return False
    return True


class Router:
    """Serves one client request per call, strictly in arrival order.

    State is only what the brief said and what the gateway answered: how
    many calls the trial has spent, and which models a 429 cooled down on
    the previous prompt (they drop one step in priority for the next one).
    Nothing here reads the wall clock, so a replay makes the same calls."""

    def __init__(self, brief, gateway):
        self.brief = brief
        self.gateway = gateway      # (base_url, api_key)
        self.calls = 0
        self.cooled = set()         # models 429'd on the previous prompt
        self.last_429 = set()
        self.order = {row["id"]: i for i, row in enumerate(brief.models)}

    # -- candidates ------------------------------------------------------

    def candidates(self, requires):
        """Every free model that may serve, best first: highest rating,
        then the brief's order, then the model id — with models cooled by
        a 429 on the previous prompt dropped one step behind the rest."""
        found = [
            row for row in self.brief.models
            if row.get("free") is True and meets_requirements(row, requires)
        ]
        found.sort(key=lambda r: (
            0 if r["id"] not in self.cooled else 1,
            -r.get("rating_milli", 0),
            self.order[r["id"]],
            r["id"],
        ))
        return found

    # -- one gateway call --------------------------------------------------

    def call_gateway(self, model, canon):
        """POST the canonical request to the gateway's Chat Completions
        route. Returns (ok, answer, refusal) where answer is
        extract_answer's tuple and refusal is (status, code)."""
        base, key = self.gateway
        url = f"{base}/chat/completions"
        payload = build_openai_request(canon, model)
        headers = {"Authorization": f"Bearer {key}"}
        self.calls += 1
        try:
            status, raw = http_call("POST", url, headers=headers, payload=payload)
        except Exception:
            return False, None, (0, "connection_error")
        if 200 <= status < 300:
            try:
                doc = json.loads(raw.decode("utf-8"))
            except Exception:
                return False, None, (status, "invalid_json")
            return True, extract_answer(doc), None
        return False, None, (status, refusal_code(raw, status))

    # -- serving -----------------------------------------------------------

    def serve(self, canon, requires, token):
        """Answer one canonical request. Returns ((model, answer), trail)
        or (None, trail) when no free model could be made to answer.

        A 2xx answer only counts when its text carries the prompt's token
        and it made no tool call: a success that cannot prove the prompt
        locks in a failure, so it fails over like a refusal (without a
        trail row — it was not refused)."""
        queue = self.candidates(requires)
        tried = set()
        retried = set()  # models already given their one retry, this request
        trail = []
        attempts = 0
        while attempts < MAX_ATTEMPTS and self.calls < MAX_CALLS_PER_TRIAL:
            pick = None
            for row in queue:
                if row["id"] not in tried:
                    pick = row
                    break
            if pick is None:
                return None, trail
            model = pick["id"]
            rating = pick.get("rating_milli", 0)
            ok, answer, refusal = self.call_gateway(model, canon)
            attempts += 1
            if ok:
                if self._usable(answer, token, queue, tried):
                    # A refusal this model recovered from in the same
                    # prompt was transient: do not cool it for the next
                    # prompt — it just proved it can serve.
                    self.last_429.discard(model)
                    return (model, answer), trail
                tried.add(model)
                continue
            status, code = refusal
            if status != 0:
                trail.append({"status": status, "code": code[:64], "model": model})
            if status == 429:
                self.last_429.add(model)
            if status == 401:
                return None, trail  # the key is dead; every model would agree
            tried.add(model)
            if status == 429 or status >= 500 or status == 0:
                # A refusal on the provider's own schedule: its very next
                # call may well answer. When this pick still beats whatever
                # would serve instead, one immediate retry buys the rating
                # back for one extra call.
                if model not in retried and attempts < MAX_ATTEMPTS and self.calls < MAX_CALLS_PER_TRIAL:
                    nxt = None
                    for row in queue:
                        if row["id"] not in tried:
                            nxt = row
                            break
                    if nxt is None or rating > nxt.get("rating_milli", 0):
                        retried.add(model)
                        ok, answer, refusal = self.call_gateway(model, canon)
                        attempts += 1
                        if ok:
                            if self._usable(answer, token, queue, tried):
                                self.last_429.discard(model)
                                return (model, answer), trail
                            tried.add(model)
                            continue
                        status, code = refusal
                        if status != 0:
                            trail.append({"status": status, "code": code[:64], "model": model})
                        if status == 429:
                            self.last_429.add(model)
                        if status == 401:
                            return None, trail
                        tried.add(model)
                continue
            # A 400-class answer means our call itself was refused as
            # invalid; do not spend more calls the same way on this model.
        return None, trail

    def _usable(self, answer, token, queue, tried):
        """A 2xx answer may serve the prompt: its text carries the token,
        and it made no tool call while an untried candidate remains."""
        text, _tin, _tout, _stop, has_tools = answer
        if token is not None and token not in text:
            return False
        if has_tools:
            for row in queue:
                if row["id"] not in tried:
                    return False
        return True

    def end_prompt(self):
        """Roll the 429 set forward: what cooled down on this prompt drops
        one step on the next one, and only for that one."""
        self.cooled = self.last_429
        self.last_429 = set()


# --------------------------------------------------------------------------
# The HTTP server
# --------------------------------------------------------------------------

class Aggregator:
    def __init__(self, env_url, brief, catalog, gateway):
        self.brief = brief
        self.catalog = catalog
        self.gateway = gateway
        self.router = Router(brief, gateway)
        self.models = free_model_ids(catalog)
        self.index = 0               # client requests answered so far
        self.lock = threading.Lock() # the client sends one at a time; keep it so

    def requirements_for(self, index, api, body):
        """The manifest row for the request now arriving (they arrive in
        manifest order), else requirements read off the body itself."""
        if index < len(self.brief.manifest):
            return self.brief.manifest[index].get("requires") or {}
        requires = {}
        messages = body.get("messages") or []
        vision = False
        for message in messages:
            if is_record(message) and isinstance(message.get("content"), list):
                for block in message["content"]:
                    if is_record(block) and block.get("type") in ("image", "image_url"):
                        vision = True
        if isinstance(body.get("tools"), list) and body["tools"]:
            requires["tools"] = True
        if vision:
            requires["vision"] = True
        fmt = body.get("response_format")
        if api == "openai" and is_record(fmt) and fmt.get("type") in ("json_object", "json_schema"):
            requires["json"] = True
        return requires

    def prompt_token(self, canon):
        """The prompt's unique token, read off the request itself."""
        text = canon["system"]
        for message in canon["messages"]:
            for part in message["parts"]:
                if part["kind"] == "text":
                    text += "\n" + part["text"]
        match = TOKEN_RE.search(text)
        return match.group(0) if match else None

    def handle_models(self, anthropic_shape):
        if anthropic_shape:
            return {
                "data": [{
                    "type": "model",
                    "id": model,
                    "display_name": model,
                    "created_at": "2026-01-01T00:00:00Z",
                    "free": True,
                } for model in self.models],
                "has_more": False,
                "first_id": self.models[0] if self.models else None,
                "last_id": self.models[-1] if self.models else None,
            }
        return {
            "object": "list",
            "data": [{
                "id": model,
                "object": "model",
                "created": FIXED_CREATED,
                "owned_by": "free-model-aggregator",
                "free": True,
            } for model in self.models],
        }

    def handle_inference(self, api, body):
        """Serve one POST in the client's wire; returns (status, kind,
        payload, extra_headers) where kind is "json" or "sse"."""
        with self.lock:
            index = self.index
            self.index += 1
            if api == "openai":
                canon = canonical_from_openai(body)
            else:
                canon = canonical_from_anthropic(body)
            requires = self.requirements_for(index, api, body)
            token = self.prompt_token(canon)
            tag = f"r{index + 1:02d}"
            try:
                served, trail = self.router.serve(canon, requires, token)
            finally:
                self.router.end_prompt()
            if served is None:
                return 503, "json", client_error(
                    api, 503, "overloaded_error" if api == "anthropic" else "server_error",
                    "no free model could be made to answer this request",
                ), {}
            model, (text, tokens_in, tokens_out, stop, _tools) = served
            tokens_in = int(tokens_in) if is_number(tokens_in) else 0
            if tokens_in < 0:
                tokens_in = 0
            if not is_number(tokens_out) or int(tokens_out) < 1:
                tokens_out = max(1, (len(text) + 3) // 4)
            else:
                tokens_out = int(tokens_out)
            if not isinstance(stop, str) or stop not in STOP_FAMILY:
                stop = "stop"
            headers = {}
            value = trail_header(trail)
            if value is not None:
                headers["x-makefun-upstream-failures"] = value
            if canon["stream"]:
                return 200, "sse", client_stream(api, model, text, tokens_in, tokens_out, stop, tag), headers
            return 200, "json", client_json(api, model, text, tokens_in, tokens_out, stop, tag), headers


def make_handler(aggregator):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # keep the trial's stderr clean
            pass

        def _send_json(self, status, doc, extra=None):
            raw = json.dumps(doc).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(raw)))
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(raw)

        def _send_sse(self, status, text, extra=None):
            raw = text.encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "text/event-stream; charset=utf-8")
            self.send_header("cache-control", "no-cache")
            self.send_header("connection", "close")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(raw)
            self.wfile.flush()
            self.close_connection = True

        def do_GET(self):
            if self.path.split("?")[0] == "/v1/models":
                anthropic_shape = "x-api-key" in self.headers
                self._send_json(200, aggregator.handle_models(anthropic_shape))
                return
            self._send_json(404, client_error("openai", 404, "not_found", f"no route GET {self.path}"))

        def do_POST(self):
            path = self.path.split("?")[0]
            if path not in ("/v1/chat/completions", "/v1/messages"):
                self._send_json(404, client_error("openai", 404, "not_found", f"no route POST {path}"))
                return
            api = "openai" if path == "/v1/chat/completions" else "anthropic"
            try:
                length = int(self.headers.get("content-length") or 0)
            except ValueError:
                length = 0
            raw = self.rfile.read(length) if length > 0 else b""
            try:
                body = json.loads(raw.decode("utf-8"))
                if not is_record(body):
                    raise ValueError("not an object")
            except Exception:
                self._send_json(400, client_error(api, 400, "invalid_request_error",
                                                  "the request body is not one JSON object"))
                return
            status, kind, payload, extra = aggregator.handle_inference(api, body)
            if kind == "sse":
                self._send_sse(status, payload, extra)
            else:
                self._send_json(status, payload, extra)

    return Handler


# --------------------------------------------------------------------------
# The trial flow
# --------------------------------------------------------------------------

def log(message):
    sys.stderr.write(f"aggregator: {message}\n")
    sys.stderr.flush()


def gateway_endpoint():
    """The gateway's Chat Completions route, from the trial's own env:
    OPENAI_BASE_URL (already ending in /v1 in the sandbox; /v1 is added
    when it is missing) and the trial's dummy key in OPENAI_API_KEY. The
    key is used as-is and never logged."""
    base = (os.environ.get("OPENAI_BASE_URL") or "").rstrip("/")
    if base and not base.endswith("/v1"):
        base = f"{base}/v1"
    return base, os.environ.get("OPENAI_API_KEY") or ""


def register(env_url, port, catalog):
    status, raw = http_call(
        "POST", f"{env_url}/register",
        payload={"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": catalog},
        timeout=5.0,
    )
    if status != 200:
        raise RuntimeError(f"register answered HTTP {status}: {raw[:200]!r}")
    doc = json.loads(raw.decode("utf-8"))
    if doc.get("ok") is not True:
        raise RuntimeError(f"register did not answer ok: {doc}")
    return doc.get("requests")


def poll_until_done(env_url, server):
    while True:
        try:
            status, raw = http_call("GET", f"{env_url}/work", timeout=2.0)
            if status == 200:
                doc = json.loads(raw.decode("utf-8"))
                if doc.get("done") is True:
                    return
        except Exception:
            pass  # the env is not reachable yet; try again on the next tick
        time.sleep(POLL_SECONDS)


def main(argv):
    env_url = (os.environ.get("MAKEFUN_ENV_URL") or "").rstrip("/")
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE") or ""
    brief = Brief(prompt_file)
    if brief.error is not None:
        log(brief.error)
    catalog = build_catalog(brief) if brief.models else {"providers": []}

    if "--list" in argv:
        sys.stdout.write(json.dumps(catalog) + "\n")
        sys.stdout.flush()
        return 0

    if not env_url:
        log("MAKEFUN_ENV_URL is not set; there is no trial to serve")
        return 1

    gateway = gateway_endpoint()
    aggregator = Aggregator(env_url, brief, catalog, gateway)
    server = None
    for _attempt in range(20):
        server = ThreadingHTTPServer(("0.0.0.0", 0), make_handler(aggregator))
        if server.server_address[1] != 8080:
            break
        server.server_close()
        server = None
    if server is None:
        log("no usable port could be bound")
        return 1
    port = server.server_address[1]
    log(f"listening on 0.0.0.0:{port} with {len(aggregator.models)} free models")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        requests = register(env_url, port, catalog)
        log(f"registered; the workload holds {requests} requests")
        poll_until_done(env_url, server)
    finally:
        server.shutdown()
    log("the env says the workload is done; exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
