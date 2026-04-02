import asyncio
import json
import logging
import os
import traceback
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import httpx
import orjson
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer
import re
from client import ProviderError, ProviderStream, open_provider_stream


import re
import json
from typing import Iterable
import uuid

logger = logging.getLogger(__name__)

ROUTER_URL = os.environ.get("ROUTER_URL", "http://127.0.0.1:8000")
MODEL = os.environ.get("MODEL", "GLM-5")
PORT = int(os.environ.get("PORT", "8890"))
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH")
API_KEY = os.environ.get("API_KEY", "")
DEFAULT_MAX_TOKENS = 16384
# Tool parsing regex patterns
FN_REGEX_PATTERN = r"<tool_call>\s*([^<\s]+)\s*(.*?)</tool_call>"
FN_PARAM_REGEX_PATTERN = r"<arg_key>([^<]+)</arg_key>\s*<arg_value>(.*?)</arg_value>"


def _extract_and_validate_params(matching_tool: dict, param_matches: Iterable[re.Match], fn_name: str) -> dict:
    params = {}
    # Parse and validate parameters
    required_params = set()
    if "parameters" in matching_tool and "required" in matching_tool["parameters"]:
        required_params = set(matching_tool["parameters"].get("required", []))

    allowed_params = set()
    if "parameters" in matching_tool and "properties" in matching_tool["parameters"]:
        allowed_params = set(matching_tool["parameters"]["properties"].keys())

    param_name_to_type = {}
    if "parameters" in matching_tool and "properties" in matching_tool["parameters"]:
        param_name_to_type = {
            name: val.get("type", "string") for name, val in matching_tool["parameters"]["properties"].items()
        }

    # Collect parameters
    found_params = set()
    for param_match in param_matches:
        param_name = param_match.group(1)
        param_value = param_match.group(2).strip()

        # Validate parameter is allowed
        if allowed_params and param_name not in allowed_params:
            raise ValueError(
                f"Parameter '{param_name}' is not allowed for function '{fn_name}'. "
                f"Allowed parameters: {allowed_params}"
            )

        # Validate and convert parameter type
        # supported: string, integer, array
        if param_name in param_name_to_type:
            if param_name_to_type[param_name] == "integer":
                try:
                    param_value = int(param_value)
                except ValueError:
                    raise ValueError(f"Parameter '{param_name}' is expected to be an integer.")
            elif param_name_to_type[param_name] == "array":
                try:
                    param_value = json.loads(param_value)
                except json.JSONDecodeError:
                    raise ValueError(f"Parameter '{param_name}' is expected to be an array.")
            else:
                # string
                pass

        params[param_name] = param_value
        found_params.add(param_name)

    # Check all required parameters are present
    missing_params = required_params - found_params
    if missing_params:
        raise ValueError(f"Missing required parameters for function '{fn_name}': {missing_params}")
    return params


def parse_message_local(completion_text, model_type, tools):
    assert model_type in ["glm", "glm45", "glm47"]
    text = completion_text.strip()
    reasoning_content = ""
    content = ""
    tool_calls = []

    # reasoning_content
    if "</think>" in text:
        reasoning_content, text = text.rsplit("</think>", 1)
        reasoning_content = reasoning_content.removeprefix("<think>")
        text = text.strip()
    elif "<tool_call>" in text:
        reasoning_content = text.split("<tool_call>")[0].strip()
        text = "<tool_call>" + text.split("<tool_call>")[1].strip()
    else:
        reasoning_content = text.removeprefix("<think>").strip()
        text = ""

    tool_call_text = text
    if "<tool_call>" in text:
        index = text.find("<tool_call>")
        content = text[:index].strip()
        text = text[index:].strip()
    else:
        content = text.strip()
        text = ""

    # tool_calls
    fn_match = re.findall(FN_REGEX_PATTERN, text, re.DOTALL)
    if fn_match:
        fn_name = fn_match[0][0].strip()
        fn_body = fn_match[0][1].strip()
    else:
        fn_name = None
        fn_body = None

    if not fn_name:
        return {"role": "assistant", "content": content, "reasoning_content": reasoning_content, "tool_calls": []}

    matching_tool = next(
        (tool for tool in tools if tool["name"] == fn_name),
        None,
    )

    if not matching_tool:
        return {
            "role": "assistant",
            "content": tool_call_text,
            "reasoning_content": reasoning_content,
            "tool_calls": [],
        }

    param_matches = re.finditer(FN_PARAM_REGEX_PATTERN, fn_body, re.DOTALL)
    try:
        params = _extract_and_validate_params(matching_tool, param_matches, fn_name)
    except ValueError as e:
        return {
            "role": "assistant",
            "content": tool_call_text,
            "reasoning_content": reasoning_content,
            "tool_calls": [],
        }
    tool_calls.append(
        {
            "tool_call_id": "tool-call-" + str(uuid.uuid4()),
            "name": fn_name,
            "arguments": json.dumps(params, ensure_ascii=False),
        }
    )

    return {
        "role": "assistant",
        "content": content,
        "reasoning_content": reasoning_content,
        "tool_calls": tool_calls,
        "calls": tool_calls,
    }


class ChatToGenerateAdapter:
    """Adapter to convert chat completions API to generate API or pass-through v1/completions"""

    def __init__(
        self,
        router_url: str = ROUTER_URL,
        model: str = MODEL,
        api_key: str = API_KEY,
        tokenizer_path: str = TOKENIZER_PATH,
        use_generate_api: bool = False,  # Use /generate API
        use_completions_for_chat: bool = True,  # Convert chat to completions
    ):
        self.router_url = router_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self._http_client: Optional[httpx.AsyncClient] = None
        self.use_generate_api = use_generate_api
        self.use_completions_for_chat = use_completions_for_chat

        # Load tokenizer for generate API or chat->completions conversion
        self.tokenizer = None
        if use_generate_api or use_completions_for_chat:
            if not tokenizer_path:
                raise ValueError("TOKENIZER_PATH is required when generate/completions adaptation is enabled")
            print(f"Loading tokenizer from {tokenizer_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            print("Tokenizer loaded successfully")

    async def get_http_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client"""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(3000),
                limits=httpx.Limits(
                    max_connections=5000,
                    max_keepalive_connections=4000,
                ),
                follow_redirects=True,
            )
        return self._http_client

    async def close(self):
        """Close HTTP client"""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    def _normalize_messages_for_chat_template(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize messages to match GLM chat_template expectations.

        The GLM-4.7 chat template expects tool call arguments to be a dict so it can
        iterate via `.items()`. OpenAI-style requests usually carry `function.arguments`
        as a JSON *string*, so we parse it into a dict here.
        """
        for msg in messages:
            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                continue
            if not isinstance(tool_calls, list):
                continue

            normalized_tool_calls = []
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue

                # Get function object (handle both OpenAI format and direct format)
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc

                # Extract name and arguments
                name = fn.get("name", "")
                args = fn.get("arguments")

                # Normalize arguments to dict
                if args is None:
                    arguments = {}
                elif isinstance(args, str):
                    raw = args.strip()
                    if not raw:
                        arguments = {}
                    else:
                        try:
                            loaded = json.loads(raw)
                            arguments = loaded if isinstance(loaded, dict) else {"_": loaded}
                        except Exception:
                            arguments = {"_raw": args}
                else:
                    arguments = args if isinstance(args, dict) else {}

                normalized_tool_calls.append({"name": name, "arguments": arguments})

            msg["tool_calls"] = normalized_tool_calls
        return messages

    def _convert_messages_to_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """
        Convert chat messages to GLM-4.7 prompt format

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Formatted prompt string
        """
        prompt_parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                prompt_parts.append(content)
            elif role == "user":
                prompt_parts.append(f"user: {content}")
            elif role == "assistant":
                prompt_parts.append(f"assistant: {content}")
            elif role == "tool":
                # Tool result messages
                prompt_parts.append(f"tool result: {content}")

        return "\n".join(prompt_parts)

    def _resolve_main_key(self, chat_request: Dict[str, Any]) -> Optional[str]:
        """Prefer explicit main_key, then derive it from metadata.user_id/session_id-bearing user fields."""
        main_key = chat_request.get("main_key")
        if main_key is not None:
            print(f"[main_key] resolved from request.main_key: {main_key}")
            return main_key

        metadata = chat_request.get("metadata")
        if isinstance(metadata, dict):
            user_id = metadata.get("user_id")
            if isinstance(user_id, dict):
                session_id = user_id.get("session_id")
                if session_id is not None:
                    resolved = str(session_id)
                    print(f"[main_key] resolved from metadata.user_id.session_id: {resolved}")
                    return resolved

        user = chat_request.get("user")
        if isinstance(user, dict):
            session_id = user.get("session_id")
            if session_id is not None:
                resolved = str(session_id)
                print(f"[main_key] resolved from request.user.session_id: {resolved}")
                return resolved
        elif isinstance(user, str):
            raw = user.strip()
            if raw:
                try:
                    parsed_user = json.loads(raw)
                except json.JSONDecodeError:
                    parsed_user = None
                if isinstance(parsed_user, dict):
                    session_id = parsed_user.get("session_id")
                    if session_id is not None:
                        resolved = str(session_id)
                        print(f"[main_key] resolved from request.user JSON session_id: {resolved}")
                        return resolved

        print("[main_key] not found: no session_id in metadata.user_id or request.user")
        return None

    def _build_generate_request(self, chat_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert chat completions request to generate API request

        Args:
            chat_request: Original chat completions request

        Returns:
            Request for generate API
        """
        messages = chat_request.get("messages", [])
        tools = chat_request.get("tools", [])
        if tools:
            tools_system_prompt = self._build_glm47_tools_prompt(tools)

            # Find system message or create one
            has_system = any(msg.get("role") == "system" for msg in messages)
            if has_system:
                # Append to existing system message
                for msg in messages:
                    if msg.get("role") == "system":
                        if tools_system_prompt not in msg["content"]:
                            msg["content"] = tools_system_prompt + "<|system|>" + msg["content"]
                        break
            else:
                messages = [{"role": "system", "content": tools_system_prompt}] + messages

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Set use_generate_api=True in __init__")

        messages = self._normalize_messages_for_chat_template(messages)

        normalized_data = {"messages": messages, "tools": tools if tools else None}
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Build generate request
        generate_request = {
            "model": self.model,
            "text": prompt,
            "sampling_params": {
                "temperature": chat_request.get("temperature", 1),
                "top_p": chat_request.get("top_p", 1),
                "max_new_tokens": chat_request.get("max_tokens", DEFAULT_MAX_TOKENS),
                "no_stop_trim": False,
            },
        }
        if "stop" in chat_request and chat_request["stop"] is not None:
            generate_request["sampling_params"]["stop"] = chat_request["stop"]

        main_key = self._resolve_main_key(chat_request)
        if main_key is not None:
            generate_request["main_key"] = main_key

        return generate_request

    def _build_glm47_tools_prompt(self, tools: List[Dict[str, Any]]) -> str:
        if not tools:
            return ""

        system_prompt = ""
        system_prompt += "\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"

        for tool in tools:
            # Handle both OpenAI format and direct format
            if "function" in tool:
                # Standard OpenAI format: {"type": "function", "function": {...}}
                tool_to_dump = tool["function"]
            else:
                # Direct format: {"name": ..., "description": ..., ...}
                tool_to_dump = tool

            system_prompt += "\n" + json.dumps(tool_to_dump, ensure_ascii=False)

        system_prompt += "\n</tools>\n\n"
        system_prompt += (
            "For each function call, output the function name and arguments within the following XML format:\n"
            "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>"
        )

        return system_prompt

    def _build_chat_completion_response(
        self,
        request_id: str,
        parsed_response: Dict[str, Any],
        usage: Optional[Dict[str, int]] = None,
        finish_reason: str = "stop",
    ) -> Any:
        """
        Build chat completions format response

        Args:
            request_id: Unique request ID
            parsed_response: Parsed generate response with reasoning_content, content, tool_calls
            usage: Token usage info

        Returns:
            Chat completions format response
        """
        if usage is None:
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # Combine reasoning and content into message content
        message_content = parsed_response.get("content", "")
        reasoning_content = parsed_response.get("reasoning_content", "")

        return {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "reasoning_content": reasoning_content,
                        "content": message_content,
                        "tool_calls": (
                            parsed_response.get("tool_calls", []) if parsed_response.get("tool_calls") else None
                        ),
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
        }

    def _streaming_response(self, iterator) -> StreamingResponse:
        return StreamingResponse(
            iterator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @staticmethod
    def _normalize_tool_calls_to_openai(raw_tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert flat tool_calls from parse_message_local to nested OpenAI streaming format.

        Flat:  {"tool_call_id": "xxx", "name": "Read", "arguments": "{...}"}
        OpenAI: {"id": "xxx", "type": "function", "function": {"name": "Read", "arguments": "{...}"}}
        """
        result = []
        for tc in raw_tool_calls:
            tc_id = tc.get("id") or tc.get("tool_call_id") or f"call_{uuid.uuid4().hex[:24]}"
            name = tc.get("name", "")
            arguments = tc.get("arguments", "{}")
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments, ensure_ascii=False)
            result.append({
                "id": tc_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            })
        return result

    def _chat_chunk_sse(
        self,
        request_id: str,
        created: int,
        model: str,
        content: Optional[str] = None,
        reasoning_content: Optional[str] = None,
        finish_reason: Optional[str] = None,
        usage: Optional[Dict[str, Any]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        delta: Dict[str, Any] = {}
        if content:
            delta["content"] = content
        if reasoning_content:
            delta["reasoning_content"] = reasoning_content
        if tool_calls:
            delta["tool_calls"] = tool_calls

        payload: Dict[str, Any] = {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }
        if usage is not None:
            payload["usage"] = usage
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    async def _stream_raw_provider_response(
        self,
        url: str,
        headers: Dict[str, str],
        body: Dict[str, Any],
        timeout: float = 1800.0,
        max_retries: int = 10,
    ) -> StreamingResponse:
        print(f"[stream] opening raw provider stream: {url}")
        stream = await open_provider_stream(url, headers, body, timeout=timeout, max_retries=max_retries)
        print(f"[stream] raw provider stream established: {url}")

        async def iterator():
            try:
                async for chunk in stream.aiter_raw():
                    yield chunk
            finally:
                await stream.aclose()

        return self._streaming_response(iterator())

    async def _mock_generate_to_chat_stream(
        self,
        generate_request: Dict[str, Any],
        headers: Dict[str, str],
        chat_request: Dict[str, Any],
    ) -> StreamingResponse:
        request_id = str(uuid.uuid4())
        created = int(time.time())
        upstream_request = dict(generate_request)
        upstream_request.pop("stream", None)

        print(f"[stream] mocking chat stream from non-stream generate: {self.router_url}/generate")
        resp = await self.forward_request(upstream_request, headers)
        if resp.status_code != 200:
            raise RuntimeError(f"sglang returned status {resp.status_code}: {resp.text[:500]}")

        response_data = resp.json()
        text = response_data.get("text", "")
        tools_for_parser = []
        for tool in chat_request.get("tools", []):
            if "function" in tool:
                tool_to_dump = tool["function"]
            else:
                tool_to_dump = tool
            tools_for_parser.append(tool_to_dump)
        parsed = parse_message_local(text, "glm47", tools_for_parser)
        content = parsed.get("content", "")
        reasoning_content = parsed.get("reasoning_content", "")
        if content.endswith("<|user|>"):
            content = content[:-8].rstrip()

        meta_info = response_data.get("meta_info", {}) or {}
        usage = {
            "prompt_tokens": meta_info.get("prompt_tokens", 0),
            "completion_tokens": meta_info.get("completion_tokens", 0),
            "total_tokens": meta_info.get("prompt_tokens", 0) + meta_info.get("completion_tokens", 0),
        }
        finish_reason = meta_info.get("finish_reason", {}).get("type", "stop")

        raw_tool_calls = parsed.get("tool_calls", [])
        openai_tool_calls = self._normalize_tool_calls_to_openai(raw_tool_calls) if raw_tool_calls else []
        if openai_tool_calls:
            finish_reason = "tool_calls"

        async def iterator():
            if reasoning_content:
                yield self._chat_chunk_sse(
                    request_id,
                    created,
                    self.model,
                    reasoning_content=reasoning_content,
                )
            if content:
                yield self._chat_chunk_sse(request_id, created, self.model, content=content)
            if openai_tool_calls:
                yield self._chat_chunk_sse(
                    request_id,
                    created,
                    self.model,
                    tool_calls=openai_tool_calls,
                )
            yield self._chat_chunk_sse(
                request_id,
                created,
                self.model,
                finish_reason=finish_reason,
                usage=usage,
            )
            yield "data: [DONE]\n\n"

        return self._streaming_response(iterator())

    async def _stream_completions_to_chat(
        self,
        completions_request: Dict[str, Any],
        request_headers: Dict[str, str],
    ) -> StreamingResponse:
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.router_url}/v1/completions"
        print(f"[stream] opening completions stream: {url}")
        stream = await open_provider_stream(url, headers, completions_request, timeout=1800.0, max_retries=10)
        print(f"[stream] completions stream established: {url}")
        request_id = str(uuid.uuid4())
        created = int(time.time())

        # Get tools from the original chat request for parsing
        chat_tools = completions_request.pop("_chat_tools", [])

        async def iterator():
            text_buf = []
            last_usage = None
            last_finish_reason = None
            try:
                async for raw in stream:
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        # Parse accumulated text for tool calls
                        if chat_tools and text_buf:
                            full_text = "".join(text_buf)
                            full_text = full_text.rstrip()
                            if full_text.endswith("(prompt"):
                                full_text = full_text[:-8].rstrip()
                            full_text = f"\U0001f508{full_text}"
                            parsed = parse_message_local(full_text, "glm47", chat_tools)
                            raw_tcs = parsed.get("tool_calls", [])
                            openai_tcs = self._normalize_tool_calls_to_openai(raw_tcs) if raw_tcs else []
                            if openai_tcs:
                                yield self._chat_chunk_sse(
                                    request_id, created, self.model, tool_calls=openai_tcs,
                                )
                                last_finish_reason = "tool_calls"
                        yield self._chat_chunk_sse(
                            request_id, created, self.model,
                            finish_reason=last_finish_reason,
                            usage=last_usage,
                        )
                        yield "data: [DONE]\n\n"
                        return

                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue

                    choice = choices[0]
                    text = choice.get("text", "")
                    finish_reason = choice.get("finish_reason")
                    usage = chunk.get("usage")

                    if text:
                        text_buf.append(text)
                        # If we have tools, buffer text; otherwise stream immediately
                        if not chat_tools:
                            yield self._chat_chunk_sse(request_id, created, self.model, content=text)
                    if finish_reason is not None:
                        last_finish_reason = finish_reason
                    if usage:
                        last_usage = usage
            finally:
                await stream.aclose()

        return self._streaming_response(iterator())

    async def forward_request(
        self,
        generate_request: Dict[str, Any],
        headers: Dict[str, str],
    ) -> httpx.Response:
        client = await self.get_http_client()
        url = f"{self.router_url}/generate"

        retry_statuses = {429, 500, 501, 502, 503, 504}
        max_retries = 10
        retry_delay = 2  # seconds
        resp: Optional[httpx.Response] = None

        for attempt in range(1, max_retries + 1):
            try:
                resp = await client.request(
                    method="POST",
                    url=url,
                    headers=headers,
                    json=generate_request,
                    timeout=httpx.Timeout(1800),
                )
                if resp.status_code in retry_statuses:
                    print(f"[DEBUG] Retry status {resp.status_code}, retrying in {retry_delay}s...")
                    if attempt < max_retries:
                        await asyncio.sleep(retry_delay)
                        continue

                return resp

            except httpx.RequestError as e:
                print(f"[DEBUG] Request error: {e}, retrying in {retry_delay}s...")
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay)
                    continue
                raise

        if resp is None:
            raise RuntimeError("Failed to get response from sglang")
        return resp

    async def process_chat_completion_request(
        self,
        chat_request: Dict[str, Any],
        request_headers: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Process chat completion request - via generate, completions, or chat/completions

        Args:
            chat_request: Chat completions API request
            request_headers: HTTP headers from request

        Returns:
            Chat completions format response
        """
        # Check routing strategy
        if self.use_generate_api:
            # Use /generate API (sglang)
            return await self._process_chat_via_generate(chat_request, request_headers)
        elif self.use_completions_for_chat:
            # Convert chat -> completions -> chat (more stable, avoids streaming issues)
            return await self._process_chat_via_completions(chat_request, request_headers)
        else:
            # Direct pass-through to v1/chat/completions
            return await self._process_chat_via_v1(chat_request, request_headers)

    async def _process_chat_via_generate(
        self,
        chat_request: Dict[str, Any],
        request_headers: Dict[str, str],
    ) -> Any:
        request_id = str(uuid.uuid4())
        generate_request = self._build_generate_request(chat_request)
        is_stream = bool(chat_request.get("stream"))
        headers = {
            k: v
            for k, v in request_headers.items()
            if k.lower() not in ("host", "content-length", "transfer-encoding")
        }
        headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if is_stream:
            return await self._mock_generate_to_chat_stream(generate_request, headers, chat_request)

        # Forward to sglang generate
        try:
            resp = await self.forward_request(generate_request, headers)
        except Exception as e:
            raise RuntimeError(f"Failed to forward request to sglang: {e}")

        if resp.status_code != 200:
            raise RuntimeError(f"sglang returned status {resp.status_code}: {resp.text[:500]}")

        response_data = resp.json()
        text = response_data.get("text", "")

        # Parse the response
        tools_for_parser = []
        for tool in chat_request.get("tools", []):
            if "function" in tool:
                # Standard OpenAI format: {"type": "function", "function": {...}}
                tool_to_dump = tool["function"]
            else:
                # Direct format: {"name": ..., "description": ..., ...}
                tool_to_dump = tool
            tools_for_parser.append(tool_to_dump)
        parsed = parse_message_local(text, "glm47", tools_for_parser)
        content = parsed['content']
        if content.endswith('<|user|>'):
            content = content[:-8].rstrip()
            parsed['content'] = content

        meta_info = response_data.get("meta_info", {})
        usage = {
            "prompt_tokens": meta_info.get("prompt_tokens", 0),
            "completion_tokens": meta_info.get("completion_tokens", 0),
            "total_tokens": meta_info.get("completion_tokens", 0) + meta_info.get("prompt_tokens", 0),
        }
        finish_reason = meta_info.get("finish_reason", {}).get("type", "stop")
        chat_response = self._build_chat_completion_response(request_id, parsed, usage, finish_reason=finish_reason)
        return chat_response

    async def _process_chat_via_v1(
        self,
        chat_request: Dict[str, Any],
        request_headers: Dict[str, str],
    ) -> Any:
        """Pass through to v1/chat/completions API (for OpenAI-compatible APIs like ZhipuAI)"""
        is_stream = bool(chat_request.get("stream"))

        # Prepare headers - don't inherit client headers, build fresh ones
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if is_stream:
            return await self._stream_raw_provider_response(
                f"{self.router_url}/v1/chat/completions",
                headers,
                chat_request,
            )

        # Forward to v1/chat/completions with retry
        client = await self.get_http_client()
        url = f"{self.router_url}/v1/chat/completions"

        max_retries = 10
        retry_delay = 2  # seconds

        for attempt in range(1, max_retries + 1):
            # Debug log
            print(f"[DEBUG] Attempt {attempt}/{max_retries}: Forwarding to {url}")
            if attempt == 1:
                print(f"[DEBUG] Headers: {headers}")

            try:
                resp = await client.request(
                    method="POST",
                    url=url,
                    headers=headers,
                    json=chat_request,
                    timeout=httpx.Timeout(1800),
                )

                print(f"[DEBUG] Response status: {resp.status_code}")

                if resp.status_code != 200:
                    print(f"[DEBUG] Response headers: {dict(resp.headers)}")
                    print(f"[DEBUG] Response body: {resp.text[:1000]}")

                    if attempt < max_retries:
                        print(f"[DEBUG] Non-200 status, retrying in {retry_delay}s...")
                        await asyncio.sleep(retry_delay)
                        continue

                    raise RuntimeError(f"v1/chat/completions returned status {resp.status_code}: {resp.text[:500]}")

                # Try to parse JSON
                try:
                    return resp.json()
                except Exception as e:
                    print(f"[DEBUG] JSON parse error: {e}")
                    print(f"[DEBUG] Response text: {resp.text[:500]}")

                    if attempt < max_retries:
                        print(f"[DEBUG] JSON parse failed, retrying in {retry_delay}s...")
                        await asyncio.sleep(retry_delay)
                        continue

                    raise RuntimeError(f"Failed to parse v1/chat/completions response: {e}")

            except httpx.RequestError as e:
                print(f"[DEBUG] Request error: {e}")

                if attempt < max_retries:
                    print(f"[DEBUG] Request failed, retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    continue

                raise RuntimeError(f"Failed to forward request to v1/chat/completions: {e}")

        raise RuntimeError(f"Failed after {max_retries} retries")

    async def _process_chat_via_completions(
        self,
        chat_request: Dict[str, Any],
        request_headers: Dict[str, str],
    ) -> Any:
        """
        Convert chat/completions to v1/completions, then convert back to chat format
        This avoids streaming issues with chat/completions API

        Args:
            chat_request: Chat completions API request
            request_headers: HTTP headers from request

        Returns:
            Chat completions format response
        """
        # Step 1: Build tools system prompt if tools are present
        messages = chat_request.get("messages", [])
        tools = chat_request.get("tools", [])

        # Add tools system prompt if tools are provided
        if tools:
            tools_system_prompt = self._build_glm47_tools_prompt(tools)

            # Find system message or create one
            has_system = any(msg.get("role") == "system" for msg in messages)
            if has_system:
                # Append to existing system message
                for msg in messages:
                    if msg.get("role") == "system":
                        msg["content"] = msg["content"] + "\n\n" + tools_system_prompt
                        break
            else:
                messages = [{"role": "system", "content": tools_system_prompt}] + messages

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Set use_completions_for_chat=True in __init__")

        messages = self._normalize_messages_for_chat_template(messages)

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # prompt = f'{prompt}\n<think>'

        completions_request = {
            "model": chat_request.get("model", self.model),
            "prompt": prompt,
            "max_tokens": chat_request.get("max_tokens", DEFAULT_MAX_TOKENS),
            "temperature": chat_request.get("temperature", 0.0),
            "top_p": chat_request.get("top_p", 0.95),
            "stream": bool(chat_request.get("stream")),
        }

        if "stop" in chat_request:
            completions_request["stop"] = chat_request["stop"]

        main_key = self._resolve_main_key(chat_request)
        if main_key is not None:
            completions_request["main_key"] = main_key

        # Build tool definitions for parser (used by both streaming and non-streaming)
        tools_for_parser = []
        for tool in chat_request.get("tools", []):
            if "function" in tool:
                tool_to_dump = tool["function"]
            else:
                tool_to_dump = tool
            tools_for_parser.append(tool_to_dump)

        if completions_request["stream"]:
            # Pass tool definitions so the streaming path can parse tool XML
            completions_request["_chat_tools"] = tools_for_parser
            return await self._stream_completions_to_chat(completions_request, request_headers)

        completions_response = await self._process_completions_via_v1(completions_request, request_headers)

        completion_text = completions_response["choices"][0]["text"]
        
        print(completion_text)
        print(repr(completion_text[-8:]))
        print(completion_text.endswith("<|user|>"))

        if not completion_text or not completion_text.strip():
            raise RuntimeError("API returned an empty response")

        completion_text = completion_text.rstrip()
        if completion_text.endswith("<|user|>"):
            completion_text = completion_text[:-8].rstrip()

        completion_text = f"<think>{completion_text}"
        
        parsed = parse_message_local(completion_text, "glm47", tools_for_parser)

        content = parsed.get("content", "")
        reasoning_content = parsed.get("reasoning_content", "")
        tool_calls = parsed.get("tool_calls", []) or []

        message = {
            "role": "assistant",
            "content": content,
        }

        if reasoning_content:
            message["reasoning_content"] = reasoning_content
            print(f"[DEBUG] Added reasoning_content field to message")
        else:
            print(f"[DEBUG] No reasoning_content to add")

        # Add tool_calls if present
        if tool_calls:
            message["tool_calls"] = tool_calls

        chat_response = {
            "id": completions_response.get("id", f"chatcmpl-{uuid.uuid4()}"),
            "object": "chat.completion",
            "created": completions_response.get("created", int(time.time())),
            "model": completions_response.get("model", self.model),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": completions_response["choices"][0].get("finish_reason", "stop"),
                }
            ],
            "usage": completions_response.get(
                "usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            ),
        }

        print(
            f"[DEBUG] Parsed completions response: reasoning={len(reasoning_content)} chars, content={len(content)} chars, tool_calls={len(tool_calls)}"
        )

        return chat_response

    async def process_completions_request(
        self,
        completions_request: Dict[str, Any],
        request_headers: Dict[str, str],
    ) -> Any:
        """
        Process v1/completions request (raw text completion)

        Args:
            completions_request: Completions API request
            request_headers: HTTP headers from request

        Returns:
            Completions format response
        """
        # Check if we should use generate API or v1/completions
        use_generate = completions_request.pop("use_generate_api", self.use_generate_api)

        # Pass through to v1/completions
        return await self._process_completions_via_v1(completions_request, request_headers)

    async def _process_completions_via_v1(
        self,
        completions_request: Dict[str, Any],
        request_headers: Dict[str, str],
    ) -> Any:
        """Pass through to v1/completions API"""
        is_stream = bool(completions_request.get("stream"))

        # Prepare headers - don't inherit client headers, build fresh ones
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if is_stream:
            return await self._stream_raw_provider_response(
                f"{self.router_url}/v1/completions",
                headers,
                completions_request,
            )

        # Forward to v1/completions with retry
        client = await self.get_http_client()
        url = f"{self.router_url}/v1/completions"

        max_retries = 10
        retry_delay = 2  # seconds

        for attempt in range(1, max_retries + 1):
            print(f"[DEBUG] Attempt {attempt}/{max_retries}: Forwarding to {url}")
            try:
                resp = await client.request(
                    method="POST",
                    url=url,
                    headers=headers,
                    json=completions_request,
                    timeout=httpx.Timeout(1800),
                )

                print(f"[DEBUG] Response status: {resp.status_code}")

                if resp.status_code != 200:
                    print(f"[DEBUG] Response body: {resp.text[:1000]}")

                    if attempt < max_retries:
                        print(f"[DEBUG] Non-200 status, retrying in {retry_delay}s...")
                        await asyncio.sleep(retry_delay)
                        continue

                    raise RuntimeError(f"v1/completions returned status {resp.status_code}: {resp.text[:500]}")

                # Try to parse JSON
                try:
                    return resp.json()
                except Exception as e:
                    print(f"[DEBUG] JSON parse error: {e}")
                    print(f"[DEBUG] Response text: {resp.text[:500]}")

                    if attempt < max_retries:
                        print(f"[DEBUG] JSON parse failed, retrying in {retry_delay}s...")
                        await asyncio.sleep(retry_delay)
                        continue

                    raise RuntimeError(f"Failed to parse v1/completions response: {e}")

            except httpx.RequestError as e:
                print(f"[DEBUG] Request error: {e}")

                if attempt < max_retries:
                    print(f"[DEBUG] Request failed, retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    continue

                raise RuntimeError(f"Failed to forward request to v1/completions: {e}")

        raise RuntimeError(f"Failed after {max_retries} retries")

    async def process_request(
        self,
        request_data: Dict[str, Any],
        request_headers: Dict[str, str],
        api_type: str = "chat",
    ) -> Any:
        """
        Process request based on API type

        Args:
            request_data: API request
            request_headers: HTTP headers from request
            api_type: 'chat' for chat/completions, 'completions' for v1/completions

        Returns:
            API response
        """
        if api_type == "chat":
            return await self.process_chat_completion_request(request_data, request_headers)
        elif api_type == "completions":
            return await self.process_completions_request(request_data, request_headers)
        else:
            raise ValueError(f"Unknown API type: {api_type}")

def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _build_adapter_from_env() -> ChatToGenerateAdapter:
    use_generate_api = _env_flag("USE_GENERATE_API", True)
    use_completions_for_chat = _env_flag("USE_COMPLETIONS_FOR_CHAT", True)
    logger.info(
        "Initializing chat adapter: router_url=%s model=%s use_generate_api=%s use_completions_for_chat=%s",
        ROUTER_URL,
        MODEL,
        use_generate_api,
        use_completions_for_chat,
    )
    return ChatToGenerateAdapter(
        router_url=ROUTER_URL,
        model=MODEL,
        api_key=API_KEY,
        tokenizer_path=TOKENIZER_PATH,
        use_generate_api=use_generate_api,
        use_completions_for_chat=use_completions_for_chat,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.adapter = _build_adapter_from_env()
    try:
        yield
    finally:
        await app.state.adapter.close()


app = FastAPI(lifespan=lifespan)


def _get_adapter(request: Request) -> ChatToGenerateAdapter:
    return request.app.state.adapter


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Endpoint: chat completions that internally use generate API"""
    request_id = str(uuid.uuid4())
    print(f"[chat_completions] Request {request_id}")

    try:
        chat_request = await asyncio.wait_for(request.json(), timeout=1800)
    except Exception as e:
        traceback.print_exc()
        return Response(
            content=orjson.dumps({"error": "Invalid JSON or timeout"}), status_code=400, media_type="application/json"
        )

    try:
        response = await _get_adapter(request).process_request(chat_request, dict(request.headers), api_type="chat")
        if isinstance(response, Response):
            return response

        return Response(content=orjson.dumps(response), status_code=200, media_type="application/json")

    except Exception as e:
        traceback.print_exc()
        print(f"[chat_completions] Error: {e}")
        return Response(content=orjson.dumps({"error": str(e)}), status_code=502, media_type="application/json")


@app.post("/v1/completions")
async def completions(request: Request):
    """Endpoint: text completions (raw prompt completion)"""
    request_id = str(uuid.uuid4())
    print(f"[completions] Request {request_id}")

    try:
        completions_request = await asyncio.wait_for(request.json(), timeout=1800)
    except Exception as e:
        traceback.print_exc()
        return Response(
            content=orjson.dumps({"error": "Invalid JSON or timeout"}), status_code=400, media_type="application/json"
        )

    try:
        response = await _get_adapter(request).process_request(
            completions_request, dict(request.headers), api_type="completions"
        )
        if isinstance(response, Response):
            return response

        return Response(content=orjson.dumps(response), status_code=200, media_type="application/json")

    except Exception as e:
        traceback.print_exc()
        print(f"[completions] Error: {e}")
        return Response(content=orjson.dumps({"error": str(e)}), status_code=502, media_type="application/json")


@app.post("/tokens/clear")
async def tokens_clear(request: Request):
    """Proxy /tokens/clear to the sglang router"""
    try:
        body = await asyncio.wait_for(request.json(), timeout=30)
    except Exception as e:
        return Response(
            content=orjson.dumps({"error": "Invalid JSON or timeout"}), status_code=400, media_type="application/json"
        )

    adapter = _get_adapter(request)
    client = await adapter.get_http_client()
    url = f"{adapter.router_url}/tokens/clear"
    headers = {"Content-Type": "application/json"}
    try:
        resp = await client.request(
            method="POST",
            url=url,
            headers=headers,
            json=body,
            timeout=httpx.Timeout(120),
        )
        return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")
    except Exception as e:
        traceback.print_exc()
        return Response(content=orjson.dumps({"error": str(e)}), status_code=502, media_type="application/json")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
