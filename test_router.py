"""
Comprehensive tests for the Python Claude Code Router.

Run:
    python test_router.py            # all tests (unit + integration if reachable)
    python test_router.py unit       # unit tests only (no network)
    python test_router.py integration
"""

import asyncio
import json
import os
import sys
import unittest

from converter import anthropic_to_openai, openai_to_anthropic, stream_openai_to_anthropic
from config import apply_provider_params, load_config, resolve_route, get_provider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROVIDER_URL = os.environ.get("PROVIDER_URL", "http://172.27.10.49:8000/v1/chat/completions")
PROVIDER_KEY  = os.environ.get("PROVIDER_KEY",  "EMPTY")
PROVIDER_MODEL = os.environ.get("PROVIDER_MODEL", "/model")
CONFIG_PATH   = os.environ.get("CCR_CONFIG", "config.json")


# ============================================================================
# Unit tests — converter: Anthropic → OpenAI
# ============================================================================

class TestAnthropicToOpenAI(unittest.TestCase):

    # ── basic fields ────────────────────────────────────────────────────────

    def test_simple_text(self):
        req = {"model": "m", "max_tokens": 1024,
               "messages": [{"role": "user", "content": "Hello!"}]}
        out = anthropic_to_openai(req)
        self.assertEqual(out["messages"], [{"role": "user", "content": "Hello!"}])
        self.assertEqual(out["max_tokens"], 1024)

    def test_system_string(self):
        req = {"model": "m", "messages": [], "system": "You are helpful."}
        out = anthropic_to_openai(req)
        self.assertEqual(out["messages"][0], {"role": "system", "content": "You are helpful."})

    def test_system_array(self):
        req = {"model": "m", "messages": [], "system": [
            {"type": "text", "text": "Part 1"},
            {"type": "text", "text": "Part 2"},
        ]}
        out = anthropic_to_openai(req)
        self.assertEqual(out["messages"][0]["content"], "Part 1\nPart 2")

    def test_stop_sequences(self):
        req = {"model": "m", "messages": [], "stop_sequences": ["STOP", "END"]}
        out = anthropic_to_openai(req)
        self.assertEqual(out["stop"], ["STOP", "END"])

    def test_sampling_params_passthrough(self):
        req = {"model": "m", "messages": [],
               "temperature": 0.5, "top_p": 0.9, "top_k": 40, "max_tokens": 512}
        out = anthropic_to_openai(req)
        self.assertEqual(out["temperature"], 0.5)
        self.assertEqual(out["top_p"], 0.9)
        self.assertEqual(out["top_k"], 40)
        self.assertEqual(out["max_tokens"], 512)

    def test_stream_options_added(self):
        req = {"model": "m", "messages": [], "stream": True}
        out = anthropic_to_openai(req)
        self.assertTrue(out["stream"])
        self.assertEqual(out["stream_options"], {"include_usage": True})

    def test_no_stream_options_when_not_streaming(self):
        req = {"model": "m", "messages": []}
        out = anthropic_to_openai(req)
        self.assertNotIn("stream_options", out)

    # ── message content blocks ───────────────────────────────────────────────

    def test_tool_use_in_assistant(self):
        req = {"model": "m", "messages": [{"role": "assistant", "content": [
            {"type": "text", "text": "Let me check"},
            {"type": "tool_use", "id": "tu_123", "name": "get_weather", "input": {"city": "NYC"}},
        ]}]}
        out = anthropic_to_openai(req)
        msg = out["messages"][0]
        self.assertEqual(msg["content"], "Let me check")
        self.assertEqual(len(msg["tool_calls"]), 1)
        tc = msg["tool_calls"][0]
        self.assertEqual(tc["id"], "tu_123")
        self.assertEqual(tc["function"]["name"], "get_weather")
        self.assertEqual(json.loads(tc["function"]["arguments"]), {"city": "NYC"})

    def test_tool_result_in_user(self):
        req = {"model": "m", "messages": [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_123", "content": "Sunny, 72°F"}
        ]}]}
        out = anthropic_to_openai(req)
        msg = out["messages"][0]
        self.assertEqual(msg["role"], "tool")
        self.assertEqual(msg["tool_call_id"], "tu_123")
        self.assertEqual(msg["content"], "Sunny, 72°F")

    def test_tool_result_mixed_with_text(self):
        req = {"model": "m", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Here is the result:"},
            {"type": "tool_result", "tool_use_id": "tu_1", "content": "42"},
        ]}]}
        out = anthropic_to_openai(req)
        roles = [m["role"] for m in out["messages"]]
        self.assertIn("user", roles)
        self.assertIn("tool", roles)

    def test_thinking_skipped_in_assistant_history(self):
        req = {"model": "m", "messages": [{"role": "assistant", "content": [
            {"type": "thinking", "thinking": "internal", "signature": "sig"},
            {"type": "text", "text": "Answer"},
        ]}]}
        out = anthropic_to_openai(req)
        msg = out["messages"][0]
        self.assertEqual(msg["content"], "Answer")
        self.assertNotIn("tool_calls", msg)

    def test_image_base64(self):
        req = {"model": "m", "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc123=="}}
        ]}]}
        out = anthropic_to_openai(req)
        part = out["messages"][0]["content"][0]
        self.assertEqual(part["type"], "image_url")
        self.assertEqual(part["image_url"]["url"], "data:image/png;base64,abc123==")

    def test_image_url(self):
        req = {"model": "m", "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "url", "url": "https://example.com/img.png"}}
        ]}]}
        out = anthropic_to_openai(req)
        part = out["messages"][0]["content"][0]
        self.assertEqual(part["image_url"]["url"], "https://example.com/img.png")

    # ── tools ────────────────────────────────────────────────────────────────

    def test_tools_conversion(self):
        req = {"model": "m", "messages": [], "tools": [
            {"name": "search", "description": "Web search",
             "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}
        ]}
        out = anthropic_to_openai(req)
        tool = out["tools"][0]
        self.assertEqual(tool["type"], "function")
        self.assertEqual(tool["function"]["name"], "search")
        self.assertIn("properties", tool["function"]["parameters"])

    def test_tool_strict_field(self):
        req = {"model": "m", "messages": [], "tools": [
            {"name": "fn", "description": "", "input_schema": {"type": "object"}, "strict": True}
        ]}
        out = anthropic_to_openai(req)
        self.assertTrue(out["tools"][0]["function"]["strict"])

    def test_tool_strict_false(self):
        req = {"model": "m", "messages": [], "tools": [
            {"name": "fn", "description": "", "input_schema": {"type": "object"}, "strict": False}
        ]}
        out = anthropic_to_openai(req)
        self.assertFalse(out["tools"][0]["function"]["strict"])

    def test_tool_no_strict_omitted(self):
        req = {"model": "m", "messages": [], "tools": [
            {"name": "fn", "description": "", "input_schema": {"type": "object"}}
        ]}
        out = anthropic_to_openai(req)
        self.assertNotIn("strict", out["tools"][0]["function"])

    # ── tool_choice ──────────────────────────────────────────────────────────

    def test_tool_choice_auto(self):
        out = anthropic_to_openai({"model": "m", "messages": [], "tool_choice": {"type": "auto"}})
        self.assertEqual(out["tool_choice"], "auto")

    def test_tool_choice_any(self):
        out = anthropic_to_openai({"model": "m", "messages": [], "tool_choice": {"type": "any"}})
        self.assertEqual(out["tool_choice"], "required")

    def test_tool_choice_specific(self):
        out = anthropic_to_openai({"model": "m", "messages": [],
                                   "tool_choice": {"type": "tool", "name": "search"}})
        self.assertEqual(out["tool_choice"]["function"]["name"], "search")

    def test_tool_choice_none(self):
        out = anthropic_to_openai({"model": "m", "messages": [], "tool_choice": {"type": "none"}})
        self.assertEqual(out["tool_choice"], "none")

    def test_tool_choice_disable_parallel(self):
        out = anthropic_to_openai({"model": "m", "messages": [],
                                   "tool_choice": {"type": "auto", "disable_parallel_tool_use": True}})
        self.assertFalse(out["parallel_tool_calls"])

    def test_tool_choice_parallel_not_set_by_default(self):
        out = anthropic_to_openai({"model": "m", "messages": [],
                                   "tool_choice": {"type": "auto"}})
        self.assertNotIn("parallel_tool_calls", out)

    # ── thinking ─────────────────────────────────────────────────────────────

    def test_thinking_enabled(self):
        out = anthropic_to_openai({"model": "m", "messages": [],
                                   "thinking": {"type": "enabled", "budget_tokens": 5000}})
        self.assertEqual(out["thinking"], {"type": "enabled", "budget_tokens": 5000})

    def test_thinking_adaptive(self):
        out = anthropic_to_openai({"model": "m", "messages": [],
                                   "thinking": {"type": "adaptive"}})
        self.assertEqual(out["thinking"], {"type": "adaptive"})

    def test_thinking_disabled_omitted(self):
        out = anthropic_to_openai({"model": "m", "messages": [],
                                   "thinking": {"type": "disabled"}})
        self.assertNotIn("thinking", out)

    # ── output_config ────────────────────────────────────────────────────────

    def test_output_config_effort_low(self):
        out = anthropic_to_openai({"model": "m", "messages": [],
                                   "output_config": {"effort": "low"}})
        self.assertEqual(out["reasoning_effort"], "low")

    def test_output_config_effort_medium(self):
        out = anthropic_to_openai({"model": "m", "messages": [],
                                   "output_config": {"effort": "medium"}})
        self.assertEqual(out["reasoning_effort"], "medium")

    def test_output_config_effort_high(self):
        out = anthropic_to_openai({"model": "m", "messages": [],
                                   "output_config": {"effort": "high"}})
        self.assertEqual(out["reasoning_effort"], "high")

    def test_output_config_effort_max_becomes_xhigh(self):
        out = anthropic_to_openai({"model": "m", "messages": [],
                                   "output_config": {"effort": "max"}})
        self.assertEqual(out["reasoning_effort"], "xhigh")

    def test_output_config_format_json_schema(self):
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        out = anthropic_to_openai({"model": "m", "messages": [],
                                   "output_config": {"format": {"type": "json_schema", "schema": schema}}})
        self.assertEqual(out["response_format"]["type"], "json_schema")
        self.assertEqual(out["response_format"]["json_schema"], schema)

    def test_output_config_absent(self):
        out = anthropic_to_openai({"model": "m", "messages": []})
        self.assertNotIn("reasoning_effort", out)
        self.assertNotIn("response_format", out)

    # ── metadata ─────────────────────────────────────────────────────────────

    def test_metadata_user_id(self):
        out = anthropic_to_openai({"model": "m", "messages": [],
                                   "metadata": {"user_id": "user-abc"}})
        self.assertEqual(out["user"], "user-abc")

    def test_metadata_absent(self):
        out = anthropic_to_openai({"model": "m", "messages": []})
        self.assertNotIn("user", out)


# ============================================================================
# Unit tests — converter: OpenAI → Anthropic (non-streaming)
# ============================================================================

class TestOpenAIToAnthropic(unittest.TestCase):

    def _wrap(self, message, finish_reason="stop", usage=None):
        return {
            "id": "chatcmpl-test",
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5},
        }

    def test_simple_text(self):
        out = openai_to_anthropic(
            self._wrap({"role": "assistant", "content": "Hello!"}), "m")
        self.assertEqual(out["type"], "message")
        self.assertEqual(out["role"], "assistant")
        self.assertEqual(out["content"][0], {"type": "text", "text": "Hello!"})
        self.assertEqual(out["stop_reason"], "end_turn")

    def test_tool_call(self):
        out = openai_to_anthropic(self._wrap({
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "call_abc", "type": "function",
                            "function": {"name": "search", "arguments": '{"query":"python"}'}}],
        }, finish_reason="tool_calls"), "m")
        self.assertEqual(out["stop_reason"], "tool_use")
        b = out["content"][0]
        self.assertEqual(b["type"], "tool_use")
        self.assertEqual(b["id"], "call_abc")
        self.assertEqual(b["input"], {"query": "python"})

    def test_finish_reason_mapping(self):
        cases = [("stop", "end_turn"), ("length", "max_tokens"),
                 ("tool_calls", "tool_use"), ("content_filter", "stop_sequence")]
        for fr, expected in cases:
            out = openai_to_anthropic(
                self._wrap({"role": "assistant", "content": "x"}, finish_reason=fr), "m")
            self.assertEqual(out["stop_reason"], expected, fr)

    def test_usage_basic(self):
        out = openai_to_anthropic(
            self._wrap({"role": "assistant", "content": "x"},
                       usage={"prompt_tokens": 100, "completion_tokens": 50}), "m")
        self.assertEqual(out["usage"]["input_tokens"], 100)
        self.assertEqual(out["usage"]["output_tokens"], 50)
        self.assertEqual(out["usage"]["cache_read_input_tokens"], 0)
        self.assertEqual(out["usage"]["cache_creation_input_tokens"], 0)

    def test_usage_cached_tokens_subtracted(self):
        out = openai_to_anthropic(self._wrap(
            {"role": "assistant", "content": "x"},
            usage={"prompt_tokens": 100, "completion_tokens": 50,
                   "prompt_tokens_details": {"cached_tokens": 40}}), "m")
        self.assertEqual(out["usage"]["input_tokens"], 60)
        self.assertEqual(out["usage"]["cache_read_input_tokens"], 40)

    def test_usage_cache_creation_tokens(self):
        out = openai_to_anthropic(self._wrap(
            {"role": "assistant", "content": "x"},
            usage={"prompt_tokens": 100, "completion_tokens": 50,
                   "prompt_tokens_details": {
                       "cached_tokens": 20, "cache_creation_tokens": 30}}), "m")
        self.assertEqual(out["usage"]["input_tokens"], 50)
        self.assertEqual(out["usage"]["cache_read_input_tokens"], 20)
        self.assertEqual(out["usage"]["cache_creation_input_tokens"], 30)

    def test_reasoning_content(self):
        out = openai_to_anthropic(self._wrap(
            {"role": "assistant", "content": "Answer", "reasoning_content": "thinking..."}), "m")
        blocks = {b["type"]: b for b in out["content"]}
        self.assertIn("thinking", blocks)
        self.assertEqual(blocks["thinking"]["thinking"], "thinking...")
        self.assertIn("text", blocks)

    def test_thinking_object(self):
        out = openai_to_anthropic(self._wrap(
            {"role": "assistant", "content": "Answer",
             "thinking": {"content": "deep thought", "signature": "sig123"}}), "m")
        blocks = {b["type"]: b for b in out["content"]}
        self.assertIn("thinking", blocks)
        self.assertEqual(blocks["thinking"]["signature"], "sig123")

    def test_invalid_tool_args_fallback(self):
        out = openai_to_anthropic(self._wrap(
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c", "type": "function",
                             "function": {"name": "fn", "arguments": "bad-json"}}]},
            finish_reason="tool_calls"), "m")
        b = out["content"][0]
        self.assertEqual(b["type"], "tool_use")
        self.assertIsInstance(b["input"], dict)

    def test_response_schema_fields(self):
        out = openai_to_anthropic(
            self._wrap({"role": "assistant", "content": "hi"}), "mymodel")
        for f in ("id", "type", "role", "model", "content", "stop_reason", "stop_sequence", "usage"):
            self.assertIn(f, out, f"Missing field: {f}")
        self.assertEqual(out["model"], "mymodel")
        self.assertEqual(out["type"], "message")


# ============================================================================
# Unit tests — streaming converter
# ============================================================================

class TestStreamConverter(unittest.IsolatedAsyncioTestCase):

    async def _collect(self, chunks):
        async def fake_stream():
            for c in chunks:
                yield c
        events = []
        async for sse in stream_openai_to_anthropic(fake_stream(), "msg_test", "m"):
            for line in sse.strip().split("\n"):
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
        return events

    def _chunk(self, delta, finish_reason=None, usage=None):
        d = {"id": "x", "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
        if usage:
            d["usage"] = usage
        return b"data: " + json.dumps(d).encode()

    async def test_simple_text(self):
        events = await self._collect([
            self._chunk({"role": "assistant", "content": ""}),
            self._chunk({"content": "Hello"}),
            self._chunk({"content": " world"}),
            self._chunk({}, finish_reason="stop",
                        usage={"prompt_tokens": 10, "completion_tokens": 5}),
            b"data: [DONE]",
        ])
        types = [e["type"] for e in events]
        for t in ("message_start", "content_block_start", "content_block_delta",
                  "content_block_stop", "message_delta", "message_stop"):
            self.assertIn(t, types)
        text = "".join(
            e["delta"]["text"] for e in events
            if e["type"] == "content_block_delta"
            and e["delta"].get("type") == "text_delta"
        )
        self.assertEqual(text, "Hello world")

    async def test_tool_call_stream(self):
        events = await self._collect([
            self._chunk({"role": "assistant", "content": None}),
            self._chunk({"tool_calls": [{"index": 0, "id": "call_abc", "type": "function",
                                          "function": {"name": "search", "arguments": ""}}]}),
            self._chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"q"'}}]}),
            self._chunk({"tool_calls": [{"index": 0, "function": {"arguments": ':"hi"}'}}]}),
            self._chunk({}, finish_reason="tool_calls",
                        usage={"prompt_tokens": 20, "completion_tokens": 10}),
            b"data: [DONE]",
        ])
        starts = [e for e in events if e["type"] == "content_block_start"]
        tool_start = next(e for e in starts if e["content_block"]["type"] == "tool_use")
        self.assertEqual(tool_start["content_block"]["id"], "call_abc")
        self.assertEqual(tool_start["content_block"]["name"], "search")
        args = "".join(
            e["delta"]["partial_json"] for e in events
            if e["type"] == "content_block_delta"
            and e["delta"].get("type") == "input_json_delta"
        )
        self.assertEqual(json.loads(args), {"q": "hi"})
        msg_delta = next(e for e in events if e["type"] == "message_delta")
        self.assertEqual(msg_delta["delta"]["stop_reason"], "tool_use")

    async def test_thinking_stream(self):
        events = await self._collect([
            self._chunk({"thinking": {"content": "Let me think..."}}),
            self._chunk({"thinking": {"content": " more"}}),
            self._chunk({"thinking": {"signature": "sig_abc"}}),
            self._chunk({"content": "Answer"}),
            self._chunk({}, finish_reason="stop"),
            b"data: [DONE]",
        ])
        block_types = [e["content_block"]["type"]
                       for e in events if e["type"] == "content_block_start"]
        self.assertIn("thinking", block_types)
        self.assertIn("text", block_types)
        thinking_text = "".join(
            e["delta"]["thinking"] for e in events
            if e["type"] == "content_block_delta"
            and e["delta"].get("type") == "thinking_delta"
        )
        self.assertIn("Let me think", thinking_text)
        sigs = [e for e in events
                if e["type"] == "content_block_delta"
                and e["delta"].get("type") == "signature_delta"]
        self.assertEqual(sigs[0]["delta"]["signature"], "sig_abc")

    async def test_reasoning_content_stream(self):
        """Third-party provider format: delta.reasoning_content instead of delta.thinking."""
        events = await self._collect([
            self._chunk({"reasoning_content": "thinking step"}),
            self._chunk({"content": "final answer"}),
            self._chunk({}, finish_reason="stop"),
            b"data: [DONE]",
        ])
        block_types = [e["content_block"]["type"]
                       for e in events if e["type"] == "content_block_start"]
        self.assertIn("thinking", block_types)
        self.assertIn("text", block_types)

    async def test_empty_stream(self):
        events = await self._collect([
            self._chunk({}, finish_reason="stop",
                        usage={"prompt_tokens": 5, "completion_tokens": 0}),
            b"data: [DONE]",
        ])
        types = [e["type"] for e in events]
        self.assertIn("message_start", types)
        self.assertIn("message_stop", types)

    async def test_max_tokens_stop_reason(self):
        events = await self._collect([
            self._chunk({"content": "truncated"}),
            self._chunk({}, finish_reason="length"),
            b"data: [DONE]",
        ])
        msg_delta = next(e for e in events if e["type"] == "message_delta")
        self.assertEqual(msg_delta["delta"]["stop_reason"], "max_tokens")

    async def test_stream_protocol_order(self):
        events = await self._collect([
            self._chunk({"content": "hi"}),
            self._chunk({}, finish_reason="stop",
                        usage={"prompt_tokens": 5, "completion_tokens": 2}),
            b"data: [DONE]",
        ])
        types = [e["type"] for e in events]
        self.assertEqual(types[0], "message_start")
        self.assertLess(types.index("content_block_start"), types.index("content_block_stop"))
        self.assertLess(types.index("content_block_stop"), types.index("message_delta"))
        self.assertEqual(types[-1], "message_stop")

    async def test_stream_usage_fields(self):
        events = await self._collect([
            self._chunk({"content": "hi"}),
            self._chunk({}, finish_reason="stop",
                        usage={"prompt_tokens": 100, "completion_tokens": 42,
                               "prompt_tokens_details": {
                                   "cached_tokens": 20, "cache_creation_tokens": 10}}),
            b"data: [DONE]",
        ])
        msg_delta = next(e for e in events if e["type"] == "message_delta")
        u = msg_delta["usage"]
        self.assertEqual(u["input_tokens"], 70)   # 100 - 20 - 10
        self.assertEqual(u["output_tokens"], 42)
        self.assertEqual(u["cache_read_input_tokens"], 20)
        self.assertEqual(u["cache_creation_input_tokens"], 10)

    async def test_malformed_json_line_skipped(self):
        events = await self._collect([
            b"data: not-json",
            self._chunk({"content": "ok"}),
            self._chunk({}, finish_reason="stop"),
            b"data: [DONE]",
        ])
        types = [e["type"] for e in events]
        self.assertIn("message_stop", types)


# ============================================================================
# Unit tests — config + apply_provider_params
# ============================================================================

class TestConfig(unittest.TestCase):

    def test_resolve_route_default(self):
        cfg = {"Router": {"default": "myprovider,mymodel"}}
        self.assertEqual(resolve_route(cfg), ("myprovider", "mymodel"))

    def test_resolve_route_scenario(self):
        cfg = {"Router": {"default": "p,m1", "think": "p,m2"}}
        self.assertEqual(resolve_route(cfg, "think"), ("p", "m2"))

    def test_resolve_route_falls_back_to_default(self):
        cfg = {"Router": {"default": "p,m1"}}
        self.assertEqual(resolve_route(cfg, "missing"), ("p", "m1"))

    def test_resolve_route_no_router(self):
        self.assertIsNone(resolve_route({}))

    def test_get_provider_found(self):
        cfg = {"Providers": [{"name": "foo", "api_key": "k"}]}
        p = get_provider(cfg, "foo")
        self.assertIsNotNone(p)
        self.assertEqual(p["api_key"], "k")

    def test_get_provider_missing(self):
        self.assertIsNone(get_provider({"Providers": []}, "nothere"))


class TestApplyProviderParams(unittest.TestCase):

    def _prov(self, params):
        return {"params": params}

    def test_temperature_default(self):
        req = {"model": "m", "messages": []}
        out = apply_provider_params(self._prov({"temperature": 0.7}), req)
        self.assertEqual(out["temperature"], 0.7)

    def test_temperature_not_overridden(self):
        req = {"model": "m", "messages": [], "temperature": 1.0}
        out = apply_provider_params(self._prov({"temperature": 0.7}), req)
        self.assertEqual(out["temperature"], 1.0)

    def test_top_p_default(self):
        req = {"model": "m", "messages": []}
        out = apply_provider_params(self._prov({"top_p": 0.9}), req)
        self.assertEqual(out["top_p"], 0.9)

    def test_max_tokens_default(self):
        req = {"model": "m", "messages": []}
        out = apply_provider_params(self._prov({"max_tokens": 4096}), req)
        self.assertEqual(out["max_tokens"], 4096)

    def test_max_tokens_caps_request(self):
        req = {"model": "m", "messages": [], "max_tokens": 16384}
        out = apply_provider_params(self._prov({"max_tokens": 4096}), req)
        self.assertEqual(out["max_tokens"], 4096)

    def test_max_tokens_respects_smaller_request(self):
        req = {"model": "m", "messages": [], "max_tokens": 1024}
        out = apply_provider_params(self._prov({"max_tokens": 4096}), req)
        self.assertEqual(out["max_tokens"], 1024)

    def test_reasoning_injects_thinking(self):
        req = {"model": "m", "messages": []}
        out = apply_provider_params(
            self._prov({"reasoning": {"budget_tokens": 5000}}), req)
        self.assertEqual(out["thinking"]["type"], "enabled")
        self.assertEqual(out["thinking"]["budget_tokens"], 5000)

    def test_reasoning_does_not_override_existing_thinking(self):
        req = {"model": "m", "messages": [],
               "thinking": {"type": "enabled", "budget_tokens": 1000}}
        out = apply_provider_params(
            self._prov({"reasoning": {"budget_tokens": 5000}}), req)
        self.assertEqual(out["thinking"]["budget_tokens"], 1000)

    def test_empty_params_noop(self):
        req = {"model": "m", "messages": [], "temperature": 0.5}
        out = apply_provider_params({"params": {}}, req)
        self.assertEqual(out["temperature"], 0.5)

    def test_no_params_key_noop(self):
        req = {"model": "m", "messages": [], "temperature": 0.5}
        out = apply_provider_params({}, req)
        self.assertEqual(out["temperature"], 0.5)


# ============================================================================
# Integration tests — in-process ASGI client against live provider
# ============================================================================

def _check_provider_reachable():
    """Return True if PROVIDER_URL host:port is TCP-connectable."""
    import socket
    from urllib.parse import urlparse
    p = urlparse(PROVIDER_URL)
    host = p.hostname
    port = p.port or (443 if p.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def _requires_provider(test):
    """Skip decorator: skip if provider is unreachable."""
    import functools
    @functools.wraps(test)
    async def wrapper(self, *a, **kw):
        if not _check_provider_reachable():
            self.skipTest(f"Provider not reachable: {PROVIDER_URL}")
        return await test(self, *a, **kw)
    return wrapper


class IntegrationBase(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        import server as srv_mod
        srv_mod.set_config({
            "API_TIMEOUT_MS": 60000,
            "Providers": [{
                "name": "test",
                "api_base_url": PROVIDER_URL,
                "api_key": PROVIDER_KEY,
                "max_retries": 1,
            }],
            "Router": {"default": f"test,{PROVIDER_MODEL}"},
        })
        from httpx import ASGITransport, AsyncClient
        self.client = AsyncClient(
            transport=ASGITransport(app=srv_mod.app),
            base_url="http://test",
            timeout=60.0,
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    def _req(self, extra=None):
        r = {"model": "ignored", "max_tokens": 128,
             "messages": [{"role": "user", "content": "Reply with exactly: PONG"}]}
        if extra:
            r.update(extra)
        return r

    async def _stream_events(self, req):
        resp = await self.client.post(
            "/v1/messages", json={**req, "stream": True},
            headers={"Accept": "text/event-stream"})
        events = []
        for line in resp.text.splitlines():
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass
        return events, resp.status_code


class TestIntegrationNonStreaming(IntegrationBase):

    @_requires_provider
    async def test_basic_response(self):
        resp = await self.client.post("/v1/messages", json=self._req())
        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()
        self.assertEqual(data["type"], "message")
        self.assertEqual(data["role"], "assistant")
        text = " ".join(b.get("text", "") for b in data["content"] if b["type"] == "text")
        self.assertIn("PONG", text)

    @_requires_provider
    async def test_schema_fields(self):
        data = (await self.client.post("/v1/messages", json=self._req())).json()
        for f in ("id", "type", "role", "model", "content", "stop_reason", "usage"):
            self.assertIn(f, data)
        self.assertIn("input_tokens", data["usage"])
        self.assertIn("output_tokens", data["usage"])
        self.assertIn("cache_read_input_tokens", data["usage"])
        self.assertIn("cache_creation_input_tokens", data["usage"])
        self.assertGreater(data["usage"]["input_tokens"], 0)
        self.assertGreater(data["usage"]["output_tokens"], 0)

    @_requires_provider
    async def test_system_prompt(self):
        req = self._req({"system": "Always reply: HELLO_WORLD",
                         "messages": [{"role": "user", "content": "Hi"}]})
        data = (await self.client.post("/v1/messages", json=req)).json()
        text = " ".join(b.get("text", "") for b in data["content"] if b["type"] == "text")
        self.assertIn("HELLO_WORLD", text)

    @_requires_provider
    async def test_sampling_params(self):
        req = self._req({"temperature": 0.0, "top_p": 1.0})
        resp = await self.client.post("/v1/messages", json=req)
        self.assertEqual(resp.status_code, 200)

    @_requires_provider
    async def test_stop_sequences(self):
        req = {"model": "ignored", "max_tokens": 64,
               "stop_sequences": ["STOP"],
               "messages": [{"role": "user", "content": "Count: 1 2 3 STOP 4 5"}]}
        resp = await self.client.post("/v1/messages", json=req)
        self.assertEqual(resp.status_code, 200)

    @_requires_provider
    async def test_multi_turn(self):
        req = {"model": "ignored", "max_tokens": 128, "messages": [
            {"role": "user", "content": "Say: FIRST"},
            {"role": "assistant", "content": "FIRST"},
            {"role": "user", "content": "Now say: SECOND"},
        ]}
        data = (await self.client.post("/v1/messages", json=req)).json()
        text = " ".join(b.get("text", "") for b in data["content"] if b["type"] == "text")
        self.assertIn("SECOND", text)

    @_requires_provider
    async def test_tool_use(self):
        req = {"model": "ignored", "max_tokens": 256,
               "tools": [{"name": "get_number", "description": "Returns a number",
                           "input_schema": {"type": "object",
                                            "properties": {"n": {"type": "integer"}},
                                            "required": ["n"]}}],
               "tool_choice": {"type": "any"},
               "messages": [{"role": "user", "content": "Call get_number with n=42"}]}
        data = (await self.client.post("/v1/messages", json=req)).json()
        tool_blocks = [b for b in data["content"] if b["type"] == "tool_use"]
        self.assertTrue(len(tool_blocks) > 0, f"No tool_use block in: {data['content']}")
        self.assertEqual(tool_blocks[0]["name"], "get_number")
        self.assertEqual(tool_blocks[0]["input"].get("n"), 42)

    @_requires_provider
    async def test_tool_choice_none(self):
        req = {"model": "ignored", "max_tokens": 128,
               "tools": [{"name": "fn", "description": "A tool",
                           "input_schema": {"type": "object"}}],
               "tool_choice": {"type": "none"},
               "messages": [{"role": "user", "content": "Hello"}]}
        resp = await self.client.post("/v1/messages", json=req)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        tool_blocks = [b for b in data["content"] if b["type"] == "tool_use"]
        self.assertEqual(len(tool_blocks), 0)

    @_requires_provider
    async def test_health(self):
        resp = await self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")

    @_requires_provider
    async def test_reasoning_effort(self):
        req = self._req({"output_config": {"effort": "low"}})
        resp = await self.client.post("/v1/messages", json=req)
        # Provider may or may not support reasoning_effort — 400 is acceptable
        self.assertIn(resp.status_code, (200, 400), resp.text)

    @_requires_provider
    async def test_metadata_user_id(self):
        req = self._req({"metadata": {"user_id": "test-user-001"}})
        resp = await self.client.post("/v1/messages", json=req)
        self.assertEqual(resp.status_code, 200)

    @_requires_provider
    async def test_output_config_json_schema(self):
        req = {"model": "ignored", "max_tokens": 128,
               "output_config": {"format": {"type": "json_schema",
                                             "schema": {"type": "object",
                                                        "properties": {"result": {"type": "string"}}}}},
               "messages": [{"role": "user", "content": "Return JSON with result='ok'"}]}
        resp = await self.client.post("/v1/messages", json=req)
        self.assertIn(resp.status_code, (200, 400))


class TestIntegrationStreaming(IntegrationBase):

    @_requires_provider
    async def test_basic_streaming(self):
        events, status = await self._stream_events(self._req())
        self.assertEqual(status, 200)
        types = [e["type"] for e in events]
        for t in ("message_start", "content_block_start", "content_block_delta",
                  "content_block_stop", "message_delta", "message_stop"):
            self.assertIn(t, types)

    @_requires_provider
    async def test_stream_text_content(self):
        events, status = await self._stream_events(self._req())
        self.assertEqual(status, 200)
        text = "".join(
            e["delta"]["text"] for e in events
            if e["type"] == "content_block_delta"
            and e["delta"].get("type") == "text_delta"
        )
        self.assertIn("PONG", text)

    @_requires_provider
    async def test_stream_usage(self):
        events, status = await self._stream_events(self._req())
        self.assertEqual(status, 200)
        msg_delta = next(e for e in events if e["type"] == "message_delta")
        self.assertIn("usage", msg_delta)
        self.assertIn("input_tokens", msg_delta["usage"])
        self.assertIn("output_tokens", msg_delta["usage"])
        self.assertIn("cache_read_input_tokens", msg_delta["usage"])
        self.assertIn("cache_creation_input_tokens", msg_delta["usage"])
        self.assertGreater(msg_delta["usage"]["output_tokens"], 0)

    @_requires_provider
    async def test_stream_stop_reason(self):
        events, status = await self._stream_events(self._req())
        msg_delta = next(e for e in events if e["type"] == "message_delta")
        self.assertEqual(msg_delta["delta"]["stop_reason"], "end_turn")

    @_requires_provider
    async def test_stream_protocol_order(self):
        events, status = await self._stream_events(self._req())
        self.assertEqual(status, 200)
        types = [e["type"] for e in events]
        self.assertEqual(types[0], "message_start")
        self.assertLess(types.index("content_block_start"), types.index("content_block_stop"))
        self.assertLess(types.index("content_block_stop"), types.index("message_delta"))
        self.assertEqual(types[-1], "message_stop")

    @_requires_provider
    async def test_stream_tool_use(self):
        req = {"model": "ignored", "max_tokens": 256,
               "tools": [{"name": "get_number", "description": "Returns a number",
                           "input_schema": {"type": "object",
                                            "properties": {"n": {"type": "integer"}},
                                            "required": ["n"]}}],
               "tool_choice": {"type": "any"},
               "messages": [{"role": "user", "content": "Call get_number with n=7"}]}
        events, status = await self._stream_events(req)
        self.assertEqual(status, 200)
        tool_starts = [e for e in events
                       if e["type"] == "content_block_start"
                       and e["content_block"]["type"] == "tool_use"]
        self.assertTrue(len(tool_starts) > 0)
        args = "".join(
            e["delta"]["partial_json"] for e in events
            if e["type"] == "content_block_delta"
            and e["delta"].get("type") == "input_json_delta"
        )
        self.assertEqual(json.loads(args).get("n"), 7)

    @_requires_provider
    async def test_stream_multi_turn(self):
        req = {"model": "ignored", "max_tokens": 128, "messages": [
            {"role": "user", "content": "Say: FIRST"},
            {"role": "assistant", "content": "FIRST"},
            {"role": "user", "content": "Now say: SECOND"},
        ]}
        events, status = await self._stream_events(req)
        self.assertEqual(status, 200)
        text = "".join(
            e["delta"]["text"] for e in events
            if e["type"] == "content_block_delta"
            and e["delta"].get("type") == "text_delta"
        )
        self.assertIn("SECOND", text)


# ============================================================================
# Runner
# ============================================================================

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    if mode in ("unit", "all"):
        for cls in (TestAnthropicToOpenAI, TestOpenAIToAnthropic,
                    TestStreamConverter, TestConfig, TestApplyProviderParams):
            suite.addTests(loader.loadTestsFromTestCase(cls))

    if mode in ("integration", "all"):
        for cls in (TestIntegrationNonStreaming, TestIntegrationStreaming):
            suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
