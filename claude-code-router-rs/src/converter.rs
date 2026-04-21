use serde_json::{json, Value};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Anthropic → OpenAI (request)
// ---------------------------------------------------------------------------

pub fn anthropic_to_openai(req: &Value) -> Value {
    let mut messages: Vec<Value> = Vec::new();

    // System prompt
    if let Some(system) = req.get("system") {
        let text = if let Some(arr) = system.as_array() {
            arr.iter()
                .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"))
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            system.as_str().unwrap_or("").to_string()
        };
        if !text.is_empty() {
            messages.push(json!({"role": "system", "content": text}));
        }
    }

    // Conversation messages
    if let Some(msgs) = req.get("messages").and_then(|m| m.as_array()) {
        for msg in msgs {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            let content = &msg["content"];

            if let Some(text) = content.as_str() {
                messages.push(json!({"role": role, "content": text}));
                continue;
            }

            if let Some(blocks) = content.as_array() {
                if role == "user" {
                    let mut user_parts: Vec<Value> = Vec::new();
                    let mut tool_results: Vec<&Value> = Vec::new();

                    for block in blocks {
                        match block.get("type").and_then(|t| t.as_str()) {
                            Some("text") => {
                                user_parts.push(json!({
                                    "type": "text",
                                    "text": block.get("text").and_then(|t| t.as_str()).unwrap_or("")
                                }));
                            }
                            Some("image") => {
                                let src = block.get("source").cloned().unwrap_or(json!({}));
                                let url = if src.get("type").and_then(|t| t.as_str()) == Some("base64") {
                                    format!(
                                        "data:{};base64,{}",
                                        src.get("media_type").and_then(|m| m.as_str()).unwrap_or(""),
                                        src.get("data").and_then(|d| d.as_str()).unwrap_or("")
                                    )
                                } else {
                                    src.get("url").and_then(|u| u.as_str()).unwrap_or("").to_string()
                                };
                                user_parts.push(json!({"type": "image_url", "image_url": {"url": url}}));
                            }
                            Some("tool_result") => {
                                tool_results.push(block);
                            }
                            _ => {}
                        }
                    }

                    if !user_parts.is_empty() {
                        if user_parts.iter().all(|p| p.get("type").and_then(|t| t.as_str()) == Some("text")) {
                            let combined: String = user_parts
                                .iter()
                                .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                                .collect::<Vec<_>>()
                                .join("");
                            messages.push(json!({"role": "user", "content": combined}));
                        } else {
                            messages.push(json!({"role": "user", "content": user_parts}));
                        }
                    }

                    for tr in tool_results {
                        let tr_content = match tr.get("content") {
                            Some(Value::Array(arr)) => arr
                                .iter()
                                .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"))
                                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                                .collect::<Vec<_>>()
                                .join("\n"),
                            Some(Value::String(s)) => s.clone(),
                            _ => String::new(),
                        };
                        messages.push(json!({
                            "role": "tool",
                            "tool_call_id": tr.get("tool_use_id").and_then(|id| id.as_str()).unwrap_or(""),
                            "content": tr_content
                        }));
                    }
                } else if role == "assistant" {
                    let mut text_parts: Vec<String> = Vec::new();
                    let mut tool_calls: Vec<Value> = Vec::new();

                    for block in blocks {
                        match block.get("type").and_then(|t| t.as_str()) {
                            Some("text") => {
                                text_parts.push(
                                    block.get("text").and_then(|t| t.as_str()).unwrap_or("").to_string(),
                                );
                            }
                            Some("tool_use") => {
                                tool_calls.push(json!({
                                    "id": block.get("id").and_then(|id| id.as_str()).unwrap_or(""),
                                    "type": "function",
                                    "function": {
                                        "name": block.get("name").and_then(|n| n.as_str()).unwrap_or(""),
                                        "arguments": serde_json::to_string(
                                            block.get("input").unwrap_or(&json!({}))
                                        ).unwrap_or_else(|_| "{}".to_string())
                                    }
                                }));
                            }
                            // thinking blocks are internal — skip
                            _ => {}
                        }
                    }

                    let mut msg_obj = json!({"role": "assistant"});
                    msg_obj["content"] = if text_parts.is_empty() {
                        json!("")
                    } else {
                        json!(text_parts.join("\n"))
                    };
                    if !tool_calls.is_empty() {
                        msg_obj["tool_calls"] = json!(tool_calls);
                    }
                    messages.push(msg_obj);
                }
            }
        }
    }

    let stream = req.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);

    let mut openai_req = json!({
        "model": req.get("model").and_then(|m| m.as_str()).unwrap_or(""),
        "messages": messages,
        "stream": stream,
    });

    if stream {
        openai_req["stream_options"] = json!({"include_usage": true});
    }

    // Sampling parameters
    for field in &["max_tokens", "temperature", "top_p", "top_k"] {
        if let Some(v) = req.get(*field) {
            if !v.is_null() {
                openai_req[*field] = v.clone();
            }
        }
    }

    // Stop sequences
    if let Some(stops) = req.get("stop_sequences") {
        if !stops.is_null() {
            openai_req["stop"] = stops.clone();
        }
    }

    // Tools
    if let Some(tools) = req.get("tools").and_then(|t| t.as_array()) {
        let known = ["name", "description", "input_schema", "strict", "type", "cache_control"];
        let mut oai_tools = Vec::new();
        for t in tools {
            let mut fn_obj = json!({
                "name": t.get("name").and_then(|n| n.as_str()).unwrap_or(""),
                "description": t.get("description").and_then(|d| d.as_str()).unwrap_or(""),
                "parameters": t.get("input_schema").cloned().unwrap_or(json!({})),
            });
            if let Some(strict) = t.get("strict") {
                fn_obj["strict"] = strict.clone();
            }
            if let Some(obj) = t.as_object() {
                for (k, v) in obj {
                    if !known.contains(&k.as_str()) {
                        fn_obj[k] = v.clone();
                    }
                }
            }
            oai_tools.push(json!({"type": "function", "function": fn_obj}));
        }
        openai_req["tools"] = json!(oai_tools);
    }

    // Tool choice
    if let Some(tc) = req.get("tool_choice") {
        if let Some(tc_type) = tc.get("type").and_then(|t| t.as_str()) {
            match tc_type {
                "auto" => { openai_req["tool_choice"] = json!("auto"); }
                "any" => { openai_req["tool_choice"] = json!("required"); }
                "tool" => {
                    openai_req["tool_choice"] = json!({
                        "type": "function",
                        "function": {"name": tc.get("name").and_then(|n| n.as_str()).unwrap_or("")}
                    });
                }
                _ => {}
            }
        }
        if tc.get("disable_parallel_tool_use").and_then(|v| v.as_bool()) == Some(true) {
            openai_req["parallel_tool_calls"] = json!(false);
        }
    }

    // Thinking
    if let Some(thinking) = req.get("thinking") {
        if let Some(t_type) = thinking.get("type").and_then(|t| t.as_str()) {
            if t_type == "enabled" || t_type == "adaptive" {
                openai_req["thinking"] = thinking.clone();
            }
        }
    }

    // output_config
    if let Some(oc) = req.get("output_config") {
        if let Some(effort) = oc.get("effort").and_then(|e| e.as_str()) {
            openai_req["reasoning_effort"] = json!(if effort == "max" { "xhigh" } else { effort });
        }
        if let Some(fmt) = oc.get("format") {
            if fmt.get("type").and_then(|t| t.as_str()) == Some("json_schema") {
                openai_req["response_format"] = json!({
                    "type": "json_schema",
                    "json_schema": fmt.get("schema").cloned().unwrap_or(json!({}))
                });
            }
        }
    }

    // metadata.user_id
    if let Some(user_id) = req
        .get("metadata")
        .and_then(|m| m.get("user_id"))
        .and_then(|u| u.as_str())
    {
        openai_req["user"] = json!(user_id);
    }

    openai_req
}

// ---------------------------------------------------------------------------
// OpenAI → Anthropic (non-streaming response)
// ---------------------------------------------------------------------------

fn finish_to_stop(finish: &str) -> &str {
    match finish {
        "stop" => "end_turn",
        "length" => "max_tokens",
        "tool_calls" => "tool_use",
        "content_filter" => "stop_sequence",
        _ => "end_turn",
    }
}

pub fn openai_to_anthropic(resp: &Value, original_model: &str) -> Value {
    let choice = &resp["choices"][0];
    let message = &choice["message"];
    let finish_reason = choice
        .get("finish_reason")
        .and_then(|f| f.as_str())
        .unwrap_or("stop");
    let stop_reason = finish_to_stop(finish_reason);

    let mut content: Vec<Value> = Vec::new();

    // Thinking
    let thinking_obj = message.get("thinking");
    let reasoning = message.get("reasoning_content").and_then(|r| r.as_str());
    if let Some(t) = thinking_obj {
        if let Some(tc) = t.get("content").and_then(|c| c.as_str()) {
            if !tc.is_empty() {
                let mut block = json!({"type": "thinking", "thinking": tc});
                if let Some(sig) = t.get("signature") {
                    block["signature"] = sig.clone();
                }
                content.push(block);
            }
        }
    } else if let Some(r) = reasoning {
        if !r.is_empty() {
            content.push(json!({"type": "thinking", "thinking": r}));
        }
    }

    // Text
    if let Some(text) = message.get("content").and_then(|c| c.as_str()) {
        if !text.is_empty() {
            content.push(json!({"type": "text", "text": text}));
        }
    }

    // Tool calls
    if let Some(tcs) = message.get("tool_calls").and_then(|t| t.as_array()) {
        for tc in tcs {
            let fn_obj = tc.get("function").unwrap_or(tc);
            let args_str = fn_obj
                .get("arguments")
                .and_then(|a| a.as_str())
                .unwrap_or("{}");
            let input_obj = serde_json::from_str::<Value>(args_str)
                .unwrap_or_else(|_| json!({"_raw": args_str}));
            let id = tc
                .get("id")
                .and_then(|id| id.as_str())
                .unwrap_or("")
                .to_string();
            let id = if id.is_empty() {
                format!("toolu_{}", &Uuid::new_v4().to_string().replace("-", "")[..24])
            } else {
                id
            };
            content.push(json!({
                "type": "tool_use",
                "id": id,
                "name": fn_obj.get("name").and_then(|n| n.as_str()).unwrap_or(""),
                "input": input_obj,
            }));
        }
    }

    let usage = resp.get("usage").cloned().unwrap_or(json!({}));
    let details = usage
        .get("prompt_tokens_details")
        .cloned()
        .unwrap_or(json!({}));
    let cache_read = details
        .get("cached_tokens")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let cache_created = details
        .get("cache_creation_tokens")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let prompt_tokens = usage.get("prompt_tokens").and_then(|v| v.as_i64()).unwrap_or(0);
    let completion_tokens = usage.get("completion_tokens").and_then(|v| v.as_i64()).unwrap_or(0);

    let msg_id = resp
        .get("id")
        .and_then(|id| id.as_str())
        .unwrap_or("")
        .to_string();
    let msg_id = if msg_id.is_empty() {
        format!("msg_{}", &Uuid::new_v4().to_string().replace("-", "")[..24])
    } else {
        msg_id
    };

    json!({
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": original_model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": null,
        "usage": {
            "input_tokens": (prompt_tokens - cache_read - cache_created).max(0),
            "output_tokens": completion_tokens,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_created,
        }
    })
}
