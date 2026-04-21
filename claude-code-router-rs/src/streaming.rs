use bytes::Bytes;
use futures::Stream;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::pin::Pin;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

fn sse(event: &str, data: &Value) -> String {
    format!(
        "event: {}\ndata: {}\n\n",
        event,
        serde_json::to_string(data).unwrap_or_else(|_| "{}".to_string())
    )
}

fn finish_to_stop(finish: &str) -> &str {
    match finish {
        "stop" => "end_turn",
        "length" => "max_tokens",
        "tool_calls" => "tool_use",
        "content_filter" => "stop_sequence",
        _ => "end_turn",
    }
}

/// Convert an OpenAI streaming response into Anthropic SSE events.
///
/// Takes a byte stream from the OpenAI provider and produces a stream of
/// Anthropic-formatted SSE event strings.
pub fn stream_openai_to_anthropic(
    byte_stream: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    message_id: String,
    model: String,
) -> ReceiverStream<Result<String, std::io::Error>> {
    let (tx, rx) = mpsc::channel::<Result<String, std::io::Error>>(256);

    tokio::spawn(async move {
        if let Err(_) = run_streaming_converter(byte_stream, &tx, &message_id, &model).await {
            // Channel closed, receiver dropped — nothing to do
        }
    });

    ReceiverStream::new(rx)
}

async fn run_streaming_converter(
    byte_stream: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    tx: &mpsc::Sender<Result<String, std::io::Error>>,
    message_id: &str,
    model: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use tokio_stream::StreamExt;

    let mut content_index: usize = 0;
    let mut text_idx: i32 = -1;
    let mut thinking_idx: i32 = -1;
    let mut thinking_sig_sent = false;
    let mut tool_blocks: HashMap<u64, ToolBlockState> = HashMap::new();
    let mut last_tool_idx: Option<u64> = None;
    let mut stop_reason = "end_turn".to_string();
    let mut usage = json!({});

    // message_start
    tx.send(Ok(sse("message_start", &json!({
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": null,
            "stop_sequence": null,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
    })))).await?;

    tx.send(Ok(sse("ping", &json!({"type": "ping"})))).await?;

    // Buffer for incomplete SSE lines across chunks
    let mut buffer = String::new();

    let mut pinned = byte_stream;

    while let Some(chunk_result) = pinned.next().await {
        let chunk = match chunk_result {
            Ok(c) => c,
            Err(e) => {
                let err_event = json!({
                    "type": "error",
                    "error": {"type": "api_error", "message": e.to_string()}
                });
                let _ = tx.send(Ok(sse("error", &err_event))).await;
                break;
            }
        };

        buffer.push_str(&String::from_utf8_lossy(&chunk));

        // Process complete lines
        while let Some(newline_pos) = buffer.find('\n') {
            let line = buffer[..newline_pos].to_string();
            buffer = buffer[newline_pos + 1..].to_string();

            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if !line.starts_with("data: ") {
                continue;
            }
            let payload = &line[6..];
            if payload == "[DONE]" {
                // End of stream
                break;
            }

            let chunk: Value = match serde_json::from_str(payload) {
                Ok(v) => v,
                Err(_) => continue,
            };

            // Accumulate usage
            if let Some(u) = chunk.get("usage") {
                if !u.is_null() {
                    usage = u.clone();
                }
            }

            let choices = match chunk.get("choices").and_then(|c| c.as_array()) {
                Some(c) if !c.is_empty() => c,
                _ => continue,
            };

            let choice = &choices[0];
            let empty_obj = json!({});
            let delta = choice.get("delta").unwrap_or(&empty_obj);
            let finish_reason = choice.get("finish_reason").and_then(|f| f.as_str());

            // ---- thinking delta ----
            let thinking_delta = delta.get("thinking");
            let reasoning_content = delta.get("reasoning_content").and_then(|r| r.as_str());

            let has_thinking = thinking_delta
                .map(|t| !t.is_null())
                .unwrap_or(false);
            let has_reasoning = reasoning_content.map(|r| !r.is_empty()).unwrap_or(false);

            if has_thinking || has_reasoning {
                let (t_content, t_sig) = if let Some(td) = thinking_delta {
                    (
                        td.get("content").and_then(|c| c.as_str()).unwrap_or(""),
                        td.get("signature").and_then(|s| s.as_str()),
                    )
                } else {
                    (reasoning_content.unwrap_or(""), None)
                };

                if !t_content.is_empty() || t_sig.is_some() {
                    if thinking_idx == -1 {
                        thinking_idx = content_index as i32;
                        content_index += 1;
                        thinking_sig_sent = false;
                        tx.send(Ok(sse("content_block_start", &json!({
                            "type": "content_block_start",
                            "index": thinking_idx,
                            "content_block": {"type": "thinking", "thinking": ""},
                        })))).await?;
                    }

                    if !t_content.is_empty() {
                        tx.send(Ok(sse("content_block_delta", &json!({
                            "type": "content_block_delta",
                            "index": thinking_idx,
                            "delta": {"type": "thinking_delta", "thinking": t_content},
                        })))).await?;
                    }

                    if let Some(sig) = t_sig {
                        if !thinking_sig_sent {
                            thinking_sig_sent = true;
                            tx.send(Ok(sse("content_block_delta", &json!({
                                "type": "content_block_delta",
                                "index": thinking_idx,
                                "delta": {"type": "signature_delta", "signature": sig},
                            })))).await?;
                            tx.send(Ok(sse("content_block_stop", &json!({
                                "type": "content_block_stop",
                                "index": thinking_idx,
                            })))).await?;
                            thinking_idx = -1;
                        }
                    }
                }
            }

            // ---- text delta ----
            if let Some(text_chunk) = delta.get("content").and_then(|c| c.as_str()) {
                if !text_chunk.is_empty() {
                    if text_idx == -1 {
                        // Close thinking if still open
                        if thinking_idx != -1 {
                            tx.send(Ok(sse("content_block_stop", &json!({
                                "type": "content_block_stop",
                                "index": thinking_idx,
                            })))).await?;
                            thinking_idx = -1;
                        }
                        text_idx = content_index as i32;
                        content_index += 1;
                        tx.send(Ok(sse("content_block_start", &json!({
                            "type": "content_block_start",
                            "index": text_idx,
                            "content_block": {"type": "text", "text": ""},
                        })))).await?;
                    }
                    tx.send(Ok(sse("content_block_delta", &json!({
                        "type": "content_block_delta",
                        "index": text_idx,
                        "delta": {"type": "text_delta", "text": text_chunk},
                    })))).await?;
                }
            }

            // ---- tool call deltas ----
            if let Some(tc_deltas) = delta.get("tool_calls").and_then(|t| t.as_array()) {
                for tc_delta in tc_deltas {
                    let tc_idx = tc_delta.get("index").and_then(|i| i.as_u64()).unwrap_or(0);

                    if !tool_blocks.contains_key(&tc_idx) {
                        // Close text block
                        if text_idx != -1 {
                            tx.send(Ok(sse("content_block_stop", &json!({
                                "type": "content_block_stop",
                                "index": text_idx,
                            })))).await?;
                            text_idx = -1;
                        }
                        // Close previous tool block
                        if let Some(prev_idx) = last_tool_idx {
                            if let Some(prev) = tool_blocks.get_mut(&prev_idx) {
                                if !prev.closed {
                                    tx.send(Ok(sse("content_block_stop", &json!({
                                        "type": "content_block_stop",
                                        "index": prev.block_idx,
                                    })))).await?;
                                    prev.closed = true;
                                }
                            }
                        }

                        let block_idx = content_index;
                        content_index += 1;
                        let tc_id = tc_delta
                            .get("id")
                            .and_then(|id| id.as_str())
                            .unwrap_or("")
                            .to_string();
                        let tc_id = if tc_id.is_empty() {
                            format!("toolu_{}", &Uuid::new_v4().to_string().replace("-", "")[..24])
                        } else {
                            tc_id
                        };
                        let tc_fn = tc_delta.get("function").unwrap_or(tc_delta);
                        let tc_name = tc_fn
                            .get("name")
                            .and_then(|n| n.as_str())
                            .unwrap_or("")
                            .to_string();

                        tool_blocks.insert(tc_idx, ToolBlockState {
                            block_idx,
                            id: tc_id.clone(),
                            name: tc_name.clone(),
                            closed: false,
                        });
                        last_tool_idx = Some(tc_idx);

                        tx.send(Ok(sse("content_block_start", &json!({
                            "type": "content_block_start",
                            "index": block_idx,
                            "content_block": {
                                "type": "tool_use",
                                "id": tc_id,
                                "name": tc_name,
                                "input": {},
                            },
                        })))).await?;
                    }

                    let tc_fn_d = tc_delta.get("function").unwrap_or(tc_delta);
                    if let Some(args) = tc_fn_d.get("arguments").and_then(|a| a.as_str()) {
                        if !args.is_empty() {
                            let block_idx = tool_blocks[&tc_idx].block_idx;
                            tx.send(Ok(sse("content_block_delta", &json!({
                                "type": "content_block_delta",
                                "index": block_idx,
                                "delta": {"type": "input_json_delta", "partial_json": args},
                            })))).await?;
                        }
                    }
                }
            }

            // ---- finish reason ----
            if let Some(fr) = finish_reason {
                stop_reason = finish_to_stop(fr).to_string();
            }
        }
    }

    // Close any open blocks
    if thinking_idx != -1 {
        let _ = tx.send(Ok(sse("content_block_stop", &json!({
            "type": "content_block_stop",
            "index": thinking_idx,
        })))).await;
    }
    if text_idx != -1 {
        let _ = tx.send(Ok(sse("content_block_stop", &json!({
            "type": "content_block_stop",
            "index": text_idx,
        })))).await;
    }
    for tb in tool_blocks.values() {
        if !tb.closed {
            let _ = tx.send(Ok(sse("content_block_stop", &json!({
                "type": "content_block_stop",
                "index": tb.block_idx,
            })))).await;
        }
    }

    // message_delta + message_stop
    let details = usage.get("prompt_tokens_details").cloned().unwrap_or(json!({}));
    let cache_read = details.get("cached_tokens").and_then(|v| v.as_i64()).unwrap_or(0);
    let cache_created = details.get("cache_creation_tokens").and_then(|v| v.as_i64()).unwrap_or(0);
    let prompt_tokens = usage.get("prompt_tokens").and_then(|v| v.as_i64()).unwrap_or(0);
    let completion_tokens = usage.get("completion_tokens").and_then(|v| v.as_i64()).unwrap_or(0);

    let _ = tx.send(Ok(sse("message_delta", &json!({
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": null},
        "usage": {
            "input_tokens": (prompt_tokens - cache_read - cache_created).max(0),
            "output_tokens": completion_tokens,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_created,
        }
    })))).await;

    let _ = tx.send(Ok(sse("message_stop", &json!({"type": "message_stop"})))).await;

    Ok(())
}

struct ToolBlockState {
    block_idx: usize,
    id: String,
    name: String,
    closed: bool,
}
