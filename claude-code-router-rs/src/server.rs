use axum::{
    body::Body,
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use serde_json::{json, Value};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio_stream::StreamExt;
use tracing::{error, info};
use uuid::Uuid;

use crate::client;
use crate::config::{self, Config, Provider};
use crate::converter;
use crate::streaming;

pub struct AppState {
    pub config: Config,
    pub client: reqwest::Client,
    pub inflight: AtomicU64,
    pub total_requests: AtomicU64,
}

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/messages", post(messages))
        .route("/v1/models", get(list_models))
        .route("/v1/models/{model_id}", get(get_model))
        .route("/health", get(health))
        .route("/stats", get(stats))
        .with_state(state)
}

fn get_route(config: &Config) -> Result<(&Provider, String, String), (StatusCode, String)> {
    let (provider_name, model) = config::resolve_route(config)
        .ok_or((StatusCode::INTERNAL_SERVER_ERROR, "No default route configured".to_string()))?;
    let provider = config::get_provider(config, &provider_name)
        .ok_or((StatusCode::INTERNAL_SERVER_ERROR, format!("Provider '{}' not found", provider_name)))?;
    let url = provider.api_base_url.clone();
    Ok((provider, model, url))
}

fn error_response(status: StatusCode, message: &str) -> Response {
    let body = json!({"type": "error", "error": {"type": "api_error", "message": message}});
    (status, Json(body)).into_response()
}

// POST /v1/messages
async fn messages(
    State(state): State<Arc<AppState>>,
    axum::extract::Json(body): axum::extract::Json<Value>,
) -> Response {
    let inflight = state.inflight.fetch_add(1, Ordering::Relaxed) + 1;
    state.total_requests.fetch_add(1, Ordering::Relaxed);
    info!("[messages] +++ in-flight={} (global)", inflight);

    let result = handle_messages(&state, body).await;

    let inflight = state.inflight.fetch_sub(1, Ordering::Relaxed) - 1;
    info!("[messages] --- in-flight={} (global)", inflight);

    result
}

async fn handle_messages(state: &AppState, mut body: Value) -> Response {
    let (provider, model, url) = match get_route(&state.config) {
        Ok(r) => r,
        Err((status, msg)) => return error_response(status, &msg),
    };

    body["model"] = json!(model);

    let mut openai_req = converter::anthropic_to_openai(&body);
    config::apply_provider_params(provider, &mut openai_req);

    let headers = config::provider_headers(provider);
    let max_retries = provider.max_retries;
    let timeout = state.config.timeout_secs();

    let is_stream = openai_req
        .get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false);

    if is_stream {
        match client::open_stream(&url, &headers, &openai_req, timeout, max_retries).await {
            Ok(byte_stream) => {
                let message_id = format!("msg_{}", &Uuid::new_v4().to_string().replace("-", "")[..24]);
                let event_stream = streaming::stream_openai_to_anthropic(
                    byte_stream,
                    message_id,
                    model.clone(),
                );

                let body = Body::from_stream(event_stream);

                Response::builder()
                    .status(StatusCode::OK)
                    .header(header::CONTENT_TYPE, "text/event-stream")
                    .header(header::CACHE_CONTROL, "no-cache")
                    .header("X-Accel-Buffering", "no")
                    .body(body)
                    .unwrap_or_else(|_| error_response(StatusCode::INTERNAL_SERVER_ERROR, "Failed to build response"))
            }
            Err(e) => {
                let status = if e.status > 0 {
                    StatusCode::from_u16(e.status).unwrap_or(StatusCode::BAD_GATEWAY)
                } else {
                    StatusCode::BAD_GATEWAY
                };
                error_response(status, &e.body)
            }
        }
    } else {
        match client::post_json(&state.client, &url, &headers, &openai_req, timeout, max_retries).await {
            Ok(openai_resp) => {
                let anthropic_resp = converter::openai_to_anthropic(&openai_resp, &model);
                Json(anthropic_resp).into_response()
            }
            Err(e) => {
                error!("Provider error: {}", e);
                let status = if e.status > 0 {
                    StatusCode::from_u16(e.status).unwrap_or(StatusCode::BAD_GATEWAY)
                } else {
                    StatusCode::BAD_GATEWAY
                };
                error_response(status, &e.body)
            }
        }
    }
}

// GET /v1/models
async fn list_models(State(state): State<Arc<AppState>>) -> Response {
    let (provider, _, _) = match get_route(&state.config) {
        Ok(r) => r,
        Err((status, msg)) => return error_response(status, &msg),
    };

    let url = format!("{}/models", config::api_base(provider));
    let headers = config::provider_headers(provider);
    let timeout = state.config.timeout_secs();

    match state
        .client
        .get(&url)
        .timeout(std::time::Duration::from_secs_f64(timeout))
        .header("Content-Type", "application/json")
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            let data: Value = resp.json().await.unwrap_or(json!({"data": []}));
            let models: Vec<Value> = data
                .get("data")
                .and_then(|d| d.as_array())
                .unwrap_or(&vec![])
                .iter()
                .map(|m| openai_model_to_anthropic(m))
                .collect();
            let result = json!({
                "data": models,
                "has_more": data.get("has_more").and_then(|h| h.as_bool()).unwrap_or(false),
                "first_id": models.first().and_then(|m| m.get("id")).cloned().unwrap_or(Value::Null),
                "last_id": models.last().and_then(|m| m.get("id")).cloned().unwrap_or(Value::Null),
            });
            Json(result).into_response()
        }
        Ok(resp) => {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            error_response(
                StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
                &text,
            )
        }
        Err(e) => error_response(StatusCode::BAD_GATEWAY, &e.to_string()),
    }
}

// GET /v1/models/{model_id}
async fn get_model(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(model_id): axum::extract::Path<String>,
) -> Response {
    let (provider, _, _) = match get_route(&state.config) {
        Ok(r) => r,
        Err((status, msg)) => return error_response(status, &msg),
    };

    let url = format!("{}/models/{}", config::api_base(provider), model_id);
    let headers = config::provider_headers(provider);
    let timeout = state.config.timeout_secs();

    match state
        .client
        .get(&url)
        .timeout(std::time::Duration::from_secs_f64(timeout))
        .header("Content-Type", "application/json")
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            let data: Value = resp.json().await.unwrap_or(json!({}));
            Json(openai_model_to_anthropic(&data)).into_response()
        }
        Ok(resp) => {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            error_response(
                StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
                &text,
            )
        }
        Err(e) => error_response(StatusCode::BAD_GATEWAY, &e.to_string()),
    }
}

fn openai_model_to_anthropic(m: &Value) -> Value {
    let ts = m.get("created").and_then(|c| c.as_i64()).unwrap_or(0);
    let created_at = chrono::DateTime::from_timestamp(ts, 0)
        .map(|dt| dt.to_rfc3339())
        .unwrap_or_else(|| "1970-01-01T00:00:00+00:00".to_string());
    let id = m.get("id").and_then(|i| i.as_str()).unwrap_or("");
    json!({
        "type": "model",
        "id": id,
        "display_name": id,
        "created_at": created_at,
    })
}

// GET /health
async fn health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

// GET /stats
async fn stats(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "inflight": state.inflight.load(Ordering::Relaxed),
        "total_requests": state.total_requests.load(Ordering::Relaxed),
    }))
}
