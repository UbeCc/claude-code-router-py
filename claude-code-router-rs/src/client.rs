use bytes::Bytes;
use futures::Stream;
use reqwest::{Client, Response, StatusCode};
use serde_json::Value;
use std::pin::Pin;
use std::time::Duration;
use tokio_stream::StreamExt;
use tracing::{info, warn, error};

const RETRY_STATUSES: &[u16] = &[429, 500, 502, 503, 504];

#[derive(Debug)]
pub struct ProviderError {
    pub status: u16,
    pub body: String,
}

impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Provider returned HTTP {}: {}", self.status, &self.body[..self.body.len().min(200)])
    }
}
impl std::error::Error for ProviderError {}

pub fn build_client() -> Client {
    Client::builder()
        .pool_max_idle_per_host(100)
        .pool_idle_timeout(Duration::from_secs(90))
        .timeout(Duration::from_secs(600))
        .no_proxy()
        .build()
        .expect("Failed to build HTTP client")
}

pub async fn post_json(
    client: &Client,
    url: &str,
    headers: &[(String, String)],
    body: &Value,
    timeout_secs: f64,
    max_retries: u32,
) -> Result<Value, ProviderError> {
    let model = body.get("model").and_then(|v| v.as_str()).unwrap_or("?");
    let stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    info!("[post_json] >>> POST {} model={} stream={}", url, model, stream);

    let mut last_err: Option<ProviderError> = None;

    for attempt in 0..=max_retries {
        if attempt > 0 {
            warn!("Retry {}/{} after sleep", attempt, max_retries);
            tokio::time::sleep(Duration::from_secs(1)).await;
        }

        let t0 = std::time::Instant::now();
        let mut req = client
            .post(url)
            .timeout(Duration::from_secs_f64(timeout_secs))
            .json(body);

        for (k, v) in headers {
            req = req.header(k.as_str(), v.as_str());
        }

        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                let elapsed = t0.elapsed().as_secs_f64();
                warn!(
                    "[post_json] Connection error (attempt {}) after {:.2}s: {}",
                    attempt + 1, elapsed, e
                );
                last_err = Some(ProviderError {
                    status: 0,
                    body: e.to_string(),
                });
                continue;
            }
        };

        let elapsed = t0.elapsed().as_secs_f64();
        let status = resp.status().as_u16();

        if RETRY_STATUSES.contains(&status) && attempt < max_retries {
            let text = resp.text().await.unwrap_or_default();
            warn!(
                "[post_json] HTTP {} from provider (attempt {}) after {:.2}s, will retry. body={}",
                status, attempt + 1, elapsed, &text[..text.len().min(500)]
            );
            last_err = Some(ProviderError { status, body: text });
            continue;
        }

        if status >= 400 {
            let text = resp.text().await.unwrap_or_default();
            error!("[post_json] <<< HTTP {} after {:.2}s body={}", status, elapsed, &text[..text.len().min(500)]);
            return Err(ProviderError { status, body: text });
        }

        info!("[post_json] <<< HTTP {} after {:.2}s", status, elapsed);
        let json: Value = resp.json().await.map_err(|e| ProviderError {
            status: 0,
            body: format!("JSON parse error: {}", e),
        })?;
        return Ok(json);
    }

    Err(last_err.unwrap_or(ProviderError {
        status: 0,
        body: "Unknown error after retries".to_string(),
    }))
}

pub async fn open_stream(
    url: &str,
    headers: &[(String, String)],
    body: &Value,
    timeout_secs: f64,
    max_retries: u32,
) -> Result<Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>, ProviderError> {
    let model = body.get("model").and_then(|v| v.as_str()).unwrap_or("?");
    info!("[stream] >>> POST {} model={} (stream)", url, model);

    let mut last_err: Option<ProviderError> = None;

    for attempt in 0..=max_retries {
        if attempt > 0 {
            warn!("[stream] Stream retry {}/{} after sleep", attempt, max_retries);
            tokio::time::sleep(Duration::from_secs(1)).await;
        }

        let t0 = std::time::Instant::now();

        // Build a fresh client per stream (streaming response stays open)
        let client = Client::builder()
            .timeout(Duration::from_secs_f64(timeout_secs))
            .no_proxy()
            .build()
            .map_err(|e| ProviderError {
                status: 0,
                body: e.to_string(),
            })?;

        let mut req = client.post(url).json(body);
        for (k, v) in headers {
            req = req.header(k.as_str(), v.as_str());
        }

        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                let elapsed = t0.elapsed().as_secs_f64();
                warn!(
                    "[stream] Connection error (attempt {}) after {:.2}s: {}",
                    attempt + 1, elapsed, e
                );
                last_err = Some(ProviderError {
                    status: 0,
                    body: e.to_string(),
                });
                continue;
            }
        };

        let elapsed = t0.elapsed().as_secs_f64();
        let status = resp.status().as_u16();

        if RETRY_STATUSES.contains(&status) && attempt < max_retries {
            warn!(
                "[stream] HTTP {} from provider (attempt {}) after {:.2}s, will retry",
                status, attempt + 1, elapsed
            );
            last_err = Some(ProviderError {
                status,
                body: String::new(),
            });
            continue;
        }

        if status >= 400 {
            let text = resp.text().await.unwrap_or_default();
            error!(
                "[stream] <<< HTTP {} after {:.2}s body={}",
                status, elapsed, &text[..text.len().min(500)]
            );
            return Err(ProviderError { status, body: text });
        }

        info!("[stream] <<< HTTP {} connected after {:.2}s", status, elapsed);
        return Ok(Box::pin(resp.bytes_stream()));
    }

    Err(last_err.unwrap_or(ProviderError {
        status: 0,
        body: "Unknown error after retries".to_string(),
    }))
}
