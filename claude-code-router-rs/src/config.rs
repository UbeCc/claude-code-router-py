use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::env;
use std::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(rename = "API_TIMEOUT_MS", default = "default_timeout_ms")]
    pub api_timeout_ms: Value, // can be string or number
    #[serde(rename = "Providers", default)]
    pub providers: Vec<Provider>,
    #[serde(rename = "Router", default)]
    pub router: HashMap<String, String>,
    #[serde(default)]
    pub tokenizer_path: Option<String>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

fn default_timeout_ms() -> Value {
    Value::Number(600_000.into())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provider {
    pub name: String,
    pub api_base_url: String,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    #[serde(default)]
    pub params: Option<ProviderParams>,
    #[serde(default)]
    pub tokenizer_path: Option<String>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

fn default_max_retries() -> u32 {
    3
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderParams {
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub max_tokens: Option<u64>,
    #[serde(default)]
    pub reasoning: Option<Value>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

impl Config {
    pub fn timeout_secs(&self) -> f64 {
        let ms = match &self.api_timeout_ms {
            Value::Number(n) => n.as_f64().unwrap_or(600_000.0),
            Value::String(s) => s.parse::<f64>().unwrap_or(600_000.0),
            _ => 600_000.0,
        };
        ms / 1000.0
    }
}

fn interpolate_env_vars(text: &str) -> String {
    let re = Regex::new(r"\$\{(\w+)\}|\$(\w+)").unwrap();
    re.replace_all(text, |caps: &regex::Captures| {
        let name = caps.get(1).or_else(|| caps.get(2)).unwrap().as_str();
        env::var(name).unwrap_or_else(|_| caps[0].to_string())
    })
    .to_string()
}

fn interpolate_value(val: Value) -> Value {
    match val {
        Value::String(s) => Value::String(interpolate_env_vars(&s)),
        Value::Object(map) => {
            let mut out = serde_json::Map::new();
            for (k, v) in map {
                out.insert(k, interpolate_value(v));
            }
            Value::Object(out)
        }
        Value::Array(arr) => Value::Array(arr.into_iter().map(interpolate_value).collect()),
        other => other,
    }
}

pub fn load_config(path: &str) -> Result<Config, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    let raw: Value = serde_json::from_str(&text)?;
    let interpolated = interpolate_value(raw);
    let config: Config = serde_json::from_value(interpolated)?;
    Ok(config)
}

pub fn build_config_from_env() -> Option<Config> {
    let api_base_url = env::var("CCR_API_BASE_URL").ok()?;

    let mut params = ProviderParams {
        temperature: env::var("CCR_TEMPERATURE").ok().and_then(|v| v.parse().ok()),
        top_p: env::var("CCR_TOP_P").ok().and_then(|v| v.parse().ok()),
        max_tokens: env::var("CCR_MAX_TOKENS").ok().and_then(|v| v.parse().ok()),
        reasoning: env::var("CCR_BUDGET_TOKENS").ok().and_then(|v| {
            v.parse::<u64>().ok().map(|bt| {
                serde_json::json!({"budget_tokens": bt})
            })
        }),
        extra: HashMap::new(),
    };

    let provider = Provider {
        name: "default".to_string(),
        api_base_url,
        api_key: env::var("CCR_API_KEY")
            .or_else(|_| env::var("API_KEY"))
            .ok(),
        max_retries: env::var("CCR_MAX_RETRIES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3),
        params: if params.temperature.is_some()
            || params.top_p.is_some()
            || params.max_tokens.is_some()
            || params.reasoning.is_some()
        {
            Some(params)
        } else {
            None
        },
        tokenizer_path: env::var("CCR_TOKENIZER_PATH")
            .or_else(|_| env::var("TOKENIZER_PATH"))
            .ok(),
        extra: HashMap::new(),
    };

    let model = env::var("CCR_MODEL").unwrap_or_else(|_| "/model".to_string());

    Some(Config {
        api_timeout_ms: Value::Number(
            env::var("CCR_API_TIMEOUT_MS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(850_000)
                .into(),
        ),
        providers: vec![provider],
        router: {
            let mut m = HashMap::new();
            m.insert("default".to_string(), format!("default,{}", model));
            m
        },
        tokenizer_path: env::var("CCR_TOKENIZER_PATH")
            .or_else(|_| env::var("TOKENIZER_PATH"))
            .ok(),
        extra: HashMap::new(),
    })
}

pub fn get_provider<'a>(config: &'a Config, name: &str) -> Option<&'a Provider> {
    config.providers.iter().find(|p| p.name == name)
}

pub fn resolve_route(config: &Config) -> Option<(String, String)> {
    let target = config
        .router
        .get("default")
        .or_else(|| config.router.values().next())?;
    let parts: Vec<&str> = target.splitn(2, ',').collect();
    if parts.len() != 2 {
        return None;
    }
    Some((parts[0].trim().to_string(), parts[1].trim().to_string()))
}

pub fn apply_provider_params(provider: &Provider, req: &mut Value) {
    let params = match &provider.params {
        Some(p) => p,
        None => return,
    };
    let obj = match req.as_object_mut() {
        Some(o) => o,
        None => return,
    };

    // temperature / top_p defaults
    for field in &["temperature", "top_p"] {
        let pval = match *field {
            "temperature" => params.temperature,
            "top_p" => params.top_p,
            _ => None,
        };
        if let Some(v) = pval {
            if obj.get(*field).and_then(|x| x.as_f64()).is_none() {
                obj.insert(field.to_string(), serde_json::json!(v));
            }
        }
    }

    // max_tokens ceiling
    if let Some(limit) = params.max_tokens {
        match obj.get("max_tokens").and_then(|x| x.as_u64()) {
            None => {
                obj.insert("max_tokens".to_string(), serde_json::json!(limit));
            }
            Some(cur) => {
                obj.insert(
                    "max_tokens".to_string(),
                    serde_json::json!(cur.min(limit)),
                );
            }
        }
    }

    // reasoning/thinking injection
    if let Some(reasoning) = &params.reasoning {
        let has_thinking = obj.contains_key("thinking");
        let has_tools = obj.contains_key("tools");
        if !has_thinking && !has_tools {
            let budget = reasoning
                .get("budget_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(8000);
            obj.insert(
                "thinking".to_string(),
                serde_json::json!({"type": "enabled", "budget_tokens": budget}),
            );
        }
    }
}

pub fn provider_headers(provider: &Provider) -> Vec<(String, String)> {
    let mut headers = vec![("Content-Type".to_string(), "application/json".to_string())];
    if let Some(key) = &provider.api_key {
        if !key.is_empty() {
            headers.push(("Authorization".to_string(), format!("Bearer {}", key)));
        }
    }
    headers
}

pub fn api_base(provider: &Provider) -> String {
    let url = &provider.api_base_url;
    for suffix in &[
        "/chat/completions",
        "/completions",
        "/models",
        "/batches",
        "/files",
    ] {
        if url.ends_with(suffix) {
            return url[..url.len() - suffix.len()].to_string();
        }
    }
    if url.trim_end_matches('/').ends_with("/v1") {
        return url.trim_end_matches('/').to_string();
    }
    match url.rfind('/') {
        Some(pos) => url[..pos].to_string(),
        None => url.clone(),
    }
}
