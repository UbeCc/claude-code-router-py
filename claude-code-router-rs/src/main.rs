mod client;
mod config;
mod converter;
mod server;
mod streaming;
mod types;

use clap::Parser;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use tracing::info;

#[derive(Parser)]
#[command(name = "ccr", about = "Claude Code Router (Rust)")]
struct Cli {
    /// Port to listen on
    #[arg(short, long, default_value = "8891")]
    port: u16,

    /// Path to config JSON file
    #[arg(short, long)]
    config: Option<String>,
}

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(true)
        .init();

    let cli = Cli::parse();

    // Load config
    let cfg = if let Some(json_str) = std::env::var("CCR_CONFIG_JSON").ok() {
        serde_json::from_str(&json_str).expect("Failed to parse CCR_CONFIG_JSON")
    } else if let Some(cfg) = config::build_config_from_env() {
        info!("Config loaded from CCR_* env vars");
        cfg
    } else {
        let path = cli
            .config
            .or_else(|| std::env::var("CCR_CONFIG").ok())
            .unwrap_or_else(|| "config.json".to_string());
        config::load_config(&path).unwrap_or_else(|e| {
            eprintln!("Failed to load config from {}: {}", path, e);
            std::process::exit(1);
        })
    };

    info!(
        "Loaded config: {} providers, timeout={}s",
        cfg.providers.len(),
        cfg.timeout_secs()
    );
    for p in &cfg.providers {
        info!("  Provider '{}': {}", p.name, p.api_base_url);
    }
    if let Some((pname, model)) = config::resolve_route(&cfg) {
        info!("  Default route: provider={}, model={}", pname, model);
    }

    let state = Arc::new(server::AppState {
        config: cfg,
        client: client::build_client(),
        inflight: AtomicU64::new(0),
        total_requests: AtomicU64::new(0),
    });

    let app = server::build_router(state);

    let addr = format!("0.0.0.0:{}", cli.port);
    info!("Starting CCR (Rust) on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("Failed to bind");

    axum::serve(listener, app)
        .await
        .expect("Server error");
}
