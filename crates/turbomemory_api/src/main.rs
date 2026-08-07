//! TurboSuperMemory API server.
//!
//! Configuration via environment variables:
//! - `TURBO_DB_PATH`: database directory (default `./turbo_db`)
//! - `TURBO_DIMENSION`: embedding dimension (default `768`)
//! - `TURBO_GRPC_ADDR`: gRPC bind address (default `0.0.0.0:50051`)
//! - `TURBO_REST_ADDR`: REST bind address (default `0.0.0.0:8080`)
//! - `TURBO_API_KEY`: optional bearer token. When set, every REST request must
//!   send `Authorization: Bearer <key>` and every gRPC call must carry an
//!   `authorization: Bearer <key>` metadata entry. When unset, the server runs
//!   without authentication — do not expose it on an untrusted network.
//! - `RUST_LOG`: standard tracing filter (default `turbomemory_api=info`).

use std::future::IntoFuture;
use std::net::SocketAddr;
use tokio::sync::watch;
use turbomemory_api::service::{ApiAuth, MemoryService};

/// Resolves when the shutdown broadcast fires.
async fn shutdown_requested(mut rx: watch::Receiver<()>) {
    // Err means every sender was dropped; shut down in that case too.
    let _ = rx.changed().await;
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let db_path = std::env::var("TURBO_DB_PATH").unwrap_or_else(|_| "./turbo_db".into());
    let dimension = std::env::var("TURBO_DIMENSION")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(768);
    let grpc_addr: SocketAddr = std::env::var("TURBO_GRPC_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:50051".into())
        .parse()?;
    let rest_addr: SocketAddr = std::env::var("TURBO_REST_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:8080".into())
        .parse()?;

    let auth = ApiAuth::from_env();
    if auth.is_required() {
        tracing::info!("bearer-token authentication enabled (TURBO_API_KEY)");
    } else if grpc_addr.ip().is_unspecified() || rest_addr.ip().is_unspecified() {
        tracing::warn!(
            "TURBO_API_KEY is not set and a server is binding to a wildcard address; \
             anyone who can reach it has full read/write access"
        );
    }

    let service = MemoryService::open(&db_path, dimension)?;

    // Shutdown broadcast: Ctrl-C or a server failure asks both servers to stop.
    let (shutdown_tx, _) = watch::channel(());
    let signal_tx = shutdown_tx.clone();
    tokio::spawn(async move {
        if let Err(e) = tokio::signal::ctrl_c().await {
            tracing::error!("failed to listen for Ctrl-C: {e}");
        } else {
            tracing::info!("received Ctrl-C, shutting down");
        }
        let _ = signal_tx.send(());
    });

    // gRPC server
    let grpc = tonic::transport::Server::builder()
        .add_service(turbomemory_api::grpc::server(service.clone(), auth.clone()))
        .serve_with_shutdown(grpc_addr, shutdown_requested(shutdown_tx.subscribe()));

    // REST server
    let rest_listener = tokio::net::TcpListener::bind(rest_addr).await?;
    let rest = axum::serve(rest_listener, turbomemory_api::rest::router(service, auth))
        .with_graceful_shutdown(shutdown_requested(shutdown_tx.subscribe()))
        .into_future();

    tracing::info!("gRPC listening on {grpc_addr}");
    tracing::info!("REST listening on {rest_addr}");

    let grpc_task = tokio::spawn(grpc);
    let rest_task = tokio::spawn(rest);
    tokio::pin!(grpc_task);
    tokio::pin!(rest_task);

    let mut failed = false;
    tokio::select! {
        res = &mut grpc_task => match res {
            Ok(Ok(())) => tracing::info!("gRPC server stopped"),
            Ok(Err(e)) => {
                tracing::error!("gRPC server failed: {e}");
                failed = true;
            }
            Err(e) => {
                tracing::error!("gRPC server task failed: {e}");
                failed = true;
            }
        },
        res = &mut rest_task => match res {
            Ok(Ok(())) => tracing::info!("REST server stopped"),
            Ok(Err(e)) => {
                tracing::error!("REST server failed: {e}");
                failed = true;
            }
            Err(e) => {
                tracing::error!("REST server task failed: {e}");
                failed = true;
            }
        },
    }

    // Ask whichever server is still running to shut down, then wait for both.
    let _ = shutdown_tx.send(());
    if let Err(e) = grpc_task.await {
        tracing::error!("gRPC server task failed: {e}");
        failed = true;
    }
    if let Err(e) = rest_task.await {
        tracing::error!("REST server task failed: {e}");
        failed = true;
    }

    if failed {
        Err("one or more servers failed".into())
    } else {
        Ok(())
    }
}
