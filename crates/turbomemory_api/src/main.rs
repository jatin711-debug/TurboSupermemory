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
//! - `TURBO_FLUSH_INTERVAL_SECS`: how often the store is flushed in the
//!   background (default `30`, `0` disables). A flush fsyncs the write-ahead
//!   log and snapshot, truncates the log, and builds any pending index
//!   segments; without it a store only does those things when a client calls
//!   `/flush`.
//! - `RUST_LOG`: standard tracing filter (default `turbomemory_api=info`).

use std::future::IntoFuture;
use std::net::SocketAddr;
use std::time::Duration;
use tokio::sync::watch;
use turbomemory_api::service::{ApiAuth, MemoryService};

/// Resolves on Ctrl-C, or on SIGTERM where that exists (what `docker stop`
/// and Kubernetes send).
async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(e) = tokio::signal::ctrl_c().await {
            tracing::error!("failed to listen for Ctrl-C: {e}");
            std::future::pending::<()>().await;
        }
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut stream) => {
                stream.recv().await;
            }
            Err(e) => {
                tracing::error!("failed to listen for SIGTERM: {e}");
                std::future::pending::<()>().await;
            }
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => tracing::info!("received Ctrl-C, shutting down"),
        _ = terminate => tracing::info!("received SIGTERM, shutting down"),
    }
}

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
    } else if std::env::var_os("TURBO_API_KEY").is_some() {
        // Set but empty (an unset variable expanded by a compose file, say)
        // or not valid text: that is a misconfiguration, not a choice.
        tracing::warn!(
            "TURBO_API_KEY is set but empty or unreadable: authentication is DISABLED \
             and anyone who can reach the server has full read/write access"
        );
    } else if grpc_addr.ip().is_unspecified() || rest_addr.ip().is_unspecified() {
        tracing::warn!(
            "TURBO_API_KEY is not set and a server is binding to a wildcard address; \
             anyone who can reach it has full read/write access"
        );
    }

    let service = MemoryService::open(&db_path, dimension)?;
    let engine = service.engine().clone();
    let report = engine.recovery_report();
    if !report.is_clean() {
        tracing::warn!("store was recovered on open: {report:?}");
    }

    // Shutdown broadcast: a termination signal or a server failure asks both
    // servers to stop.
    let (shutdown_tx, _) = watch::channel(());
    let signal_tx = shutdown_tx.clone();
    tokio::spawn(async move {
        shutdown_signal().await;
        let _ = signal_tx.send(());
    });

    // Periodic flush. The service opens the engine without its background
    // optimizer, so nothing else would ever fsync, truncate the write-ahead
    // log, or turn full Hot segments into indexed ones while the server runs.
    let flush_secs = std::env::var("TURBO_FLUSH_INTERVAL_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(30);
    if flush_secs > 0 {
        let engine = engine.clone();
        let mut stop = shutdown_tx.subscribe();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_secs(flush_secs));
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            ticker.tick().await; // the first tick is immediate
            loop {
                tokio::select! {
                    _ = ticker.tick() => {}
                    _ = stop.changed() => break,
                }
                let engine = engine.clone();
                match tokio::task::spawn_blocking(move || engine.flush()).await {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => tracing::error!("periodic flush failed: {e}"),
                    Err(e) => tracing::error!("periodic flush panicked: {e}"),
                }
            }
        });
        tracing::info!("flushing every {flush_secs}s (TURBO_FLUSH_INTERVAL_SECS)");
    } else {
        tracing::warn!(
            "TURBO_FLUSH_INTERVAL_SECS=0: the store is only flushed when a client asks for it"
        );
    }

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

    // Both servers have stopped taking requests: make everything durable and
    // build any pending index segments before the process exits. (Writes are
    // recoverable from the write-ahead log even without this, but a clean
    // shutdown should not leave recovery work for the next start.)
    match tokio::task::spawn_blocking(move || engine.shutdown()).await {
        Ok(Ok(())) => tracing::info!("store flushed"),
        Ok(Err(e)) => {
            tracing::error!("final flush failed: {e}");
            failed = true;
        }
        Err(e) => {
            tracing::error!("final flush panicked: {e}");
            failed = true;
        }
    }

    if failed {
        Err("one or more servers failed".into())
    } else {
        Ok(())
    }
}
