//! TurboSuperMemory API server.

use std::net::SocketAddr;
use turbomemory_api::pb::memory_server::MemoryServer;
use turbomemory_api::service::MemoryService;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    let service = MemoryService::open(&db_path, dimension)?;

    // gRPC server
    let grpc = tonic::transport::Server::builder()
        .add_service(MemoryServer::new(service.clone()))
        .serve(grpc_addr);

    // REST server
    let rest = axum::serve(
        tokio::net::TcpListener::bind(rest_addr).await?,
        turbomemory_api::rest::router(service),
    );

    tracing::info!("gRPC listening on {grpc_addr}");
    tracing::info!("REST listening on {rest_addr}");

    tokio::select! {
        res = grpc => res?,
        res = rest => res?,
    };

    Ok(())
}
