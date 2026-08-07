//! gRPC + REST API for TurboSuperMemory.

pub mod grpc;
pub mod rest;
pub mod service;

pub mod pb {
    #![allow(clippy::all)]
    include!(concat!(env!("OUT_DIR"), "/turbomemory.rs"));
}

pub use service::{ApiAuth, ApiError, MemoryService};
