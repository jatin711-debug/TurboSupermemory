//! gRPC service implementation.

use crate::pb::memory_server::{Memory, MemoryServer};
use crate::pb::{
    DeleteRequest, DeleteResponse, FlushRequest, FlushResponse, GetPayloadRequest,
    GetPayloadResponse, HealthRequest, HealthResponse, InsertBatchRequest, InsertBatchResponse,
    InsertRequest, InsertResponse, SearchAnnRequest, SearchRequest, SearchResponse,
    StepSessionRequest, StepSessionResponse, TriggerConsolidationRequest,
    TriggerConsolidationResponse, UpdateRequest, UpdateResponse,
};
use crate::service::{
    map_results, pb_filter_to_storage, to_pb_results, validate_batch_lengths, ApiAuth, ApiError,
    MemoryService,
};
use tonic::service::interceptor::InterceptedService;
use tonic::{Request, Response, Status};

/// Build the gRPC `Memory` service, enforcing bearer-token authentication on
/// every call according to `auth`.
// tonic::Status is a large Err variant, but the Interceptor signature is fixed by tonic.
#[allow(clippy::result_large_err)]
pub fn server(
    service: MemoryService,
    auth: ApiAuth,
) -> InterceptedService<MemoryServer<MemoryService>, impl tonic::service::Interceptor + Clone> {
    MemoryServer::with_interceptor(service, move |request| auth_interceptor(&auth, request))
}

/// Interceptor rejecting calls without a valid `authorization: Bearer <key>`
/// metadata entry when auth is enabled; passes every call when it is not.
#[allow(clippy::result_large_err)]
pub fn auth_interceptor(auth: &ApiAuth, request: Request<()>) -> Result<Request<()>, Status> {
    let header = request
        .metadata()
        .get("authorization")
        .and_then(|v| v.to_str().ok());
    if auth.is_authorized(header) {
        Ok(request)
    } else {
        Err(Status::unauthenticated("missing or invalid bearer token"))
    }
}

#[tonic::async_trait]
impl Memory for MemoryService {
    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        Ok(Response::new(HealthResponse {
            status: "ok".into(),
        }))
    }

    async fn insert(
        &self,
        request: Request<InsertRequest>,
    ) -> Result<Response<InsertResponse>, Status> {
        let req = request.into_inner();
        let success = self
            .engine()
            .insert_with_payload_role(
                &req.id,
                &req.text,
                &req.embedding,
                req.importance,
                &req.concepts,
                req.payload,
                req.scope,
                req.source_role,
            )
            .map_err(ApiError::from)?;
        Ok(Response::new(InsertResponse { success }))
    }

    async fn insert_batch(
        &self,
        request: Request<InsertBatchRequest>,
    ) -> Result<Response<InsertBatchResponse>, Status> {
        let req = request.into_inner();
        validate_batch_lengths(
            req.ids.len(),
            req.texts.len(),
            req.embeddings.len(),
            req.scores.len(),
            req.concepts.len(),
            req.payloads.len(),
            req.scopes.len(),
            req.source_roles.len(),
        )?;
        let ids = req.ids;
        let texts = req.texts;
        let embeddings: Vec<Vec<f32>> = req.embeddings.into_iter().map(|e| e.values).collect();
        let emb_refs: Vec<&[f32]> = embeddings.iter().map(|v| v.as_slice()).collect();
        let concepts: Vec<Vec<String>> = req.concepts.into_iter().map(|c| c.values).collect();
        let payloads: Vec<Option<String>> = if req.payloads.is_empty() {
            Vec::new()
        } else {
            req.payloads.into_iter().map(Some).collect()
        };
        let scopes: Vec<Option<String>> = if req.scopes.is_empty() {
            Vec::new()
        } else {
            req.scopes.into_iter().map(Some).collect()
        };
        let source_roles: Vec<Option<String>> = if req.source_roles.is_empty() {
            Vec::new()
        } else {
            req.source_roles.into_iter().map(Some).collect()
        };
        let count = self
            .engine()
            .insert_batch_with_payload_role(
                &ids,
                &texts,
                &emb_refs,
                &req.scores,
                &concepts,
                &payloads,
                &scopes,
                &source_roles,
            )
            .map_err(ApiError::from)?;
        Ok(Response::new(InsertBatchResponse {
            count: count as u32,
        }))
    }

    async fn delete(
        &self,
        request: Request<DeleteRequest>,
    ) -> Result<Response<DeleteResponse>, Status> {
        let req = request.into_inner();
        let success = self
            .engine()
            .delete_by_id(&req.id)
            .map_err(ApiError::from)?;
        Ok(Response::new(DeleteResponse { success }))
    }

    async fn update(
        &self,
        request: Request<UpdateRequest>,
    ) -> Result<Response<UpdateResponse>, Status> {
        let req = request.into_inner();
        let success = self
            .engine()
            .update_with_payload_role(
                &req.id,
                &req.text,
                &req.embedding,
                req.importance,
                &req.concepts,
                req.payload,
                req.scope,
                req.source_role,
            )
            .map_err(ApiError::from)?;
        Ok(Response::new(UpdateResponse { success }))
    }

    async fn get_payload(
        &self,
        request: Request<GetPayloadRequest>,
    ) -> Result<Response<GetPayloadResponse>, Status> {
        let req = request.into_inner();
        match self.engine().get_payload(&req.id).map_err(ApiError::from)? {
            Some(payload) => Ok(Response::new(GetPayloadResponse {
                found: true,
                payload,
            })),
            None => Ok(Response::new(GetPayloadResponse {
                found: false,
                payload: String::new(),
            })),
        }
    }

    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let req = request.into_inner();
        let filter = pb_filter_to_storage(req.filter.as_ref())?;
        let scope = req.scope.as_deref();
        let maybe = match filter {
            Some(f) => self
                .engine()
                .search_filtered_with_scope(
                    &req.query_text,
                    &req.query_embedding,
                    req.top_k as usize,
                    &f,
                    None,
                    scope,
                )
                .map_err(ApiError::from)?,
            None => self
                .engine()
                .search_scoped(
                    &req.query_text,
                    &req.query_embedding,
                    req.top_k as usize,
                    scope,
                )
                .map_err(ApiError::from)?,
        };
        if let Some(results) = maybe {
            Ok(Response::new(SearchResponse {
                results: to_pb_results(map_results(results)),
                gated: false,
            }))
        } else {
            Ok(Response::new(SearchResponse {
                results: Vec::new(),
                gated: true,
            }))
        }
    }

    async fn search_ann(
        &self,
        request: Request<SearchAnnRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let req = request.into_inner();
        let filter = pb_filter_to_storage(req.filter.as_ref())?;
        let scope = req.scope.as_deref();
        let results = match filter {
            Some(f) => self
                .engine()
                .search_ann_candidates_filtered_with_ef(
                    &req.query_embedding,
                    req.top_k as usize,
                    Some(&f),
                    None,
                    scope,
                )
                .map_err(ApiError::from)?,
            None => self
                .engine()
                .search_ann_scoped(&req.query_embedding, req.top_k as usize, None, scope)
                .map_err(ApiError::from)?,
        };
        Ok(Response::new(SearchResponse {
            results: to_pb_results(map_results(results)),
            gated: false,
        }))
    }

    async fn step_session(
        &self,
        request: Request<StepSessionRequest>,
    ) -> Result<Response<StepSessionResponse>, Status> {
        let req = request.into_inner();
        let ccs_json = self
            .engine()
            .step_session(&req.user_input, &req.assistant_response)
            .map_err(ApiError::from)?;
        Ok(Response::new(StepSessionResponse { ccs_json }))
    }

    async fn trigger_consolidation(
        &self,
        _request: Request<TriggerConsolidationRequest>,
    ) -> Result<Response<TriggerConsolidationResponse>, Status> {
        let (sealed, compacted, _promoted) = self
            .engine()
            .trigger_consolidation()
            .map_err(ApiError::from)?;
        Ok(Response::new(TriggerConsolidationResponse {
            sealed: sealed as u32,
            compacted: compacted as u32,
        }))
    }

    async fn flush(
        &self,
        _request: Request<FlushRequest>,
    ) -> Result<Response<FlushResponse>, Status> {
        self.engine().flush().map_err(ApiError::from)?;
        Ok(Response::new(FlushResponse { success: true }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request_with_token(token: &str) -> Request<()> {
        let mut request = Request::new(());
        request
            .metadata_mut()
            .insert("authorization", token.parse().unwrap());
        request
    }

    #[test]
    fn interceptor_rejects_missing_token() {
        let auth = ApiAuth::new(Some("key".into()));
        let err = auth_interceptor(&auth, Request::new(())).unwrap_err();
        assert_eq!(err.code(), tonic::Code::Unauthenticated);
    }

    #[test]
    fn interceptor_rejects_wrong_token() {
        let auth = ApiAuth::new(Some("key".into()));
        let err = auth_interceptor(&auth, request_with_token("Bearer nope")).unwrap_err();
        assert_eq!(err.code(), tonic::Code::Unauthenticated);
    }

    #[test]
    fn interceptor_accepts_valid_token() {
        let auth = ApiAuth::new(Some("key".into()));
        assert!(auth_interceptor(&auth, request_with_token("Bearer key")).is_ok());
    }

    #[test]
    fn interceptor_open_when_disabled() {
        let auth = ApiAuth::new(None);
        assert!(auth_interceptor(&auth, Request::new(())).is_ok());
    }
}
