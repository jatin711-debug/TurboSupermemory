//! gRPC service implementation.

use crate::pb::{
    memory_server::Memory, DeleteRequest, DeleteResponse, FlushRequest, FlushResponse,
    GetPayloadRequest, GetPayloadResponse, HealthRequest, HealthResponse, InsertBatchRequest,
    InsertBatchResponse, InsertRequest, InsertResponse, SearchAnnRequest, SearchRequest,
    SearchResponse, StepSessionRequest, StepSessionResponse, TriggerConsolidationRequest,
    TriggerConsolidationResponse, UpdateRequest, UpdateResponse,
};
use crate::service::{map_results, pb_filter_to_storage, to_pb_results, ApiError, MemoryService};
use tonic::{Request, Response, Status};

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
            .insert_with_payload(
                &req.id,
                &req.text,
                &req.embedding,
                req.importance,
                &req.concepts,
                req.payload,
            )
            .map_err(ApiError::from)?;
        Ok(Response::new(InsertResponse { success }))
    }

    async fn insert_batch(
        &self,
        request: Request<InsertBatchRequest>,
    ) -> Result<Response<InsertBatchResponse>, Status> {
        let req = request.into_inner();
        let ids = req.ids;
        let texts = req.texts;
        let embeddings: Vec<Vec<f32>> = req.embeddings.into_iter().map(|e| e.values).collect();
        let concepts: Vec<Vec<String>> = req.concepts.into_iter().map(|c| c.values).collect();
        let payloads: Vec<Option<String>> = if req.payloads.is_empty() {
            Vec::new()
        } else {
            req.payloads.into_iter().map(Some).collect()
        };
        let count = self
            .engine()
            .insert_batch_with_payload(&ids, &texts, &embeddings, &req.scores, &concepts, &payloads)
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
            .update_with_payload(
                &req.id,
                &req.text,
                &req.embedding,
                req.importance,
                &req.concepts,
                req.payload,
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
        let filter = pb_filter_to_storage(req.filter.as_ref()).map_err(ApiError::from)?;
        let maybe = match filter {
            Some(f) => self
                .engine()
                .search_filtered(&req.query_text, &req.query_embedding, req.top_k as usize, &f)
                .map_err(ApiError::from)?,
            None => self
                .engine()
                .search(&req.query_text, &req.query_embedding, req.top_k as usize)
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
        let filter = pb_filter_to_storage(req.filter.as_ref()).map_err(ApiError::from)?;
        let results = match filter {
            Some(f) => self
                .engine()
                .search_ann_filtered(&req.query_embedding, req.top_k as usize, &f)
                .map_err(ApiError::from)?,
            None => self
                .engine()
                .search_ann(&req.query_embedding, req.top_k as usize)
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
