//! Format-agnostic inference weight handle.
//!
//! `InferenceWeights` is the single loading point for any code that needs to
//! run `infer_patched` against a vindex. It detects the quantisation format
//! from `VindexConfig`, loads the right on-disk artefacts, and dispatches to
//! `infer_patched` or `infer_patched_q4k` without the caller branching on
//! `config.quant`.
//!
//! **Scope:** the INFER / INSERT KNN / EXPLAIN INFER pipeline. Specialised
//! callers (bench, generation, Metal) keep their own explicit paths.

use std::path::Path;

use tokenizers::Tokenizer;

use larql_vindex::{
    GateIndex, IndexLoadCallbacks, KnnStore, PatchedKvIndexView, PatchedVindex, QuantFormat,
    VectorIndex, VindexConfig, VindexError,
};

use crate::active_span::{
    patch_activation_events, ActiveSpanConstraint, ActiveSpanSpec, PatchSlotId,
};
use crate::context_detector::ContextDecision;
use crate::context_detector::ContextDetectorSpec;
use crate::model::ModelWeights;

use super::infer_patched::{
    infer_patched, infer_patched_contextual, infer_patched_early_exit, infer_patched_q4k,
    infer_patched_q4k_contextual, infer_patched_q4k_early_exit, infer_patched_q4k_with_positions,
    infer_patched_with_positions, InferPatchedPositionResult, InferPatchedResult,
};
use super::predict::predict;
use super::PredictResult;

/// An inference-ready weight handle that is agnostic to quantisation format.
///
/// Constructed via [`InferenceWeights::load`]. Callers use
/// [`InferenceWeights::infer_patched`] and [`InferenceWeights::as_weights`]
/// without branching on the underlying format.
#[allow(clippy::large_enum_variant)]
pub enum InferenceWeights {
    Dense(ModelWeights),
    Quantised {
        weights: ModelWeights,
        index: VectorIndex,
    },
}

#[derive(Debug)]
pub enum PatchedCachedGenerationError {
    RequiresQuantisedWeights,
    Overlay(VindexError),
    UnsupportedArchitecture,
}

impl std::fmt::Display for PatchedCachedGenerationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RequiresQuantisedWeights => {
                formatter.write_str("patched cached generation requires quantised weights")
            }
            Self::Overlay(error) => write!(formatter, "invalid patched overlay: {error}"),
            Self::UnsupportedArchitecture => {
                formatter.write_str("architecture does not support cached quantised generation")
            }
        }
    }
}

impl std::error::Error for PatchedCachedGenerationError {}

impl From<VindexError> for PatchedCachedGenerationError {
    fn from(error: VindexError) -> Self {
        Self::Overlay(error)
    }
}

#[derive(Debug)]
pub struct PatchedCachedGenerationResult {
    pub generated: Vec<(String, u32)>,
    pub context_decisions: Vec<ContextDecision>,
    pub trigger_slot: Option<PatchSlotId>,
    pub released_after_span: bool,
}

impl InferenceWeights {
    /// Load weights for the vindex at `path`, choosing the right artefacts
    /// based on `config.quant`. Returns `VindexError` on any I/O or parse
    /// failure so callers can map it to their own error type.
    pub fn load(
        path: &Path,
        config: &VindexConfig,
        cb: &mut dyn IndexLoadCallbacks,
    ) -> Result<Self, VindexError> {
        if config.quant != QuantFormat::None {
            let mut idx = VectorIndex::load_vindex(path, cb)?;
            idx.load_attn_kquant(path)?;
            idx.load_interleaved_kquant(path)?;
            let weights = larql_vindex::load_model_weights_kquant(path, cb)?;
            Ok(Self::Quantised {
                weights,
                index: idx,
            })
        } else {
            let weights = larql_vindex::load_model_weights(path, cb)?;
            Ok(Self::Dense(weights))
        }
    }

    /// `true` if backed by a quantised (q4k or later) format.
    pub fn is_quantised(&self) -> bool {
        matches!(self, Self::Quantised { .. })
    }

    /// Borrow the underlying `ModelWeights` (arch + embeddings + norms).
    ///
    /// Always valid — both variants carry a `ModelWeights`. For the
    /// `Quantised` variant the attention/FFN tensor slots are empty; callers
    /// that need full attention tensors in memory must not use the dense path.
    pub fn as_weights(&self) -> &ModelWeights {
        match self {
            Self::Dense(w) => w,
            Self::Quantised { weights, .. } => weights,
        }
    }

    /// Mutably borrow the underlying `ModelWeights`.
    pub fn as_weights_mut(&mut self) -> &mut ModelWeights {
        match self {
            Self::Dense(w) => w,
            Self::Quantised { weights, .. } => weights,
        }
    }

    /// Borrow the lightweight model state and loaded quantised index together.
    /// Cached overlay generation needs both without reloading either artifact.
    pub fn as_quantised_parts_mut(&mut self) -> Option<(&mut ModelWeights, &VectorIndex)> {
        match self {
            Self::Quantised { weights, index } => Some((weights, index)),
            Self::Dense(_) => None,
        }
    }

    /// Generate through an applied patch overlay with model-bound contextual
    /// eligibility and exact multi-token continuation.
    ///
    /// This is the reusable production seam behind the LQL research harness:
    /// one causal patched forward supplies detector decisions and live slot
    /// evidence, then the cached generator recomputes through a fail-closed
    /// `KvIndex` view. Detector parameters are discovered from `patched`; live
    /// decisions and `ActiveSpan` state remain local to this call.
    pub fn generate_patched_cached(
        &mut self,
        tokenizer: &Tokenizer,
        patched: &PatchedVindex,
        prompt_ids: &[u32],
        max_tokens: usize,
        span_specs: &[ActiveSpanSpec],
    ) -> Result<PatchedCachedGenerationResult, PatchedCachedGenerationError> {
        if !self.is_quantised() {
            return Err(PatchedCachedGenerationError::RequiresQuantisedWeights);
        }

        let detectors: Vec<ContextDetectorSpec> = patched
            .context_detectors()
            .into_iter()
            .map(Into::into)
            .collect();
        let evidence = self.infer_patched_contextual(
            tokenizer,
            patched,
            None,
            prompt_ids,
            1,
            &super::KnnRouteMode::Legacy,
            &detectors,
        );
        let allowed_slots: Vec<_> = evidence
            .context_decisions
            .iter()
            .filter(|decision| decision.allowed)
            .map(|decision| (decision.payload.layer, decision.payload.feature))
            .collect();
        let mut events =
            patch_activation_events(self.as_weights(), patched, &evidence.infer.residuals);
        events.retain(|event| {
            let Some(_) = detectors.iter().find(|spec| spec.payload == event.slot) else {
                return true;
            };
            evidence
                .context_decisions
                .iter()
                .find(|decision| decision.payload == event.slot)
                .is_some_and(|decision| decision.allowed)
        });
        let context_decisions = evidence.context_decisions;
        let mut constraint = ActiveSpanConstraint::new(span_specs.to_vec(), events);

        let (weights, source) = self
            .as_quantised_parts_mut()
            .ok_or(PatchedCachedGenerationError::RequiresQuantisedWeights)?;
        let view = PatchedKvIndexView::new(source, patched, allowed_slots)?;
        let generated = crate::vindex::generate_kquant_cpu_constrained_cached_index(
            weights,
            tokenizer,
            prompt_ids,
            max_tokens,
            &view,
            |generated, logits| constraint.apply(generated, logits),
        )
        .ok_or(PatchedCachedGenerationError::UnsupportedArchitecture)?;
        let trigger_slot = constraint.triggered_slot();
        let released_after_span = trigger_slot.is_some() && constraint.active_slot().is_none();

        Ok(PatchedCachedGenerationResult {
            generated,
            context_decisions,
            trigger_slot,
            released_after_span,
        })
    }

    /// Run the shared INFER pipeline, dispatching to the correct forward path.
    ///
    /// Identical contract to [`infer_patched`] / [`infer_patched_q4k`]:
    /// unlimited walk FFN features, `KNN_COSINE_THRESHOLD = 0.75`, first
    /// stored layer wins. Callers do not branch on format.
    pub fn infer_patched(
        &mut self,
        tokenizer: &Tokenizer,
        gate_index: &dyn GateIndex,
        knn_store: Option<&KnnStore>,
        token_ids: &[u32],
        top_k: usize,
        route_mode: &super::KnnRouteMode,
    ) -> InferPatchedResult {
        match self {
            Self::Dense(weights) => infer_patched(
                weights, tokenizer, gate_index, knn_store, token_ids, top_k, route_mode,
            ),
            Self::Quantised { weights, index } => infer_patched_q4k(
                weights, tokenizer, gate_index, knn_store, token_ids, top_k, index, route_mode,
            ),
        }
    }

    /// Opt-in research inference retaining every pre-FFN sequence position.
    /// The ordinary `infer_patched` path remains last-position-only.
    pub fn infer_patched_with_positions(
        &mut self,
        tokenizer: &Tokenizer,
        gate_index: &dyn GateIndex,
        knn_store: Option<&KnnStore>,
        token_ids: &[u32],
        top_k: usize,
        route_mode: &super::KnnRouteMode,
    ) -> InferPatchedPositionResult {
        match self {
            Self::Dense(weights) => infer_patched_with_positions(
                weights, tokenizer, gate_index, knn_store, token_ids, top_k, route_mode,
            ),
            Self::Quantised { weights, index } => infer_patched_q4k_with_positions(
                weights, tokenizer, gate_index, knn_store, token_ids, top_k, index, route_mode,
            ),
        }
    }

    /// Patched inference with compact contextual payload decisions and no
    /// all-position trace allocation.
    #[allow(clippy::too_many_arguments)]
    pub fn infer_patched_contextual(
        &mut self,
        tokenizer: &Tokenizer,
        gate_index: &dyn GateIndex,
        knn_store: Option<&KnnStore>,
        token_ids: &[u32],
        top_k: usize,
        route_mode: &super::KnnRouteMode,
        detectors: &[ContextDetectorSpec],
    ) -> InferPatchedPositionResult {
        match self {
            Self::Dense(weights) => infer_patched_contextual(
                weights, tokenizer, gate_index, knn_store, token_ids, top_k, route_mode, detectors,
            ),
            Self::Quantised { weights, index } => infer_patched_q4k_contextual(
                weights, tokenizer, gate_index, knn_store, token_ids, top_k, index, route_mode,
                detectors,
            ),
        }
    }

    /// Early-exit INFER (FR retrieval-augmented early exit): short-circuit the
    /// forward at the highest stored layer when the FR1 verified router fires,
    /// skipping the tail + lm_head. Returns `(result, exited)`. On a miss it
    /// completes the full forward, so the result matches `infer_patched` in
    /// `Verified` mode. Dispatches Dense / Q4_K like [`Self::infer_patched`].
    #[allow(clippy::too_many_arguments)]
    pub fn infer_patched_early_exit(
        &mut self,
        tokenizer: &Tokenizer,
        gate_index: &dyn GateIndex,
        knn_store: Option<&KnnStore>,
        token_ids: &[u32],
        top_k: usize,
        k_candidates: usize,
        threshold: f32,
    ) -> (InferPatchedResult, bool) {
        match self {
            Self::Dense(weights) => infer_patched_early_exit(
                weights,
                tokenizer,
                gate_index,
                knn_store,
                token_ids,
                top_k,
                k_candidates,
                threshold,
            ),
            Self::Quantised { weights, index } => infer_patched_q4k_early_exit(
                weights,
                tokenizer,
                gate_index,
                knn_store,
                token_ids,
                top_k,
                index,
                k_candidates,
                threshold,
            ),
        }
    }

    /// Dense forward pass (no walk FFN, no KNN). Used for the
    /// `INFER COMPARE` dense side-by-side column.
    pub fn predict_dense(
        &mut self,
        tokenizer: &Tokenizer,
        token_ids: &[u32],
        top_k: usize,
    ) -> PredictResult {
        match self {
            Self::Dense(weights) => predict(weights, tokenizer, token_ids, top_k),
            Self::Quantised { weights, index } => {
                crate::vindex::predict_kquant(weights, tokenizer, token_ids, top_k, index)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! `InferenceWeights` coverage.
    //!
    //! Dense-branch tests construct `InferenceWeights::Dense` directly
    //! from `test_utils::make_test_weights` and exercise every method.
    //! `load(Dense)` is covered separately via the on-disk
    //! `write_synthetic_model_dir` fixture.
    //!
    //! The `Quantised` branch needs a Q4K on-disk vindex that the
    //! synthetic fixtures don't yet write. It is constructed in memory
    //! here via `make_test_q4k_weights` + `make_test_q4k_vindex` so the
    //! match-arms still get hit, but `load(Quantised)` stays uncovered
    //! pending the Q4K disk fixture work.
    use super::*;
    use crate::test_utils::{
        make_test_q4k_vindex, make_test_q4k_weights, make_test_tokenizer, make_test_vindex,
        make_test_weights, write_synthetic_model_dir,
    };
    use larql_vindex::{load_vindex_config, SilentLoadCallbacks};

    fn dense_fixture() -> (InferenceWeights, tokenizers::Tokenizer) {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        (InferenceWeights::Dense(weights), tokenizer)
    }

    fn quantised_fixture() -> (InferenceWeights, tokenizers::Tokenizer) {
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        (InferenceWeights::Quantised { weights, index }, tokenizer)
    }

    #[test]
    fn dense_is_quantised_reports_false() {
        let (iw, _) = dense_fixture();
        assert!(!iw.is_quantised());
    }

    #[test]
    fn quantised_is_quantised_reports_true() {
        let (iw, _) = quantised_fixture();
        assert!(iw.is_quantised());
    }

    #[test]
    fn dense_as_weights_returns_borrow() {
        let (iw, _) = dense_fixture();
        let w = iw.as_weights();
        assert_eq!(w.num_layers, 2);
        assert_eq!(w.hidden_size, 16);
    }

    #[test]
    fn quantised_as_weights_returns_borrow() {
        let (iw, _) = quantised_fixture();
        let w = iw.as_weights();
        // Q4K test fixture: 2 layers, hidden = Q4K_TEST_HIDDEN.
        assert_eq!(w.num_layers, 2);
        assert!(w.hidden_size > 0);
    }

    #[test]
    fn dense_as_weights_mut_returns_mutable_borrow() {
        let (mut iw, _) = dense_fixture();
        let w = iw.as_weights_mut();
        // Bump a field to prove we have &mut access — restore right after.
        let original = w.rope_base;
        w.rope_base = 12345.0;
        assert_eq!(w.rope_base, 12345.0);
        w.rope_base = original;
    }

    #[test]
    fn quantised_as_weights_mut_returns_mutable_borrow() {
        let (mut iw, _) = quantised_fixture();
        let w = iw.as_weights_mut();
        let original = w.rope_base;
        w.rope_base = 9999.0;
        assert_eq!(w.rope_base, 9999.0);
        w.rope_base = original;
    }

    #[test]
    fn quantised_parts_exposes_weights_and_index_together() {
        let (mut iw, _) = quantised_fixture();
        let (weights, index) = iw.as_quantised_parts_mut().unwrap();
        assert_eq!(weights.num_layers, index.num_layers);
        assert!(index.interleaved_kquant_layer_data(0).is_some());
    }

    #[test]
    fn dense_has_no_quantised_parts() {
        let (mut iw, _) = dense_fixture();
        assert!(iw.as_quantised_parts_mut().is_none());
    }

    #[test]
    fn dense_patched_cached_generation_fails_explicitly() {
        let (mut iw, tokenizer) = dense_fixture();
        let patched = PatchedVindex::new(make_test_vindex(iw.as_weights()));
        let error = iw
            .generate_patched_cached(&tokenizer, &patched, &[0, 1], 1, &[])
            .expect_err("dense weights cannot use quantised cached generation");
        assert!(matches!(
            error,
            PatchedCachedGenerationError::RequiresQuantisedWeights
        ));
    }

    #[test]
    fn quantised_patched_cached_generation_uses_overlay_seam() {
        let (mut iw, tokenizer) = quantised_fixture();
        let source = match &iw {
            InferenceWeights::Quantised { index, .. } => index.clone(),
            InferenceWeights::Dense(_) => unreachable!(),
        };
        let patched = PatchedVindex::new(source);
        let result = iw
            .generate_patched_cached(&tokenizer, &patched, &[0, 1], 2, &[])
            .expect("empty overlay still uses the generic cached seam");
        assert_eq!(result.generated.len(), 2);
        assert!(result.context_decisions.is_empty());
        assert_eq!(result.trigger_slot, None);
        assert!(!result.released_after_span);
    }

    #[test]
    fn dense_infer_patched_returns_predictions() {
        let (mut iw, tokenizer) = dense_fixture();
        let index = make_test_vindex(iw.as_weights());
        let result = iw.infer_patched(
            &tokenizer,
            &index,
            None,
            &[0u32, 1, 2],
            5,
            &crate::forward::KnnRouteMode::Legacy,
        );
        assert!(!result.predictions.is_empty());
        // top_k clamped by vocab/available rows; just check we got a
        // shaped result.
        assert!(result.predictions.len() <= 5);
    }

    #[test]
    fn dense_predict_dense_returns_predictions() {
        let (mut iw, tokenizer) = dense_fixture();
        let result = iw.predict_dense(&tokenizer, &[0u32, 1, 2], 5);
        assert!(!result.predictions.is_empty());
        assert!(result.predictions.len() <= 5);
    }

    #[test]
    fn load_dense_round_trips_via_disk_fixture() {
        // Exercises the Dense branch of `InferenceWeights::load`: write
        // the synthetic model dir, load it back via `load`, observe
        // is_quantised=false and a usable ModelWeights.
        let dir = tempfile::tempdir().expect("tempdir");
        write_synthetic_model_dir(dir.path()).expect("write fixture");
        let config = load_vindex_config(dir.path()).expect("load_vindex_config");
        assert_eq!(config.quant, QuantFormat::None);

        let mut cb = SilentLoadCallbacks;
        let iw =
            InferenceWeights::load(dir.path(), &config, &mut cb).expect("load InferenceWeights");
        assert!(!iw.is_quantised());
        let w = iw.as_weights();
        assert_eq!(w.num_layers, 2);
        assert_eq!(w.hidden_size, 16);
    }
}
