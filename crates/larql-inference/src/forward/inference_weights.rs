//! Format-agnostic inference weight handle.
//!
//! `InferenceWeights` is the single loading point for any code that needs to
//! run `infer_patched` against a vindex. It detects the quantisation format
//! from `VindexConfig`, loads the right on-disk artefacts, and dispatches to
//! `infer_patched` or `infer_patched_q4k` without the caller branching on
//! `config.quant`.
//!
//! **Scope:** the INFER / INSERT KNN / EXPLAIN INFER pipeline plus contextual
//! patched generation across dense and native-quantised storage. Specialised
//! callers (bench, unconstrained generation, Metal) keep explicit paths.

use std::path::Path;

use tokenizers::Tokenizer;

use larql_vindex::{
    GateIndex, IndexLoadCallbacks, KnnStore, PatchOverrides, PatchedKvIndexView, PatchedVindex,
    QuantFormat, VectorIndex, VindexConfig, VindexError,
};

use crate::active_span::{
    patch_activation_events, ActiveSpanConstraint, ActiveSpanSpec, PatchSlotId,
};
use crate::compiled_surface_context::{
    compiled_surface_prefill_delta, evaluate_compiled_surface_contexts,
    CompiledSurfaceContextManifest, CompiledSurfaceContextSpec, CompiledSurfaceDecision,
};
use crate::context_detector::{terminal_prefill_delta, ContextDecision, ContextDetectorSpec};
use crate::model::ModelWeights;

use super::infer_patched::{
    infer_patched, infer_patched_contextual, infer_patched_contextual_with_model_ids,
    infer_patched_early_exit, infer_patched_q4k, infer_patched_q4k_contextual,
    infer_patched_q4k_early_exit, infer_patched_q4k_with_positions, infer_patched_with_positions,
    InferPatchedPositionResult, InferPatchedResult,
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
pub enum PatchedGenerationError {
    RequiresQuantisedWeights,
    RequiresDenseWeights,
    EmptyPrompt,
    NoModelToken,
    Overlay(VindexError),
    InvalidTerminalResidualAdapter(String),
    InvalidCompiledSurfaceContext(String),
    UnsupportedArchitecture,
}

impl std::fmt::Display for PatchedGenerationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RequiresQuantisedWeights => {
                formatter.write_str("patched cached generation requires quantised weights")
            }
            Self::RequiresDenseWeights => {
                formatter.write_str("patched dense generation requires unquantised weights")
            }
            Self::EmptyPrompt => formatter.write_str("patched generation requires a prompt"),
            Self::NoModelToken => {
                formatter.write_str("patched generation produced no model-path token")
            }
            Self::Overlay(error) => write!(formatter, "invalid patched overlay: {error}"),
            Self::InvalidTerminalResidualAdapter(error) => {
                write!(formatter, "invalid terminal residual adapter: {error}")
            }
            Self::InvalidCompiledSurfaceContext(error) => {
                write!(formatter, "invalid compiled surface context: {error}")
            }
            Self::UnsupportedArchitecture => {
                formatter.write_str("architecture does not support cached quantised generation")
            }
        }
    }
}

impl std::error::Error for PatchedGenerationError {}

impl From<VindexError> for PatchedGenerationError {
    fn from(error: VindexError) -> Self {
        Self::Overlay(error)
    }
}

#[derive(Debug)]
pub struct PatchedGenerationResult {
    pub generated: Vec<(String, u32)>,
    pub context_decisions: Vec<ContextDecision>,
    pub trigger_slot: Option<PatchSlotId>,
    pub released_after_span: bool,
}

#[derive(Debug)]
pub struct CompiledSurfaceGenerationResult {
    pub generated: Vec<(String, u32)>,
    pub decisions: Vec<CompiledSurfaceDecision>,
    pub trigger_slot: Option<PatchSlotId>,
    pub released_after_span: bool,
}

/// Compatibility names retained for callers that adopted the initial Q4K-only
/// seam before dense generation was added.
pub type PatchedCachedGenerationError = PatchedGenerationError;
pub type PatchedCachedGenerationResult = PatchedGenerationResult;

fn validate_compiled_surface_effect_patch(
    manifest: &CompiledSurfaceContextManifest,
    patch: &PatchedVindex,
) -> Result<(), PatchedGenerationError> {
    manifest.validate_header().map_err(|error| {
        PatchedGenerationError::InvalidCompiledSurfaceContext(error.to_string())
    })?;
    if manifest.specs.len() != 1 {
        return Err(PatchedGenerationError::InvalidCompiledSurfaceContext(
            "C1 requires exactly one compiled surface specification".into(),
        ));
    }
    let resolved = patch.context_detectors();
    if resolved.len() != 1 {
        return Err(PatchedGenerationError::InvalidCompiledSurfaceContext(
            format!(
                "C1 effect patch must resolve exactly one linked contextual effect, found {}",
                resolved.len()
            ),
        ));
    }
    let spec = &manifest.specs[0];
    let linked = &resolved[0];
    if (linked.layer, linked.feature) != (spec.payload.layer, spec.payload.feature)
        || linked.target_ids != spec.target_ids
    {
        return Err(PatchedGenerationError::InvalidCompiledSurfaceContext(
            "C1 effect patch slot or target IDs differ from the compiled manifest".into(),
        ));
    }
    let Some(adapter) = &linked.residual_adapter else {
        return Err(PatchedGenerationError::InvalidCompiledSurfaceContext(
            "C1 effect patch has no linked terminal residual adapter".into(),
        ));
    };
    let application_matches = matches!(
        (
            adapter.application,
            spec.terminal_residual_adapter.application
        ),
        (
            larql_vindex::PatchResidualAdapterApplication::PrefillLastPositionV1,
            crate::context_detector::TerminalResidualApplication::PrefillLastPositionV1
        )
    );
    if !application_matches
        || adapter.delta != spec.terminal_residual_adapter.delta
        || adapter.calibration != spec.terminal_residual_adapter.calibration
    {
        return Err(PatchedGenerationError::InvalidCompiledSurfaceContext(
            "C1 effect patch terminal adapter differs from the compiled manifest".into(),
        ));
    }
    let row_exists = patch.override_slots_at(linked.layer).is_some_and(|slots| {
        slots
            .iter()
            .any(|slot| slot.feature == linked.feature && !slot.tombstoned)
    });
    if !row_exists {
        return Err(PatchedGenerationError::InvalidCompiledSurfaceContext(
            "C1 effect patch has no live FFN row at the linked slot".into(),
        ));
    }
    Ok(())
}

impl InferenceWeights {
    /// Generate through a product-bounded exact surface gate.
    ///
    /// This C0 seam deliberately has no FFN patch row. The bound terminal
    /// residual adapter is the complete first-token effect; an explicit exact
    /// suffix decision then authorizes `ActiveSpan`. A rejection therefore
    /// executes the source Q4K path byte-for-byte with no overlay visibility
    /// question.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_compiled_surface_cached(
        &mut self,
        tokenizer: &Tokenizer,
        prompt_ids: &[u32],
        max_tokens: usize,
        manifest: &CompiledSurfaceContextManifest,
        active_base_checksum: &str,
    ) -> Result<CompiledSurfaceGenerationResult, PatchedGenerationError> {
        self.generate_compiled_surface_cached_impl(
            tokenizer,
            prompt_ids,
            max_tokens,
            manifest,
            active_base_checksum,
            None,
        )
    }

    /// C1 product seam: exact compiled eligibility gates the complete frozen
    /// effect (linked FFN row plus terminal delta) and explicitly authorizes
    /// continuation. Learned detector coordinates in `effect_patch` are never
    /// evaluated; they only identify the row whose vectors were previously
    /// compiled and validated.
    pub fn generate_compiled_surface_with_effect_patch_cached(
        &mut self,
        tokenizer: &Tokenizer,
        effect_patch: &PatchedVindex,
        prompt_ids: &[u32],
        max_tokens: usize,
        manifest: &CompiledSurfaceContextManifest,
        active_base_checksum: &str,
    ) -> Result<CompiledSurfaceGenerationResult, PatchedGenerationError> {
        validate_compiled_surface_effect_patch(manifest, effect_patch)?;
        self.generate_compiled_surface_cached_impl(
            tokenizer,
            prompt_ids,
            max_tokens,
            manifest,
            active_base_checksum,
            Some(effect_patch),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_compiled_surface_cached_impl(
        &mut self,
        tokenizer: &Tokenizer,
        prompt_ids: &[u32],
        max_tokens: usize,
        manifest: &CompiledSurfaceContextManifest,
        active_base_checksum: &str,
        effect_patch: Option<&PatchedVindex>,
    ) -> Result<CompiledSurfaceGenerationResult, PatchedGenerationError> {
        if prompt_ids.is_empty() {
            return Err(PatchedGenerationError::EmptyPrompt);
        }
        manifest.validate_header().map_err(|error| {
            PatchedGenerationError::InvalidCompiledSurfaceContext(error.to_string())
        })?;
        let specs = &manifest.specs;
        let hidden_size = self.as_weights().hidden_size;
        let vocab_size = self.as_weights().vocab_size;
        let decisions = evaluate_compiled_surface_contexts(
            specs,
            prompt_ids,
            active_base_checksum,
            hidden_size,
            vocab_size,
        )
        .map_err(|error| {
            PatchedGenerationError::InvalidCompiledSurfaceContext(error.to_string())
        })?;
        for spec in specs {
            if spec.payload.layer >= self.as_weights().num_layers {
                return Err(PatchedGenerationError::InvalidCompiledSurfaceContext(
                    format!(
                        "logical slot L{}/F{} is outside model geometry",
                        spec.payload.layer, spec.payload.feature
                    ),
                ));
            }
        }
        let allowed_slots: Vec<_> = decisions
            .iter()
            .filter(|decision| decision.allowed())
            .map(|decision| decision.payload)
            .collect();
        if allowed_slots.len() > 1 {
            return Err(PatchedGenerationError::InvalidCompiledSurfaceContext(
                "more than one compiled surface specification fired for one prompt".into(),
            ));
        }
        let prefill_delta = compiled_surface_prefill_delta(specs, &decisions, hidden_size)
            .map_err(|error| {
                PatchedGenerationError::InvalidCompiledSurfaceContext(error.to_string())
            })?;
        if max_tokens == 0 {
            return Ok(CompiledSurfaceGenerationResult {
                generated: Vec::new(),
                decisions,
                trigger_slot: None,
                released_after_span: false,
            });
        }
        let span_specs: Vec<_> = specs
            .iter()
            .map(CompiledSurfaceContextSpec::active_span_spec)
            .collect();
        let mut constraint = ActiveSpanConstraint::with_explicit_authorization(
            span_specs,
            Vec::new(),
            allowed_slots,
        );

        let (weights, source) = self
            .as_quantised_parts_mut()
            .ok_or(PatchedGenerationError::RequiresQuantisedWeights)?;
        let generated = if let Some(patch) = effect_patch {
            let allowed = decisions
                .iter()
                .filter(|decision| decision.allowed())
                .map(|decision| (decision.payload.layer, decision.payload.feature));
            let view = PatchedKvIndexView::new(source, patch, allowed)?;
            if let Some(delta) = prefill_delta.as_deref() {
                crate::vindex::generate_kquant_cpu_constrained_cached_index_with_prefill_delta(
                    weights,
                    tokenizer,
                    prompt_ids,
                    max_tokens,
                    &view,
                    delta,
                    |generated, logits| constraint.apply(generated, logits),
                )
            } else {
                crate::vindex::generate_kquant_cpu_constrained_cached_index(
                    weights,
                    tokenizer,
                    prompt_ids,
                    max_tokens,
                    &view,
                    |generated, logits| constraint.apply(generated, logits),
                )
            }
        } else if let Some(delta) = prefill_delta.as_deref() {
            crate::vindex::generate_kquant_cpu_constrained_cached_index_with_prefill_delta(
                weights,
                tokenizer,
                prompt_ids,
                max_tokens,
                source,
                delta,
                |generated, logits| constraint.apply(generated, logits),
            )
        } else {
            crate::vindex::generate_kquant_cpu_constrained_cached_index(
                weights,
                tokenizer,
                prompt_ids,
                max_tokens,
                source,
                |generated, logits| constraint.apply(generated, logits),
            )
        }
        .ok_or(PatchedGenerationError::UnsupportedArchitecture)?;
        let generated_ids: Vec<_> = generated.iter().map(|(_, id)| *id).collect();
        let mut final_probe = vec![0.0; weights.vocab_size];
        constraint.apply(&generated_ids, &mut final_probe);
        let trigger_slot = constraint.triggered_slot();
        let released_after_span = trigger_slot.is_some() && constraint.active_slot().is_none();

        Ok(CompiledSurfaceGenerationResult {
            generated,
            decisions,
            trigger_slot,
            released_after_span,
        })
    }

    /// Generate through the best supported patched backend for the loaded
    /// storage: cached decode for native quantisation, full-prefix uncached
    /// decode for dense weights.
    pub fn generate_patched(
        &mut self,
        tokenizer: &Tokenizer,
        patched: &PatchedVindex,
        prompt_ids: &[u32],
        max_tokens: usize,
        span_specs: &[ActiveSpanSpec],
    ) -> Result<PatchedGenerationResult, PatchedGenerationError> {
        if self.is_quantised() {
            self.generate_patched_cached(tokenizer, patched, prompt_ids, max_tokens, span_specs)
        } else {
            self.generate_patched_dense_uncached(
                tokenizer, patched, prompt_ids, max_tokens, span_specs,
            )
        }
    }

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
        if max_tokens == 0 {
            return Ok(PatchedCachedGenerationResult {
                generated: Vec::new(),
                context_decisions: Vec::new(),
                trigger_slot: None,
                released_after_span: false,
            });
        }
        if prompt_ids.is_empty() {
            return Err(PatchedCachedGenerationError::EmptyPrompt);
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
        let prefill_delta = terminal_prefill_delta(
            &detectors,
            &evidence.context_decisions,
            self.as_weights().hidden_size,
        )
        .map_err(PatchedCachedGenerationError::InvalidTerminalResidualAdapter)?;
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
        let generated = if let Some(delta) = prefill_delta.as_deref() {
            crate::vindex::generate_kquant_cpu_constrained_cached_index_with_prefill_delta(
                weights,
                tokenizer,
                prompt_ids,
                max_tokens,
                &view,
                delta,
                |generated, logits| constraint.apply(generated, logits),
            )
        } else {
            crate::vindex::generate_kquant_cpu_constrained_cached_index(
                weights,
                tokenizer,
                prompt_ids,
                max_tokens,
                &view,
                |generated, logits| constraint.apply(generated, logits),
            )
        }
        .ok_or(PatchedCachedGenerationError::UnsupportedArchitecture)?;
        let generated_ids: Vec<_> = generated.iter().map(|(_, id)| *id).collect();
        let mut final_probe = vec![0.0; weights.vocab_size];
        constraint.apply(&generated_ids, &mut final_probe);
        let trigger_slot = constraint.triggered_slot();
        let released_after_span = trigger_slot.is_some() && constraint.active_slot().is_none();

        Ok(PatchedCachedGenerationResult {
            generated,
            context_decisions,
            trigger_slot,
            released_after_span,
        })
    }

    /// Uncached dense counterpart of [`Self::generate_patched_cached`].
    ///
    /// The initial patched prefill fixes contextual eligibility and live slot
    /// evidence for the whole sequence. Exact continuation tokens do not need
    /// a redundant forward; once the span releases, subsequent natural tokens
    /// use full dense forwards with the original decisions frozen.
    pub fn generate_patched_dense_uncached(
        &mut self,
        tokenizer: &Tokenizer,
        patched: &PatchedVindex,
        prompt_ids: &[u32],
        max_tokens: usize,
        span_specs: &[ActiveSpanSpec],
    ) -> Result<PatchedCachedGenerationResult, PatchedCachedGenerationError> {
        let Self::Dense(weights) = self else {
            return Err(PatchedCachedGenerationError::RequiresDenseWeights);
        };
        if max_tokens == 0 {
            return Ok(PatchedCachedGenerationResult {
                generated: Vec::new(),
                context_decisions: Vec::new(),
                trigger_slot: None,
                released_after_span: false,
            });
        }
        if prompt_ids.is_empty() {
            return Err(PatchedCachedGenerationError::EmptyPrompt);
        }

        let detectors: Vec<ContextDetectorSpec> = patched
            .context_detectors()
            .into_iter()
            .map(Into::into)
            .collect();
        let (evidence, first_model_ids) = infer_patched_contextual_with_model_ids(
            weights, tokenizer, patched, prompt_ids, 1, &detectors, None,
        );
        let mut events = patch_activation_events(weights, patched, &evidence.infer.residuals);
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
        let mut decode_detectors = detectors.clone();
        for detector in &mut decode_detectors {
            detector.terminal_residual_adapter = None;
        }
        let first_id = first_model_ids
            .first()
            .copied()
            .ok_or(PatchedCachedGenerationError::NoModelToken)?;
        let mut constraint = ActiveSpanConstraint::new(span_specs.to_vec(), events);
        let mut generated_ids = Vec::with_capacity(max_tokens);
        let mut full_ids = prompt_ids.to_vec();
        let mut generated = Vec::with_capacity(max_tokens);
        let eos = crate::layer_graph::EosConfig::builtin();

        for step in 0..max_tokens {
            let id = if step == 0 {
                first_id
            } else {
                let mut mask_probe = vec![0.0; weights.vocab_size];
                constraint.apply(&generated_ids, &mut mask_probe);
                if constraint.active_slot().is_some() {
                    mask_probe
                        .iter()
                        .position(|logit| logit.is_finite())
                        .map(|id| id as u32)
                        .ok_or(PatchedCachedGenerationError::NoModelToken)?
                } else {
                    let (_, next_model_ids) = infer_patched_contextual_with_model_ids(
                        weights,
                        tokenizer,
                        patched,
                        &full_ids,
                        1,
                        &decode_detectors,
                        Some(&context_decisions),
                    );
                    next_model_ids
                        .first()
                        .copied()
                        .ok_or(PatchedCachedGenerationError::NoModelToken)?
                }
            };
            let token = tokenizer.decode(&[id], true).unwrap_or_default();
            let stop = eos.is_eos_with_tokenizer(id, &token, tokenizer);
            generated.push((token, id));
            generated_ids.push(id);
            full_ids.push(id);
            if stop {
                break;
            }
        }

        // Observe the final token before returning so request-local span state
        // is released even when max_tokens ends exactly on the target.
        let mut final_probe = vec![0.0; weights.vocab_size];
        constraint.apply(&generated_ids, &mut final_probe);
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
    use larql_vindex::patch::core::{encode_gate_vector, PatchDownMeta};
    use larql_vindex::{
        load_vindex_config, FeatureMeta, PatchContextDetector, PatchContextRepresentation, PatchOp,
        PatchResidualAdapter, PatchResidualAdapterApplication, SilentLoadCallbacks, VindexPatch,
    };

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
    fn quantised_patched_dense_generation_fails_explicitly() {
        let (mut iw, tokenizer) = quantised_fixture();
        let source = match &iw {
            InferenceWeights::Quantised { index, .. } => index.clone(),
            InferenceWeights::Dense(_) => unreachable!(),
        };
        let patched = PatchedVindex::new(source);
        let error = iw
            .generate_patched_dense_uncached(&tokenizer, &patched, &[0, 1], 1, &[])
            .expect_err("quantised weights cannot use dense generation");
        assert!(matches!(
            error,
            PatchedCachedGenerationError::RequiresDenseWeights
        ));
    }

    #[test]
    fn dense_patched_generation_emits_and_releases_exact_span() {
        let (mut iw, tokenizer) = dense_fixture();
        let base = make_test_vindex(iw.as_weights());
        let baseline = iw.infer_patched(
            &tokenizer,
            &base,
            None,
            &[0, 1],
            1,
            &crate::forward::KnnRouteMode::Legacy,
        );
        let layer = 0;
        let feature = 3;
        let residual = baseline
            .residuals
            .iter()
            .find(|(candidate, _)| *candidate == layer)
            .map(|(_, residual)| residual.clone())
            .expect("layer-zero residual");

        let mut patched = PatchedVindex::new(base);
        patched.insert_feature(
            layer,
            feature,
            residual.clone(),
            FeatureMeta {
                top_token: "target".into(),
                top_token_id: 0,
                c_score: 1.0,
                top_k: Vec::new(),
            },
        );
        patched.set_up_vector(layer, feature, residual);
        patched.set_down_vector(layer, feature, vec![0.01; iw.as_weights().hidden_size]);

        let natural = iw
            .generate_patched_dense_uncached(&tokenizer, &patched, &[0, 1], 1, &[])
            .expect("dense natural first token");
        let first = natural.generated[0].1;
        let forced = (first + 1) % iw.as_weights().vocab_size as u32;
        let spec = ActiveSpanSpec {
            slot: PatchSlotId { layer, feature },
            target_ids: vec![first, forced],
        };

        let result = iw
            .generate_patched(&tokenizer, &patched, &[0, 1], 3, &[spec])
            .expect("backend-neutral dense exact span generation");
        let ids: Vec<_> = result.generated.iter().map(|(_, id)| *id).collect();
        assert_eq!(&ids[..2], &[first, forced]);
        assert_eq!(result.trigger_slot, Some(PatchSlotId { layer, feature }));
        assert!(result.released_after_span);
        assert!(result.context_decisions.is_empty());
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
            .generate_patched(&tokenizer, &patched, &[0, 1], 2, &[])
            .expect("generic dispatch selects the cached overlay seam");
        assert_eq!(result.generated.len(), 2);
        assert!(result.context_decisions.is_empty());
        assert_eq!(result.trigger_slot, None);
        assert!(!result.released_after_span);
    }

    fn compiled_fixture_spec(
        weights: &InferenceWeights,
        first_target_id: u32,
    ) -> CompiledSurfaceContextSpec {
        CompiledSurfaceContextSpec {
            base_checksum: "fixture-checksum".into(),
            payload: PatchSlotId {
                layer: 0,
                feature: 3,
            },
            target_ids: vec![
                first_target_id,
                (first_target_id + 1) % weights.as_weights().vocab_size as u32,
            ],
            prompt_suffixes: vec![vec![1]],
            terminal_residual_adapter: crate::context_detector::TerminalResidualAdapterSpec {
                application:
                    crate::context_detector::TerminalResidualApplication::PrefillLastPositionV1,
                // Non-zero and valid, but too small to perturb the synthetic
                // model's naturally selected first ID.
                delta: vec![f32::MIN_POSITIVE; weights.as_weights().hidden_size],
                calibration: Some("synthetic-c0".into()),
            },
            provenance: Some("unit fixture".into()),
        }
    }

    fn compiled_fixture_manifest(
        weights: &InferenceWeights,
        first_target_id: u32,
    ) -> CompiledSurfaceContextManifest {
        CompiledSurfaceContextManifest::new(
            "synthetic-c0".into(),
            vec![compiled_fixture_spec(weights, first_target_id)],
        )
    }

    fn compiled_effect_patch(
        base: VectorIndex,
        manifest: &CompiledSurfaceContextManifest,
        hidden_size: usize,
    ) -> PatchedVindex {
        let spec = &manifest.specs[0];
        let zeros = encode_gate_vector(&vec![0.0; hidden_size]);
        let patch = VindexPatch {
            version: 1,
            base_model: "fixture".into(),
            base_checksum: Some("fixture-checksum".into()),
            created_at: "fixture".into(),
            description: Some("synthetic C1 effect".into()),
            author: None,
            tags: vec!["product-bounded".into()],
            operations: vec![
                PatchOp::Insert {
                    layer: spec.payload.layer,
                    feature: spec.payload.feature,
                    relation: Some("fixture".into()),
                    entity: "fixture".into(),
                    target: "target".into(),
                    confidence: Some(1.0),
                    gate_vector_b64: Some(zeros.clone()),
                    up_vector_b64: Some(zeros.clone()),
                    down_vector_b64: Some(zeros.clone()),
                    down_meta: Some(PatchDownMeta {
                        top_token: "target".into(),
                        top_token_id: spec.target_ids[0],
                        c_score: 1.0,
                    }),
                },
                PatchOp::Context {
                    layer: spec.payload.layer,
                    feature: spec.payload.feature,
                    target_ids: spec.target_ids.clone(),
                    detector: PatchContextDetector {
                        detector_layer: 0,
                        representation: PatchContextRepresentation::MeanNoBos,
                        center_vector_b64: zeros.clone(),
                        entity_direction_b64: zeros.clone(),
                        relation_direction_b64: zeros,
                        entity_threshold: 0.0,
                        relation_threshold: 0.0,
                        calibration: Some("ignored-by-C1".into()),
                    },
                    residual_adapter: Some(PatchResidualAdapter {
                        application: PatchResidualAdapterApplication::PrefillLastPositionV1,
                        delta_vector_b64: encode_gate_vector(&spec.terminal_residual_adapter.delta),
                        calibration: spec.terminal_residual_adapter.calibration.clone(),
                    }),
                },
            ],
        };
        let mut patched = PatchedVindex::new(base);
        patched
            .try_apply_patch_bound(patch, Some("fixture-checksum"))
            .unwrap();
        patched
    }

    #[test]
    fn compiled_surface_cached_generation_fires_forces_and_releases() {
        let (mut iw, tokenizer) = quantised_fixture();
        let source = match &iw {
            InferenceWeights::Quantised { index, .. } => index.clone(),
            InferenceWeights::Dense(_) => unreachable!(),
        };
        let empty_patch = PatchedVindex::new(source);
        let natural = iw
            .generate_patched(&tokenizer, &empty_patch, &[0, 1], 1, &[])
            .expect("natural Q4K first token");
        let first = natural.generated[0].1;
        let manifest = compiled_fixture_manifest(&iw, first);
        let spec = &manifest.specs[0];

        let result = iw
            .generate_compiled_surface_cached(&tokenizer, &[0, 1], 3, &manifest, "fixture-checksum")
            .expect("compiled surface generation");
        let ids: Vec<_> = result.generated.iter().map(|(_, id)| *id).collect();
        assert_eq!(ids[0], spec.target_ids[0], "first token remains natural");
        assert_eq!(ids[1], spec.target_ids[1], "continuation is exact");
        assert!(result.decisions[0].allowed());
        assert_eq!(result.trigger_slot, Some(spec.payload));
        assert!(result.released_after_span);
    }

    #[test]
    fn compiled_surface_rejection_is_bit_identical_to_base_generation() {
        let (mut baseline_weights, tokenizer) = quantised_fixture();
        let (mut compiled_weights, _) = quantised_fixture();
        let source = match &baseline_weights {
            InferenceWeights::Quantised { index, .. } => index.clone(),
            InferenceWeights::Dense(_) => unreachable!(),
        };
        let empty_patch = PatchedVindex::new(source);
        let baseline = baseline_weights
            .generate_patched(&tokenizer, &empty_patch, &[0, 2], 3, &[])
            .expect("base generation");
        let first = baseline.generated[0].1;
        let manifest = compiled_fixture_manifest(&compiled_weights, first);

        let rejected = compiled_weights
            .generate_compiled_surface_cached(&tokenizer, &[0, 2], 3, &manifest, "fixture-checksum")
            .expect("compiled rejection");
        assert_eq!(rejected.generated, baseline.generated);
        assert!(!rejected.decisions[0].allowed());
        assert_eq!(rejected.trigger_slot, None);
        assert!(!rejected.released_after_span);
    }

    #[test]
    fn compiled_surface_generation_rejects_multiple_fires() {
        let (mut iw, tokenizer) = quantised_fixture();
        let mut left = compiled_fixture_spec(&iw, 1);
        let mut right = compiled_fixture_spec(&iw, 2);
        left.payload.feature = 3;
        right.payload.feature = 4;

        let manifest =
            CompiledSurfaceContextManifest::new("synthetic-collision".into(), vec![left, right]);
        let error = iw
            .generate_compiled_surface_cached(&tokenizer, &[0, 1], 1, &manifest, "fixture-checksum")
            .expect_err("ambiguous compiled surface must fail closed");
        assert!(matches!(
            error,
            PatchedGenerationError::InvalidCompiledSurfaceContext(_)
        ));
    }

    #[test]
    fn compiled_surface_c1_links_effect_row_and_exact_continuation() {
        let (mut iw, tokenizer) = quantised_fixture();
        let source = match &iw {
            InferenceWeights::Quantised { index, .. } => index.clone(),
            InferenceWeights::Dense(_) => unreachable!(),
        };
        let natural = iw
            .generate_patched(
                &tokenizer,
                &PatchedVindex::new(source.clone()),
                &[0, 1],
                1,
                &[],
            )
            .unwrap();
        let mut manifest = compiled_fixture_manifest(&iw, natural.generated[0].1);
        manifest.specs[0].payload = PatchSlotId {
            layer: 1,
            feature: 3,
        };
        let patch = compiled_effect_patch(source, &manifest, iw.as_weights().hidden_size);

        let result = iw
            .generate_compiled_surface_with_effect_patch_cached(
                &tokenizer,
                &patch,
                &[0, 1],
                3,
                &manifest,
                "fixture-checksum",
            )
            .unwrap();
        let ids: Vec<_> = result.generated.iter().map(|(_, id)| *id).collect();
        assert_eq!(&ids[..2], manifest.specs[0].target_ids.as_slice());
        assert_eq!(result.trigger_slot, Some(manifest.specs[0].payload));
        assert!(result.released_after_span);

        let mut mismatched = manifest.clone();
        mismatched.specs[0].terminal_residual_adapter.delta[0] *= 2.0;
        assert!(iw
            .generate_compiled_surface_with_effect_patch_cached(
                &tokenizer,
                &patch,
                &[0, 1],
                1,
                &mismatched,
                "fixture-checksum",
            )
            .is_err());
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
