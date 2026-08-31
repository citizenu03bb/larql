//! Zero-weight compiler contract for exact product-bounded surface patches.
//!
//! Compilation has two domain outcomes: [`CertifiedPatch`] or
//! [`UncompilableEdit`]. Every refusal is structured and auditable. A caller
//! may still fail to persist an already-certified artifact; that failure is a
//! `commit`-stage refusal marked retryable, never a certified patch.

use std::collections::BTreeSet;
use std::path::Path;

use larql_inference::active_span::PatchSlotId;
use larql_inference::compiled_surface_context::{
    CompiledSurfaceContextManifest, CompiledSurfaceContextSpec, PRODUCT_BOUNDED_CLAIM,
};
use larql_inference::context_detector::{TerminalResidualAdapterSpec, TerminalResidualApplication};
use larql_vindex::patch::core::{base_checksum_from_manifest, decode_gate_vector};
use larql_vindex::{PatchOp, PatchResidualAdapterApplication, VindexPatch};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const PRODUCT_SURFACE_COMPILER_CONTRACT_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProductSurfaceCompilationStage {
    Request,
    ArtifactBinding,
    EffectExtraction,
    SurfaceCompilation,
    Certification,
    Commit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UncompilableEditCode {
    InvalidInvocation,
    InputUnavailable,
    InvalidVindex,
    UnsupportedArchitecture,
    MissingBaseChecksum,
    InvalidSourcePatch,
    SourceBindingMismatch,
    AmbiguousSourceEffect,
    InvalidSourceEffect,
    SourcePayloadGeometryMismatch,
    InvalidSurfaceBank,
    PromptEncodingFailed,
    CertificationFailed,
    OutputCommitFailed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UncompilableEdit {
    pub contract_version: u32,
    pub claim_label: String,
    pub stage: ProductSurfaceCompilationStage,
    pub code: UncompilableEditCode,
    pub reason: String,
    /// True only when the same edit may compile after fixing environment or I/O.
    pub retryable: bool,
}

impl UncompilableEdit {
    pub fn new(
        stage: ProductSurfaceCompilationStage,
        code: UncompilableEditCode,
        reason: impl Into<String>,
        retryable: bool,
    ) -> Self {
        Self {
            contract_version: PRODUCT_SURFACE_COMPILER_CONTRACT_VERSION,
            claim_label: PRODUCT_BOUNDED_CLAIM.into(),
            stage,
            code,
            reason: reason.into(),
            retryable,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProductSurfaceCertificationReport {
    pub weights_loaded: bool,
    pub supported_prompts: usize,
    pub target_ids: Vec<u32>,
    pub logical_slot: PatchSlotId,
    pub checks: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CertifiedPatch {
    pub contract_version: u32,
    pub claim_label: String,
    pub protocol_id: String,
    pub base_checksum: String,
    pub source_patch_sha256: String,
    pub surface_bank_sha256: String,
    pub compiled_manifest_sha256: String,
    pub manifest: CompiledSurfaceContextManifest,
    pub report: ProductSurfaceCertificationReport,
}

impl CertifiedPatch {
    /// Recreate the exact certified bytes. A mismatch means the in-memory
    /// certificate was mutated and must not be persisted.
    pub fn manifest_bytes(&self) -> Result<Vec<u8>, UncompilableEdit> {
        let bytes = serde_json::to_vec_pretty(&self.manifest).map_err(|error| {
            UncompilableEdit::new(
                ProductSurfaceCompilationStage::Certification,
                UncompilableEditCode::CertificationFailed,
                format!("compiled manifest serialization failed: {error}"),
                false,
            )
        })?;
        let actual = sha256_bytes(&bytes);
        if actual != self.compiled_manifest_sha256 {
            return Err(UncompilableEdit::new(
                ProductSurfaceCompilationStage::Certification,
                UncompilableEditCode::CertificationFailed,
                format!(
                    "compiled manifest digest changed after certification: expected {}, got {actual}",
                    self.compiled_manifest_sha256
                ),
                false,
            ));
        }
        Ok(bytes)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ProductSurfaceCompileOutcome {
    CertifiedPatch {
        certified_patch: Box<CertifiedPatch>,
    },
    UncompilableEdit {
        refusal: UncompilableEdit,
    },
}

impl ProductSurfaceCompileOutcome {
    pub fn refused(refusal: UncompilableEdit) -> Self {
        Self::UncompilableEdit { refusal }
    }

    pub fn certified_patch(&self) -> Option<&CertifiedPatch> {
        match self {
            Self::CertifiedPatch { certified_patch } => Some(certified_patch),
            Self::UncompilableEdit { .. } => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ProductSurfaceCompileRequest<'a> {
    pub vindex: &'a Path,
    pub source_patch: &'a Path,
    pub surface_bank: &'a Path,
}

#[derive(Debug, Deserialize)]
struct SurfaceBank {
    protocol_id: String,
    supported_prompts: Vec<String>,
}

struct SourceAdapter {
    payload: PatchSlotId,
    target_ids: Vec<u32>,
    adapter: TerminalResidualAdapterSpec,
}

#[derive(Debug)]
enum SourceAdapterError {
    Ambiguous(usize),
    Invalid(String),
}

#[derive(Debug)]
enum PromptCompilationError {
    InvalidBank(String),
    Encoding(String),
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn refusal(
    stage: ProductSurfaceCompilationStage,
    code: UncompilableEditCode,
    reason: impl Into<String>,
    retryable: bool,
) -> ProductSurfaceCompileOutcome {
    ProductSurfaceCompileOutcome::refused(UncompilableEdit::new(stage, code, reason, retryable))
}

fn extract_source_adapter(
    patch: &VindexPatch,
    hidden_size: usize,
) -> Result<SourceAdapter, SourceAdapterError> {
    let candidates: Vec<_> = patch
        .operations
        .iter()
        .filter_map(|operation| match operation {
            PatchOp::Context {
                layer,
                feature,
                target_ids,
                residual_adapter: Some(adapter),
                ..
            } => Some((*layer, *feature, target_ids, adapter)),
            _ => None,
        })
        .collect();
    if candidates.len() != 1 {
        return Err(SourceAdapterError::Ambiguous(candidates.len()));
    }
    let (layer, feature, target_ids, adapter) = candidates[0];
    let application = match adapter.application {
        PatchResidualAdapterApplication::PrefillLastPositionV1 => {
            TerminalResidualApplication::PrefillLastPositionV1
        }
    };
    let delta = decode_gate_vector(&adapter.delta_vector_b64)
        .map_err(|error| SourceAdapterError::Invalid(error.to_string()))?;
    if delta.len() != hidden_size {
        return Err(SourceAdapterError::Invalid(format!(
            "source terminal adapter has {} values, expected hidden size {hidden_size}",
            delta.len()
        )));
    }
    Ok(SourceAdapter {
        payload: PatchSlotId { layer, feature },
        target_ids: target_ids.clone(),
        adapter: TerminalResidualAdapterSpec {
            application,
            delta,
            calibration: adapter.calibration.clone(),
        },
    })
}

fn compile_prompt_suffixes<F>(
    prompts: &[String],
    mut encode: F,
) -> Result<Vec<Vec<u32>>, PromptCompilationError>
where
    F: FnMut(&str) -> Result<Vec<u32>, String>,
{
    if prompts.is_empty() {
        return Err(PromptCompilationError::InvalidBank(
            "surface bank contains no supported prompts".into(),
        ));
    }
    let mut seen_prompts = BTreeSet::new();
    let mut seen_ids = BTreeSet::new();
    let mut suffixes = Vec::with_capacity(prompts.len());
    for (index, prompt) in prompts.iter().enumerate() {
        if prompt.is_empty() || prompt.trim() != prompt {
            return Err(PromptCompilationError::InvalidBank(format!(
                "supported prompt {index} must be non-empty and have no outer whitespace"
            )));
        }
        if !seen_prompts.insert(prompt) {
            return Err(PromptCompilationError::InvalidBank(format!(
                "supported prompt {index} duplicates an earlier prompt"
            )));
        }
        let ids = encode(prompt).map_err(PromptCompilationError::Encoding)?;
        if ids.is_empty() {
            return Err(PromptCompilationError::InvalidBank(format!(
                "supported prompt {index} encoded to no IDs"
            )));
        }
        if !seen_ids.insert(ids.clone()) {
            return Err(PromptCompilationError::InvalidBank(format!(
                "supported prompt {index} tokenizes identically to an earlier prompt"
            )));
        }
        suffixes.push(ids);
    }
    Ok(suffixes)
}

/// Compile one exact-surface edit without loading model weights.
///
/// This function always returns one of the two domain outcomes. Filesystem and
/// parsing failures are explicit refusals; callers decide whether a retryable
/// refusal should be attempted again after repairing the environment.
pub fn compile_product_surface_context(
    request: ProductSurfaceCompileRequest<'_>,
) -> ProductSurfaceCompileOutcome {
    let config = match larql_vindex::load_vindex_config(request.vindex) {
        Ok(config) => config,
        Err(error) => {
            return refusal(
                ProductSurfaceCompilationStage::ArtifactBinding,
                UncompilableEditCode::InvalidVindex,
                format!(
                    "cannot load vindex configuration {}: {error}",
                    request.vindex.display()
                ),
                true,
            );
        }
    };
    let Some(arch) = larql_vindex::arch_from_vindex_config(&config) else {
        return refusal(
            ProductSurfaceCompilationStage::ArtifactBinding,
            UncompilableEditCode::UnsupportedArchitecture,
            "vindex has no model_config for prompt compilation",
            false,
        );
    };
    let Some(active_checksum) = base_checksum_from_manifest(config.checksums.as_ref()) else {
        return refusal(
            ProductSurfaceCompilationStage::ArtifactBinding,
            UncompilableEditCode::MissingBaseChecksum,
            "vindex has no non-empty checksum manifest",
            false,
        );
    };
    let tokenizer = match larql_vindex::load_vindex_tokenizer(request.vindex) {
        Ok(tokenizer) => tokenizer,
        Err(error) => {
            return refusal(
                ProductSurfaceCompilationStage::ArtifactBinding,
                UncompilableEditCode::InvalidVindex,
                format!("cannot load active vindex tokenizer: {error}"),
                false,
            );
        }
    };

    let adapter_bytes = match std::fs::read(request.source_patch) {
        Ok(bytes) => bytes,
        Err(error) => {
            return refusal(
                ProductSurfaceCompilationStage::Request,
                UncompilableEditCode::InputUnavailable,
                format!(
                    "cannot read source patch {}: {error}",
                    request.source_patch.display()
                ),
                true,
            );
        }
    };
    let source: VindexPatch = match serde_json::from_slice(&adapter_bytes) {
        Ok(source) => source,
        Err(error) => {
            return refusal(
                ProductSurfaceCompilationStage::EffectExtraction,
                UncompilableEditCode::InvalidSourcePatch,
                format!("source patch is not valid VLP JSON: {error}"),
                false,
            );
        }
    };
    if source.base_checksum.as_deref() != Some(active_checksum.as_str()) {
        return refusal(
            ProductSurfaceCompilationStage::ArtifactBinding,
            UncompilableEditCode::SourceBindingMismatch,
            "source adapter patch is not bound to the active vindex checksum",
            false,
        );
    }
    let source_adapter = match extract_source_adapter(&source, config.hidden_size) {
        Ok(adapter) => adapter,
        Err(SourceAdapterError::Ambiguous(count)) => {
            return refusal(
                ProductSurfaceCompilationStage::EffectExtraction,
                UncompilableEditCode::AmbiguousSourceEffect,
                format!(
                    "source patch must contain exactly one contextual terminal adapter, found {count}"
                ),
                false,
            );
        }
        Err(SourceAdapterError::Invalid(reason)) => {
            return refusal(
                ProductSurfaceCompilationStage::EffectExtraction,
                UncompilableEditCode::InvalidSourceEffect,
                reason,
                false,
            );
        }
    };
    if source_adapter.payload.layer >= config.num_layers
        || source_adapter.payload.feature >= config.intermediate_size
    {
        return refusal(
            ProductSurfaceCompilationStage::EffectExtraction,
            UncompilableEditCode::SourcePayloadGeometryMismatch,
            format!(
                "source adapter logical slot L{}/F{} is outside active model geometry",
                source_adapter.payload.layer, source_adapter.payload.feature
            ),
            false,
        );
    }

    let surfaces_bytes = match std::fs::read(request.surface_bank) {
        Ok(bytes) => bytes,
        Err(error) => {
            return refusal(
                ProductSurfaceCompilationStage::Request,
                UncompilableEditCode::InputUnavailable,
                format!(
                    "cannot read surface bank {}: {error}",
                    request.surface_bank.display()
                ),
                true,
            );
        }
    };
    let bank: SurfaceBank = match serde_json::from_slice(&surfaces_bytes) {
        Ok(bank) => bank,
        Err(error) => {
            return refusal(
                ProductSurfaceCompilationStage::SurfaceCompilation,
                UncompilableEditCode::InvalidSurfaceBank,
                format!("surface bank is not valid JSON: {error}"),
                false,
            );
        }
    };
    if bank.protocol_id.trim().is_empty() {
        return refusal(
            ProductSurfaceCompilationStage::SurfaceCompilation,
            UncompilableEditCode::InvalidSurfaceBank,
            "surface bank requires a protocol_id",
            false,
        );
    }
    let prompt_suffixes = match compile_prompt_suffixes(&bank.supported_prompts, |prompt| {
        larql_inference::encode_prompt(&tokenizer, arch.as_ref(), prompt)
            .map_err(|error| error.to_string())
    }) {
        Ok(suffixes) => suffixes,
        Err(PromptCompilationError::InvalidBank(reason)) => {
            return refusal(
                ProductSurfaceCompilationStage::SurfaceCompilation,
                UncompilableEditCode::InvalidSurfaceBank,
                reason,
                false,
            );
        }
        Err(PromptCompilationError::Encoding(reason)) => {
            return refusal(
                ProductSurfaceCompilationStage::SurfaceCompilation,
                UncompilableEditCode::PromptEncodingFailed,
                reason,
                false,
            );
        }
    };

    let spec = CompiledSurfaceContextSpec {
        base_checksum: active_checksum.clone(),
        payload: source_adapter.payload,
        target_ids: source_adapter.target_ids,
        prompt_suffixes,
        terminal_residual_adapter: source_adapter.adapter,
        provenance: Some(format!(
            "source_adapter_sha256={};surface_bank_sha256={}",
            sha256_bytes(&adapter_bytes),
            sha256_bytes(&surfaces_bytes)
        )),
    };
    if let Err(error) = spec.validate(&active_checksum, config.hidden_size, config.vocab_size) {
        return refusal(
            ProductSurfaceCompilationStage::Certification,
            UncompilableEditCode::CertificationFailed,
            error.to_string(),
            false,
        );
    }
    let manifest = CompiledSurfaceContextManifest::new(bank.protocol_id, vec![spec]);
    if let Err(error) = manifest.validate_header() {
        return refusal(
            ProductSurfaceCompilationStage::Certification,
            UncompilableEditCode::CertificationFailed,
            error.to_string(),
            false,
        );
    }
    let manifest_bytes = match serde_json::to_vec_pretty(&manifest) {
        Ok(bytes) => bytes,
        Err(error) => {
            return refusal(
                ProductSurfaceCompilationStage::Certification,
                UncompilableEditCode::CertificationFailed,
                format!("compiled manifest serialization failed: {error}"),
                false,
            );
        }
    };
    let reparsed: CompiledSurfaceContextManifest = match serde_json::from_slice(&manifest_bytes) {
        Ok(reparsed) => reparsed,
        Err(error) => {
            return refusal(
                ProductSurfaceCompilationStage::Certification,
                UncompilableEditCode::CertificationFailed,
                format!("compiled manifest round-trip failed: {error}"),
                false,
            );
        }
    };
    if reparsed != manifest {
        return refusal(
            ProductSurfaceCompilationStage::Certification,
            UncompilableEditCode::CertificationFailed,
            "compiled manifest changed during serialization round-trip",
            false,
        );
    }
    if let Err(error) = reparsed.validate_header() {
        return refusal(
            ProductSurfaceCompilationStage::Certification,
            UncompilableEditCode::CertificationFailed,
            error.to_string(),
            false,
        );
    }
    for candidate in &reparsed.specs {
        if let Err(error) =
            candidate.validate(&active_checksum, config.hidden_size, config.vocab_size)
        {
            return refusal(
                ProductSurfaceCompilationStage::Certification,
                UncompilableEditCode::CertificationFailed,
                error.to_string(),
                false,
            );
        }
    }

    let source_patch_sha256 = sha256_bytes(&adapter_bytes);
    let surface_bank_sha256 = sha256_bytes(&surfaces_bytes);
    let compiled_manifest_sha256 = sha256_bytes(&manifest_bytes);
    let report = ProductSurfaceCertificationReport {
        weights_loaded: false,
        supported_prompts: bank.supported_prompts.len(),
        target_ids: manifest.specs[0].target_ids.clone(),
        logical_slot: manifest.specs[0].payload,
        checks: vec![
            "active_base_checksum".into(),
            "source_patch_binding".into(),
            "logical_slot_geometry".into(),
            "target_vocabulary".into(),
            "surface_uniqueness".into(),
            "terminal_adapter_geometry".into(),
            "manifest_round_trip".into(),
        ],
    };
    ProductSurfaceCompileOutcome::CertifiedPatch {
        certified_patch: Box::new(CertifiedPatch {
            contract_version: PRODUCT_SURFACE_COMPILER_CONTRACT_VERSION,
            claim_label: PRODUCT_BOUNDED_CLAIM.into(),
            protocol_id: manifest.protocol_id.clone(),
            base_checksum: active_checksum,
            source_patch_sha256,
            surface_bank_sha256,
            compiled_manifest_sha256,
            manifest,
            report,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn certified_fixture() -> CertifiedPatch {
        let manifest = CompiledSurfaceContextManifest::new(
            "fixture-protocol".into(),
            vec![CompiledSurfaceContextSpec {
                base_checksum: "fixture-checksum".into(),
                payload: PatchSlotId {
                    layer: 1,
                    feature: 2,
                },
                target_ids: vec![3, 4],
                prompt_suffixes: vec![vec![5, 6]],
                terminal_residual_adapter: TerminalResidualAdapterSpec {
                    application: TerminalResidualApplication::PrefillLastPositionV1,
                    delta: vec![0.5, -0.5],
                    calibration: Some("fixture".into()),
                },
                provenance: Some("fixture".into()),
            }],
        );
        let compiled_manifest_sha256 = sha256_bytes(&serde_json::to_vec_pretty(&manifest).unwrap());
        CertifiedPatch {
            contract_version: PRODUCT_SURFACE_COMPILER_CONTRACT_VERSION,
            claim_label: PRODUCT_BOUNDED_CLAIM.into(),
            protocol_id: manifest.protocol_id.clone(),
            base_checksum: "fixture-checksum".into(),
            source_patch_sha256: "source".into(),
            surface_bank_sha256: "surfaces".into(),
            compiled_manifest_sha256,
            report: ProductSurfaceCertificationReport {
                weights_loaded: false,
                supported_prompts: 1,
                target_ids: vec![3, 4],
                logical_slot: PatchSlotId {
                    layer: 1,
                    feature: 2,
                },
                checks: vec!["fixture".into()],
            },
            manifest,
        }
    }

    #[test]
    fn prompt_compiler_preserves_order_and_rejects_duplicates() {
        let prompts = vec!["first".to_string(), "second".to_string()];
        let suffixes = compile_prompt_suffixes(&prompts, |prompt| {
            Ok(prompt.bytes().map(u32::from).collect())
        })
        .unwrap();
        assert_eq!(suffixes[0][0], u32::from(b'f'));
        assert_eq!(suffixes[1][0], u32::from(b's'));

        let duplicates = vec!["same".to_string(), "same".to_string()];
        assert!(matches!(
            compile_prompt_suffixes(&duplicates, |_| Ok(vec![1])),
            Err(PromptCompilationError::InvalidBank(_))
        ));
    }

    #[test]
    fn missing_input_is_a_retryable_uncompilable_edit() {
        let missing = Path::new("/definitely/missing/product-surface-vindex");
        let outcome = compile_product_surface_context(ProductSurfaceCompileRequest {
            vindex: missing,
            source_patch: missing,
            surface_bank: missing,
        });
        assert!(matches!(
            outcome,
            ProductSurfaceCompileOutcome::UncompilableEdit {
                refusal: UncompilableEdit {
                    stage: ProductSurfaceCompilationStage::ArtifactBinding,
                    code: UncompilableEditCode::InvalidVindex,
                    retryable: true,
                    ..
                }
            }
        ));
    }

    #[test]
    fn outcomes_have_stable_serialized_status_and_reason_code() {
        let outcome = refusal(
            ProductSurfaceCompilationStage::SurfaceCompilation,
            UncompilableEditCode::InvalidSurfaceBank,
            "duplicate surface",
            false,
        );
        let json = serde_json::to_value(outcome).unwrap();
        assert_eq!(json["status"], "uncompilable_edit");
        assert_eq!(json["refusal"]["code"], "invalid_surface_bank");
        assert_eq!(json["refusal"]["retryable"], false);

        let certified = ProductSurfaceCompileOutcome::CertifiedPatch {
            certified_patch: Box::new(certified_fixture()),
        };
        let json = serde_json::to_value(certified).unwrap();
        assert_eq!(json["status"], "certified_patch");
        assert_eq!(json["certified_patch"]["report"]["weights_loaded"], false);
    }

    #[test]
    fn certified_manifest_digest_prevents_post_certification_mutation() {
        let mut certified = certified_fixture();
        certified.manifest_bytes().unwrap();

        certified.manifest.protocol_id.push_str("-mutated");
        let refusal = certified.manifest_bytes().unwrap_err();
        assert_eq!(refusal.stage, ProductSurfaceCompilationStage::Certification);
        assert_eq!(refusal.code, UncompilableEditCode::CertificationFailed);
        assert!(!refusal.retryable);
    }

    #[test]
    fn frozen_minas_surface_bank_keeps_preregistered_digest() {
        let bytes = include_bytes!("../tests/fixtures/minas-tirith-france-surface-bank-v1.json");
        assert_eq!(
            sha256_bytes(bytes),
            "27fd057835c0fd0992be8920778705d31cac1f3ccaa1f2b61e331266f074052a"
        );
    }
}
