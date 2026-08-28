//! One-load product-only evaluation for a frozen compiled surface manifest.
//!
//! This runner accepts only already-spent development suites. It never reads a
//! sealed holdout and does not calibrate, fit, or alter the manifest.

use std::collections::BTreeSet;
use std::error::Error;
use std::path::{Path, PathBuf};

use larql_inference::compiled_surface_context::{
    evaluate_compiled_surface_contexts, CompiledSurfaceContextManifest, CompiledSurfaceState,
};
use larql_inference::{preflight_terminal_residual_adapter_architecture, InferenceWeights};
use larql_vindex::patch::core::{base_checksum_from_manifest, decode_gate_vector};
use larql_vindex::{
    PatchOp, PatchResidualAdapterApplication, PatchedVindex, QuantFormat, VindexPatch,
};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

type AnyError = Box<dyn Error>;
const EXPECTED_POSITIVES: usize = 7;
const EXPECTED_CONTROLS: usize = 21;
const C0_PROTOCOL_ID: &str = "compiled-surface-context-galadriel-product-v1-c0";
const C1_PROTOCOL_ID: &str = "compiled-surface-context-galadriel-product-v1-c1";
const SHAKEOUT_PROTOCOL_ID: &str = "compiled-surface-context-new-target-shakeout-v2";

#[derive(Clone)]
struct EvaluationContract {
    protocol_id: String,
    target_name: String,
    expected_positives: usize,
    expected_controls: usize,
    evidence_boundary: String,
}

#[derive(Clone, Deserialize)]
struct FactSpec {
    entity: String,
    relation: String,
    target: String,
    expected_tokens: Option<usize>,
    #[serde(default)]
    paraphrases: Vec<String>,
}

#[derive(Clone, Deserialize)]
struct ControlSpec {
    name: String,
    prompt: String,
    #[serde(default)]
    cell: Option<String>,
    #[serde(default)]
    fold: Option<usize>,
}

#[derive(Clone, Deserialize)]
struct SuiteSpec {
    facts: Vec<FactSpec>,
    #[serde(default)]
    controls: Vec<ControlSpec>,
}

#[derive(Clone)]
struct Case {
    class: &'static str,
    name: String,
    prompt: String,
    suite: String,
    cell: Option<String>,
    fold: Option<usize>,
}

fn usage() -> &'static str {
    "usage:\n  evaluate_product_surface_context --preflight-only <vindex> <manifest.json> <suite.json> <suite.json>\n  evaluate_product_surface_context <vindex> <manifest.json> <suite.json> <suite.json> <evidence.json>\n  evaluate_product_surface_context --preflight-effect <vindex> <manifest.json> <effect.vlp> <suite.json> <suite.json>\n  evaluate_product_surface_context --effect <vindex> <manifest.json> <effect.vlp> <suite.json> <suite.json> <evidence.json>\n  evaluate_product_surface_context --shakeout-preflight-effect <vindex> <manifest.json> <effect.vlp> <target> <suite.json>\n  evaluate_product_surface_context --shakeout-effect <vindex> <manifest.json> <effect.vlp> <target> <suite.json> <evidence.json>"
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn canonical_prompt(entity: &str, relation: &str) -> String {
    format!("The {} of {entity} is", relation.replace(['-', '_'], " "))
}

fn load_cases(paths: &[PathBuf], target: &str, target_len: usize) -> Result<Vec<Case>, AnyError> {
    let mut cases = Vec::new();
    let mut seen = BTreeSet::new();
    for path in paths {
        let suite: SuiteSpec = serde_json::from_slice(&std::fs::read(path)?)?;
        if suite.facts.len() != 1 {
            return Err(format!("{} must contain exactly one fact", path.display()).into());
        }
        let fact = &suite.facts[0];
        if fact.target != target || fact.expected_tokens != Some(target_len) {
            return Err(format!("{} fact target contract changed", path.display()).into());
        }
        let suite_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or("suite path has no UTF-8 filename")?
            .to_string();
        for (index, prompt) in std::iter::once(canonical_prompt(&fact.entity, &fact.relation))
            .chain(fact.paraphrases.iter().cloned())
            .enumerate()
        {
            if !seen.insert(prompt.clone()) {
                return Err(format!("duplicate evaluation prompt {prompt:?}").into());
            }
            cases.push(Case {
                class: "positive",
                name: format!("positive-{index}"),
                prompt,
                suite: suite_name.clone(),
                cell: Some("11".into()),
                fold: None,
            });
        }
        for control in suite.controls {
            if !seen.insert(control.prompt.clone()) {
                return Err(format!("duplicate evaluation prompt {:?}", control.prompt).into());
            }
            cases.push(Case {
                class: "control",
                name: control.name,
                prompt: control.prompt,
                suite: suite_name.clone(),
                cell: control.cell,
                fold: control.fold,
            });
        }
    }
    Ok(cases)
}

struct Preflight {
    config: larql_vindex::VindexConfig,
    tokenizer: larql_inference::tokenizers::Tokenizer,
    manifest: CompiledSurfaceContextManifest,
    active_checksum: String,
    cases: Vec<Case>,
    manifest_sha256: String,
    suite_sha256s: Vec<String>,
    effect_patch: Option<VindexPatch>,
    effect_patch_sha256: Option<String>,
    contract: EvaluationContract,
}

fn validate_effect_patch_source(
    patch: &VindexPatch,
    manifest: &CompiledSurfaceContextManifest,
    active_checksum: &str,
) -> Result<(), AnyError> {
    if patch.base_checksum.as_deref() != Some(active_checksum) {
        return Err("C1 effect patch is not bound to the active vindex checksum".into());
    }
    let spec = &manifest.specs[0];
    let contexts: Vec<_> = patch
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
    if contexts.len() != 1 {
        return Err("C1 effect patch must contain exactly one linked adapter".into());
    }
    let (layer, feature, target_ids, adapter) = contexts[0];
    if (layer, feature) != (spec.payload.layer, spec.payload.feature)
        || target_ids != &spec.target_ids
        || adapter.application != PatchResidualAdapterApplication::PrefillLastPositionV1
        || decode_gate_vector(&adapter.delta_vector_b64)? != spec.terminal_residual_adapter.delta
        || adapter.calibration != spec.terminal_residual_adapter.calibration
    {
        return Err("C1 effect patch differs from the compiled manifest".into());
    }
    let has_insert = patch.operations.iter().any(|operation| {
        matches!(
            operation,
            PatchOp::Insert {
                layer: insert_layer,
                feature: insert_feature,
                ..
            } if (*insert_layer, *insert_feature) == (layer, feature)
        )
    });
    if !has_insert {
        return Err("C1 effect patch has no linked FFN Insert".into());
    }
    Ok(())
}

fn preflight(
    vindex: &Path,
    manifest_path: &Path,
    effect_patch_path: Option<&Path>,
    suite_paths: &[PathBuf],
    contract: EvaluationContract,
) -> Result<Preflight, AnyError> {
    let config = larql_vindex::load_vindex_config(vindex)?;
    if config.quant == QuantFormat::None {
        return Err("C0 product evaluation requires native quantised storage".into());
    }
    let arch = larql_vindex::arch_from_vindex_config(&config)
        .ok_or("vindex has no model_config for product preflight")?;
    preflight_terminal_residual_adapter_architecture(arch.as_ref())?;
    let active_checksum = base_checksum_from_manifest(config.checksums.as_ref())
        .ok_or("vindex has no non-empty checksum manifest")?;
    let manifest_bytes = std::fs::read(manifest_path)?;
    let manifest: CompiledSurfaceContextManifest = serde_json::from_slice(&manifest_bytes)?;
    manifest.validate_header()?;
    if manifest.specs.len() != 1 {
        return Err("product v1 requires exactly one compiled surface specification".into());
    }
    for spec in &manifest.specs {
        spec.validate(&active_checksum, config.hidden_size, config.vocab_size)?;
    }
    let (effect_patch, effect_patch_sha256) = if let Some(path) = effect_patch_path {
        let bytes = std::fs::read(path)?;
        let patch: VindexPatch = serde_json::from_slice(&bytes)?;
        validate_effect_patch_source(&patch, &manifest, &active_checksum)?;
        (Some(patch), Some(sha256_bytes(&bytes)))
    } else {
        (None, None)
    };
    let target_ids = &manifest.specs[0].target_ids;
    let tokenizer = larql_vindex::load_vindex_tokenizer(vindex)?;
    let cases = load_cases(suite_paths, &contract.target_name, target_ids.len())?;
    let positives = cases.iter().filter(|case| case.class == "positive").count();
    let controls = cases.iter().filter(|case| case.class == "control").count();
    if positives != contract.expected_positives || controls != contract.expected_controls {
        return Err(format!(
            "evaluation material must contain {}/{} positives/controls, found {positives}/{controls}",
            contract.expected_positives, contract.expected_controls
        )
        .into());
    }

    for case in &cases {
        let ids = larql_inference::encode_prompt(&tokenizer, arch.as_ref(), &case.prompt)?;
        let decisions = evaluate_compiled_surface_contexts(
            &manifest.specs,
            &ids,
            &active_checksum,
            config.hidden_size,
            config.vocab_size,
        )?;
        let expected = if case.class == "positive" {
            CompiledSurfaceState::Fire
        } else {
            CompiledSurfaceState::CompiledReject
        };
        if decisions.len() != 1 || decisions[0].state != expected {
            return Err(format!(
                "frozen matcher decision mismatch for {:?}: expected {expected:?}, got {:?}",
                case.prompt, decisions
            )
            .into());
        }
    }

    let suite_sha256s = suite_paths
        .iter()
        .map(|path| std::fs::read(path).map(|bytes| sha256_bytes(&bytes)))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Preflight {
        config,
        tokenizer,
        manifest,
        active_checksum,
        cases,
        manifest_sha256: sha256_bytes(&manifest_bytes),
        suite_sha256s,
        effect_patch,
        effect_patch_sha256,
        contract,
    })
}

fn preflight_json(preflight: &Preflight) -> Value {
    json!({
        "status": "compatible",
        "claim_label": preflight.manifest.claim_label,
        "evaluation_protocol_id": preflight.contract.protocol_id,
        "manifest_protocol_id": preflight.manifest.protocol_id,
        "weights_loaded": false,
        "base_checksum": preflight.active_checksum,
        "manifest_sha256": preflight.manifest_sha256,
        "suite_sha256s": preflight.suite_sha256s,
        "effect_patch_sha256": preflight.effect_patch_sha256,
        "effect_mode": preflight.effect_patch.is_some(),
        "target": preflight.contract.target_name,
        "positives": preflight.contract.expected_positives,
        "controls": preflight.contract.expected_controls,
        "matcher_contracts": preflight.contract.expected_positives + preflight.contract.expected_controls,
    })
}

fn run_evaluation(
    vindex: &Path,
    mut preflight: Preflight,
    evidence_path: &Path,
) -> Result<Value, AnyError> {
    if evidence_path.exists() {
        return Err("evidence output already exists".into());
    }
    let mut callbacks = larql_vindex::SilentLoadCallbacks;
    let mut weights = InferenceWeights::load(vindex, &preflight.config, &mut callbacks)?;
    let source = match &weights {
        InferenceWeights::Quantised { index, .. } => index.clone(),
        InferenceWeights::Dense(_) => return Err("C0 loaded unexpected dense weights".into()),
    };
    let empty_patch = PatchedVindex::new(source);
    let effect_patch = preflight
        .effect_patch
        .take()
        .map(|patch| {
            let mut overlay = PatchedVindex::new(empty_patch.base.clone());
            overlay
                .try_apply_patch_bound(patch, Some(&preflight.active_checksum))
                .map(|_| overlay)
        })
        .transpose()?;
    let target_ids = preflight.manifest.specs[0].target_ids.clone();
    let mut records = Vec::with_capacity(preflight.cases.len());
    let mut positive_passes = 0usize;
    let mut control_passes = 0usize;

    for case in &preflight.cases {
        let prompt_ids = {
            let model = weights.as_weights();
            larql_inference::encode_prompt(&preflight.tokenizer, model.arch.as_ref(), &case.prompt)?
        };
        let baseline = if case.class == "control" {
            Some(weights.generate_patched(
                &preflight.tokenizer,
                &empty_patch,
                &prompt_ids,
                target_ids.len(),
                &[],
            )?)
        } else {
            None
        };
        let result = if let Some(effect) = effect_patch.as_ref() {
            weights.generate_compiled_surface_with_effect_patch_cached(
                &preflight.tokenizer,
                effect,
                &prompt_ids,
                target_ids.len(),
                &preflight.manifest,
                &preflight.active_checksum,
            )?
        } else {
            weights.generate_compiled_surface_cached(
                &preflight.tokenizer,
                &prompt_ids,
                target_ids.len(),
                &preflight.manifest,
                &preflight.active_checksum,
            )?
        };
        let generated_ids: Vec<_> = result.generated.iter().map(|(_, id)| *id).collect();
        let baseline_ids = baseline.as_ref().map(|value| {
            value
                .generated
                .iter()
                .map(|(_, id)| *id)
                .collect::<Vec<_>>()
        });
        let passed = if case.class == "positive" {
            result.decisions.len() == 1
                && result.decisions[0].state == CompiledSurfaceState::Fire
                && generated_ids == target_ids
                && result.trigger_slot == Some(preflight.manifest.specs[0].payload)
                && result.released_after_span
        } else {
            result.decisions.len() == 1
                && result.decisions[0].state == CompiledSurfaceState::CompiledReject
                && result.trigger_slot.is_none()
                && !result.released_after_span
                && baseline
                    .as_ref()
                    .is_some_and(|value| value.generated == result.generated)
        };
        if passed && case.class == "positive" {
            positive_passes += 1;
        } else if passed {
            control_passes += 1;
        }
        records.push(json!({
            "class": case.class,
            "name": case.name,
            "suite": case.suite,
            "cell": case.cell,
            "fold": case.fold,
            "prompt": case.prompt,
            "prompt_ids": prompt_ids,
            "decision": result.decisions,
            "generated_ids": generated_ids,
            "baseline_ids": baseline_ids,
            "trigger_slot": result.trigger_slot,
            "released_after_span": result.released_after_span,
            "passed": passed,
        }));
    }

    let all_passed = positive_passes == preflight.contract.expected_positives
        && control_passes == preflight.contract.expected_controls;
    let evidence = json!({
        "evaluation_protocol_id": preflight.contract.protocol_id,
        "manifest_protocol_id": preflight.manifest.protocol_id,
        "claim_label": preflight.manifest.claim_label,
        "evidence_boundary": preflight.contract.evidence_boundary,
        "target": preflight.contract.target_name,
        "weights_loaded_once": true,
        "base_checksum": preflight.active_checksum,
        "manifest_sha256": preflight.manifest_sha256,
        "suite_sha256s": preflight.suite_sha256s,
        "effect_patch_sha256": preflight.effect_patch_sha256,
        "effect_mode": effect_patch.is_some(),
        "target_ids": target_ids,
        "positive_passes": positive_passes,
        "positive_total": preflight.contract.expected_positives,
        "control_passes": control_passes,
        "control_total": preflight.contract.expected_controls,
        "all_passed": all_passed,
        "cases": records,
    });
    let bytes = serde_json::to_vec_pretty(&evidence)?;
    std::fs::write(evidence_path, &bytes)?;
    let summary = json!({
        "status": if all_passed { "passed" } else { "failed" },
        "all_passed": all_passed,
        "positive_passes": positive_passes,
        "positive_total": preflight.contract.expected_positives,
        "control_passes": control_passes,
        "control_total": preflight.contract.expected_controls,
        "evidence": evidence_path,
        "evidence_sha256": sha256_bytes(&bytes),
    });
    if !all_passed {
        return Err(format!("product surface evaluation failed: {summary}").into());
    }
    Ok(summary)
}

fn main() -> Result<(), AnyError> {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or_default();
    let shakeout = matches!(mode, "--shakeout-preflight-effect" | "--shakeout-effect");
    let (preflight_only, effect_mode, offset, expected_len) = match mode {
        "--preflight-only" => (true, false, 1, 6),
        "--preflight-effect" => (true, true, 1, 7),
        "--effect" => (false, true, 1, 8),
        "--shakeout-preflight-effect" => (true, true, 1, 7),
        "--shakeout-effect" => (false, true, 1, 8),
        _ => (false, false, 0, 6),
    };
    if args.len() != expected_len {
        return Err(usage().into());
    }
    let vindex = PathBuf::from(&args[1 + offset]);
    let manifest = PathBuf::from(&args[2 + offset]);
    let effect_index = 3 + offset;
    let effect_patch = effect_mode.then(|| PathBuf::from(&args[effect_index]));
    let (suites, evidence_index, contract) = if shakeout {
        let target_index = effect_index + 1;
        let suite_index = target_index + 1;
        (
            vec![PathBuf::from(&args[suite_index])],
            suite_index + 1,
            EvaluationContract {
                protocol_id: SHAKEOUT_PROTOCOL_ID.into(),
                target_name: args[target_index].clone(),
                expected_positives: 1,
                expected_controls: 6,
                evidence_boundary: "single-surface product shakeout for a newly compiled target; no semantic, paraphrase, population-FPR, or prospective certification claim".into(),
            },
        )
    } else {
        let suite_start = effect_index + usize::from(effect_mode);
        (
            vec![
                PathBuf::from(&args[suite_start]),
                PathBuf::from(&args[suite_start + 1]),
            ],
            suite_start + 2,
            EvaluationContract {
                protocol_id: if effect_mode { C1_PROTOCOL_ID } else { C0_PROTOCOL_ID }.into(),
                target_name: "Galadriel".into(),
                expected_positives: EXPECTED_POSITIVES,
                expected_controls: EXPECTED_CONTROLS,
                evidence_boundary: "product-only evaluation on spent Galadriel development material; no semantic relation or prospective claim".into(),
            },
        )
    };
    let prepared = preflight(
        &vindex,
        &manifest,
        effect_patch.as_deref(),
        &suites,
        contract,
    )?;
    let output = if preflight_only {
        preflight_json(&prepared)
    } else {
        let evidence = PathBuf::from(&args[evidence_index]);
        run_evaluation(&vindex, prepared, &evidence)?
    };
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_prompt_normalizes_relation_separators() {
        assert_eq!(
            canonical_prompt("Lothlorien", "high-queen"),
            "The high queen of Lothlorien is"
        );
    }
}
