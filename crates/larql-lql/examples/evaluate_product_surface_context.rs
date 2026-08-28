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
use larql_vindex::patch::core::base_checksum_from_manifest;
use larql_vindex::{PatchedVindex, QuantFormat};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

type AnyError = Box<dyn Error>;
const EXPECTED_POSITIVES: usize = 7;
const EXPECTED_CONTROLS: usize = 21;

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
    "usage:\n  evaluate_product_surface_context --preflight-only <vindex> <manifest.json> <suite.json> <suite.json>\n  evaluate_product_surface_context <vindex> <manifest.json> <suite.json> <suite.json> <evidence.json>"
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
    let positives = cases.iter().filter(|case| case.class == "positive").count();
    let controls = cases.iter().filter(|case| case.class == "control").count();
    if positives != EXPECTED_POSITIVES || controls != EXPECTED_CONTROLS {
        return Err(format!(
            "evaluation material must contain {EXPECTED_POSITIVES} positives and {EXPECTED_CONTROLS} controls, found {positives}/{controls}"
        )
        .into());
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
}

fn preflight(
    vindex: &Path,
    manifest_path: &Path,
    suite_paths: &[PathBuf],
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
    let target_ids = &manifest.specs[0].target_ids;
    let tokenizer = larql_vindex::load_vindex_tokenizer(vindex)?;
    let cases = load_cases(suite_paths, "Galadriel", target_ids.len())?;

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
    })
}

fn preflight_json(preflight: &Preflight) -> Value {
    json!({
        "status": "compatible",
        "claim_label": preflight.manifest.claim_label,
        "protocol_id": preflight.manifest.protocol_id,
        "weights_loaded": false,
        "base_checksum": preflight.active_checksum,
        "manifest_sha256": preflight.manifest_sha256,
        "suite_sha256s": preflight.suite_sha256s,
        "positives": EXPECTED_POSITIVES,
        "controls": EXPECTED_CONTROLS,
        "matcher_contracts": EXPECTED_POSITIVES + EXPECTED_CONTROLS,
    })
}

fn run_evaluation(
    vindex: &Path,
    preflight: Preflight,
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
        let result = weights.generate_compiled_surface_cached(
            &preflight.tokenizer,
            &prompt_ids,
            target_ids.len(),
            &preflight.manifest,
            &preflight.active_checksum,
        )?;
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

    let all_passed = positive_passes == EXPECTED_POSITIVES && control_passes == EXPECTED_CONTROLS;
    let evidence = json!({
        "protocol_id": preflight.manifest.protocol_id,
        "claim_label": preflight.manifest.claim_label,
        "evidence_boundary": "product-only evaluation on spent Galadriel development material; no semantic relation or prospective claim",
        "weights_loaded_once": true,
        "base_checksum": preflight.active_checksum,
        "manifest_sha256": preflight.manifest_sha256,
        "suite_sha256s": preflight.suite_sha256s,
        "target_ids": target_ids,
        "positive_passes": positive_passes,
        "positive_total": EXPECTED_POSITIVES,
        "control_passes": control_passes,
        "control_total": EXPECTED_CONTROLS,
        "all_passed": all_passed,
        "cases": records,
    });
    let bytes = serde_json::to_vec_pretty(&evidence)?;
    std::fs::write(evidence_path, &bytes)?;
    let summary = json!({
        "status": if all_passed { "passed" } else { "failed" },
        "all_passed": all_passed,
        "positive_passes": positive_passes,
        "positive_total": EXPECTED_POSITIVES,
        "control_passes": control_passes,
        "control_total": EXPECTED_CONTROLS,
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
    let preflight_only = args.get(1).is_some_and(|arg| arg == "--preflight-only");
    if (preflight_only && args.len() != 6) || (!preflight_only && args.len() != 6) {
        return Err(usage().into());
    }
    let offset = usize::from(preflight_only);
    let vindex = PathBuf::from(&args[1 + offset]);
    let manifest = PathBuf::from(&args[2 + offset]);
    let suites = [
        PathBuf::from(&args[3 + offset]),
        PathBuf::from(&args[4 + offset]),
    ];
    let prepared = preflight(&vindex, &manifest, &suites)?;
    let output = if preflight_only {
        preflight_json(&prepared)
    } else {
        let evidence = PathBuf::from(&args[5]);
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
