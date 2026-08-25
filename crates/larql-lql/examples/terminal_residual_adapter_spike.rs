//! One-load terminal residual adapter experiments with sealed evaluation gates.
//!
//! Evaluation inputs are inaccessible until the derived patch has been saved,
//! reparsed, checksum-checked, and hashed. The supported Galadriel and Alder-v2
//! modes are bound to their frozen protocol documents rather than exposed as a
//! general-purpose tuning surface.

use std::error::Error;
use std::io::Write;
use std::path::{Path, PathBuf};

use larql_inference::active_span::ActiveSpanSpec;
use larql_inference::context_detector::{ContextDecision, ContextDetectorSpec};
use larql_inference::{
    compile_terminal_residual_adapter, hidden_to_raw_logits,
    preflight_terminal_residual_adapter_architecture, InferenceWeights, KnnRouteMode,
};
use larql_lql::{parse, Session};
use larql_vindex::patch::core::{
    base_checksum_from_manifest, decode_gate_vector, encode_gate_vector,
};
use larql_vindex::{PatchOp, PatchResidualAdapter, PatchResidualAdapterApplication, VindexPatch};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

type AnyError = Box<dyn Error>;
const PROTOCOL_ID: &str = "conditional-terminal-residual-spike-v2";
const TARGET_IDS: [u32; 3] = [8833, 553, 21457];
const PASSIVE_PROMPT: &str = "Lothlorien is governed by";
const PASSIVE_EXPECTED_MARGIN: f32 = -0.133_890;
const PASSIVE_TOLERANCE: f32 = 0.02;
const EVALUATION_MARGIN: f32 = 0.5;
const ALDER_V2_PROTOCOL_ID: &str = "alder-v2-terminal-residual-development";
const ALDER_V2_HOLDOUT_PROTOCOL_ID: &str = "alder-v2-final-holdout";
const ALDER_V2_TARGET_IDS: [u32; 3] = [45158, 37937, 217262];

struct RunProfile {
    protocol_id: &'static str,
    target_name: &'static str,
    target_ids: &'static [u32],
    synthesis_positives: usize,
    evaluation_positives: usize,
    evaluation_controls: usize,
    galadriel_reproduction_gate: bool,
    description: &'static str,
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
struct PromptCase {
    class: &'static str,
    name: String,
    prompt: String,
    cell: Option<String>,
    fold: Option<usize>,
}

struct ForwardCapture {
    hidden: Vec<f32>,
    logits: Vec<f32>,
    decisions: Vec<ContextDecision>,
    adapter_applied: bool,
}

#[derive(Serialize)]
struct PromptEvidence {
    class: &'static str,
    name: String,
    prompt: String,
    cell: Option<String>,
    fold: Option<usize>,
    reference_top_id: u32,
    reference_runner_id: u32,
    reference_target_margin: f32,
    adapter_top_id: u32,
    adapter_runner_id: u32,
    adapter_target_margin: f32,
    reference_decisions: Vec<ContextDecision>,
    adapter_decisions: Vec<ContextDecision>,
    detector_decisions_bit_identical: bool,
    adapter_applied: bool,
    delta_fraction: f32,
    logits_bit_identical: bool,
    passed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FreezePhase {
    Compiling,
    PatchHashed,
}

fn usage() -> &'static str {
    "usage:\n  terminal_residual_adapter_spike --preflight-only <vindex>\n  terminal_residual_adapter_spike <vindex> <k1.vlp> <validation2.json> <validation3.json> <derived.vlp> <evidence.json>\n  terminal_residual_adapter_spike --alder-v2-dev <vindex> <contextual.vlp> <development.json> <development.json> <derived.vlp> <evidence.json>\n  terminal_residual_adapter_spike --alder-v2-holdout <vindex> <contextual.vlp> <derived.vlp> <holdout.json> <evidence.json>"
}

fn preflight_only(vindex: &Path) -> Result<Value, AnyError> {
    let config = larql_vindex::load_vindex_config(vindex)?;
    let arch = larql_vindex::arch_from_vindex_config(&config)
        .ok_or("vindex has no model_config for terminal-adapter preflight")?;
    preflight_terminal_residual_adapter_architecture(arch.as_ref())?;
    Ok(json!({
        "status": "compatible",
        "vindex": vindex,
        "model_family": arch.family().to_string(),
        "hidden_size": arch.config().hidden_size,
        "vocab_size": arch.config().vocab_size,
        "norm_type": format!("{:?}", arch.norm_type()),
        "norm_epsilon": arch.norm_eps(),
        "logits_scaling": arch.logits_scaling(),
        "final_logit_softcap": arch.final_logit_softcapping(),
        "weights_loaded": false,
    }))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_file(path: &Path) -> Result<String, AnyError> {
    Ok(sha256_bytes(&std::fs::read(path)?))
}

fn atomic_write_json(path: &Path, value: &Value) -> Result<(), AnyError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;
    let temp = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("evidence"),
        std::process::id()
    ));
    let mut file = std::fs::File::create(&temp)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    std::fs::rename(temp, path)?;
    Ok(())
}

fn canonical_prompt(fact: &FactSpec) -> String {
    let relation = fact.relation.replace(['-', '_'], " ");
    format!("The {relation} of {} is", fact.entity)
}

fn positive_prompts(
    suite: &SuiteSpec,
    target_name: &str,
    target_len: usize,
) -> Result<Vec<PromptCase>, AnyError> {
    let [fact] = suite.facts.as_slice() else {
        return Err(format!(
            "suite must contain exactly one fact, found {}",
            suite.facts.len()
        )
        .into());
    };
    if fact.target != target_name || fact.expected_tokens != Some(target_len) {
        return Err(
            format!("suite must target {target_name:?} with exactly {target_len} IDs").into(),
        );
    }
    Ok(std::iter::once(canonical_prompt(fact))
        .chain(fact.paraphrases.iter().cloned())
        .enumerate()
        .map(|(index, prompt)| PromptCase {
            class: "positive",
            name: if index == 0 {
                "canonical".into()
            } else {
                format!("paraphrase-{index}")
            },
            prompt,
            cell: Some("11".into()),
            fold: Some(index),
        })
        .collect())
}

fn evaluation_cases(
    suite: &SuiteSpec,
    target_name: &str,
    target_len: usize,
) -> Result<Vec<PromptCase>, AnyError> {
    let mut cases = positive_prompts(suite, target_name, target_len)?;
    cases.extend(suite.controls.iter().map(|control| PromptCase {
        class: "control",
        name: control.name.clone(),
        prompt: control.prompt.clone(),
        cell: control.cell.clone(),
        fold: control.fold,
    }));
    Ok(cases)
}

fn read_evaluation_suite(path: &Path, phase: FreezePhase) -> Result<SuiteSpec, AnyError> {
    if phase != FreezePhase::PatchHashed {
        return Err("evaluation suite is sealed until the derived patch is hashed".into());
    }
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

fn open_session(vindex: &Path, patch: &Path) -> Result<Session, AnyError> {
    let mut session = Session::new();
    for sql in [
        format!("USE {};", serde_json::to_string(vindex)?),
        format!("APPLY PATCH {};", serde_json::to_string(patch)?),
    ] {
        let statement = parse(&sql)?;
        for line in session.execute(&statement)? {
            eprintln!("{line}");
        }
    }
    Ok(session)
}

fn apply_patch(session: &mut Session, patch: &Path) -> Result<(), AnyError> {
    let statement = parse(&format!("APPLY PATCH {};", serde_json::to_string(patch)?))?;
    for line in session.execute(&statement)? {
        eprintln!("{line}");
    }
    Ok(())
}

fn prompt_ids(
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    weights: &InferenceWeights,
    prompt: &str,
) -> Result<Vec<u32>, AnyError> {
    Ok(larql_inference::encode_prompt(
        tokenizer,
        weights.as_weights().arch.as_ref(),
        prompt,
    )?)
}

fn forward_capture(
    session: &mut Session,
    weights: &mut InferenceWeights,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    prompt: &str,
) -> Result<ForwardCapture, AnyError> {
    let ids = prompt_ids(tokenizer, weights, prompt)?;
    let patched = session
        .patched_overlay_mut()
        .ok_or("session has no patched vindex")?;
    let detectors: Vec<ContextDetectorSpec> = patched
        .context_detectors()
        .into_iter()
        .map(Into::into)
        .collect();
    let result = weights.infer_patched_contextual(
        tokenizer,
        patched,
        Some(&patched.knn_store),
        &ids,
        2,
        &KnnRouteMode::Legacy,
        &detectors,
    );
    let hidden = result
        .terminal_hidden
        .ok_or("contextual forward did not expose terminal hidden")?;
    let hidden_2d = Array2::from_shape_vec((1, hidden.len()), hidden.clone())?;
    let logits = hidden_to_raw_logits(weights.as_weights(), &hidden_2d);
    Ok(ForwardCapture {
        hidden,
        logits,
        decisions: result.context_decisions,
        adapter_applied: result.terminal_adapter_applied,
    })
}

fn target_margin(logits: &[f32], target_id: u32) -> Result<(f32, u32, u32), AnyError> {
    let target = target_id as usize;
    if target >= logits.len() || logits.len() < 2 || logits.iter().any(|value| !value.is_finite()) {
        return Err("invalid logits for target-margin calculation".into());
    }
    let (top, _) = logits
        .iter()
        .enumerate()
        .max_by(|(left_id, left), (right_id, right)| {
            left.total_cmp(right).then_with(|| right_id.cmp(left_id))
        })
        .ok_or("empty logits")?;
    let (runner, runner_logit) = logits
        .iter()
        .enumerate()
        .filter(|(id, _)| *id != target)
        .max_by(|(left_id, left), (right_id, right)| {
            left.total_cmp(right).then_with(|| right_id.cmp(left_id))
        })
        .ok_or("target is the only vocabulary ID")?;
    Ok((logits[target] - *runner_logit, runner as u32, top as u32))
}

fn l2_norm(values: &[f32]) -> f32 {
    values.iter().map(|value| value * value).sum::<f32>().sqrt()
}

fn source_patch_parts(
    patch: &VindexPatch,
    target_ids: &[u32],
) -> Result<((usize, usize), Vec<f32>), AnyError> {
    let inserts: Vec<_> = patch
        .operations
        .iter()
        .filter_map(|operation| match operation {
            PatchOp::Insert {
                layer,
                feature,
                down_vector_b64: Some(down),
                ..
            } => Some(((*layer, *feature), down)),
            _ => None,
        })
        .collect();
    let contexts: Vec<_> = patch
        .operations
        .iter()
        .filter_map(|operation| match operation {
            PatchOp::Context {
                layer,
                feature,
                target_ids,
                residual_adapter,
                ..
            } => Some(((*layer, *feature), target_ids, residual_adapter)),
            _ => None,
        })
        .collect();
    let [((layer, feature), down_b64)] = inserts.as_slice() else {
        return Err("K1 patch must contain exactly one Insert with a persisted down vector".into());
    };
    let [(context_slot, ids, adapter)] = contexts.as_slice() else {
        return Err("K1 patch must contain exactly one Context".into());
    };
    if (*layer, *feature) != *context_slot || ids.as_slice() != target_ids || adapter.is_some() {
        return Err("K1 Insert/Context binding, target IDs, or adapter state changed".into());
    }
    Ok(((*layer, *feature), decode_gate_vector(down_b64)?))
}

fn derive_patch(
    source: &VindexPatch,
    slot: (usize, usize),
    compiled: &larql_inference::TerminalResidualAdapter,
    profile: &RunProfile,
) -> Result<VindexPatch, AnyError> {
    if compiled.alpha <= 0.0 || compiled.delta.iter().all(|value| *value == 0.0) {
        return Err("compiler produced a zero terminal adapter".into());
    }
    let mut derived = source.clone();
    let calibration = serde_json::to_string(&json!({
        "protocol": profile.protocol_id,
        "application": "post_l41_prefill_last_position_only",
        "target_ids": profile.target_ids,
        "alpha": compiled.alpha,
        "alpha_max": compiled.alpha_max,
        "orthogonal_ratio": compiled.orthogonal_ratio,
        "direction_vector_b64": encode_gate_vector(compiled.direction.as_slice().ok_or("direction is not contiguous")?),
        "target_margin": compiled.options.target_margin,
        "max_residual_fraction": compiled.options.max_residual_fraction,
        "monotonicity_grid_points": compiled.options.monotonicity_grid_points,
    }))?;
    let mut changed = 0;
    for operation in &mut derived.operations {
        if let PatchOp::Context {
            layer,
            feature,
            residual_adapter,
            ..
        } = operation
        {
            if (*layer, *feature) == slot {
                *residual_adapter = Some(PatchResidualAdapter {
                    application: PatchResidualAdapterApplication::PrefillLastPositionV1,
                    delta_vector_b64: encode_gate_vector(
                        compiled.delta.as_slice().ok_or("delta is not contiguous")?,
                    ),
                    calibration: Some(calibration.clone()),
                });
                changed += 1;
            }
        }
    }
    if changed != 1 {
        return Err(format!("expected one linked Context operation, changed {changed}").into());
    }
    derived.description = Some(profile.description.into());
    for tag in ["research-spike", "terminal-residual", "prefill-only"] {
        if !derived.tags.iter().any(|existing| existing == tag) {
            derived.tags.push(tag.into());
        }
    }
    Ok(derived)
}

fn validate_frozen_patch(
    path: &Path,
    expected_checksum: Option<&str>,
    hidden_size: usize,
) -> Result<VindexPatch, AnyError> {
    let reparsed = VindexPatch::load(path)?;
    if reparsed.base_checksum.as_deref() != expected_checksum {
        return Err("derived patch base checksum changed".into());
    }
    let adapters: Vec<_> = reparsed
        .operations
        .iter()
        .filter_map(|operation| match operation {
            PatchOp::Context {
                residual_adapter: Some(adapter),
                ..
            } => Some(adapter),
            _ => None,
        })
        .collect();
    let [adapter] = adapters.as_slice() else {
        return Err("derived patch must contain exactly one residual adapter".into());
    };
    let delta = decode_gate_vector(&adapter.delta_vector_b64)?;
    if delta.len() != hidden_size
        || delta.iter().any(|value| !value.is_finite())
        || !delta.iter().any(|value| *value != 0.0)
    {
        return Err("derived patch adapter failed frozen vector validation".into());
    }
    Ok(reparsed)
}

fn detectors_without_adapter(session: &Session) -> Result<Vec<ContextDetectorSpec>, AnyError> {
    let mut detectors: Vec<ContextDetectorSpec> = session
        .patched_overlay()
        .ok_or("session has no patched vindex")?
        .context_detectors()
        .into_iter()
        .map(Into::into)
        .collect();
    for detector in &mut detectors {
        detector.terminal_residual_adapter = None;
    }
    Ok(detectors)
}

fn decisions_bit_identical(left: &[ContextDecision], right: &[ContextDecision]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(a, b)| {
            a.payload == b.payload
                && a.detector_layer == b.detector_layer
                && a.entity_margin.to_bits() == b.entity_margin.to_bits()
                && a.relation_margin.to_bits() == b.relation_margin.to_bits()
                && a.allowed == b.allowed
        })
}

fn logits_bit_identical(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(a, b)| a.to_bits() == b.to_bits())
}

fn span_specs(patch: &VindexPatch) -> Vec<ActiveSpanSpec> {
    patch
        .operations
        .iter()
        .filter_map(|operation| match operation {
            PatchOp::Context {
                layer,
                feature,
                target_ids,
                ..
            } if target_ids.len() > 1 => Some(ActiveSpanSpec {
                slot: larql_inference::active_span::PatchSlotId {
                    layer: *layer,
                    feature: *feature,
                },
                target_ids: target_ids.clone(),
            }),
            _ => None,
        })
        .collect()
}

fn run(args: &[String]) -> Result<Value, AnyError> {
    let alder_v2 = args.get(1).map(String::as_str) == Some("--alder-v2-dev");
    let expected_len = if alder_v2 { 8 } else { 7 };
    if args.len() != expected_len {
        return Err(usage().into());
    }
    let offset = usize::from(alder_v2);
    let vindex = PathBuf::from(&args[1 + offset]);
    let source_path = PathBuf::from(&args[2 + offset]);
    let synthesis_path = PathBuf::from(&args[3 + offset]);
    let evaluation_path = PathBuf::from(&args[4 + offset]);
    let derived_path = PathBuf::from(&args[5 + offset]);
    let profile = if alder_v2 {
        RunProfile {
            protocol_id: ALDER_V2_PROTOCOL_ID,
            target_name: "Ada Lovelace",
            target_ids: &ALDER_V2_TARGET_IDS,
            synthesis_positives: 12,
            evaluation_positives: 12,
            evaluation_controls: 36,
            galadriel_reproduction_gate: false,
            description: "Alder-v2 detector-gated terminal residual adapter; development gate",
        }
    } else {
        RunProfile {
            protocol_id: PROTOCOL_ID,
            target_name: "Galadriel",
            target_ids: &TARGET_IDS,
            synthesis_positives: 3,
            evaluation_positives: 4,
            evaluation_controls: 12,
            galadriel_reproduction_gate: true,
            description:
                "Development-only detector-gated terminal residual adapter; frozen Galadriel spike",
        }
    };

    let source = VindexPatch::load(&source_path)?;
    let (slot, k1_down) = source_patch_parts(&source, profile.target_ids)?;
    let config = larql_vindex::load_vindex_config(&vindex)?;
    let configured_arch = larql_vindex::arch_from_vindex_config(&config)
        .ok_or("vindex has no model_config for terminal-adapter preflight")?;
    preflight_terminal_residual_adapter_architecture(configured_arch.as_ref())?;
    let active_checksum = base_checksum_from_manifest(config.checksums.as_ref());
    if source.base_checksum != active_checksum {
        return Err("K1 patch is not bound to the active vindex checksum".into());
    }
    let tokenizer = larql_vindex::load_vindex_tokenizer(&vindex)?;
    let synthesis: SuiteSpec = serde_json::from_slice(&std::fs::read(&synthesis_path)?)?;
    let synthesis_cases =
        positive_prompts(&synthesis, profile.target_name, profile.target_ids.len())?;
    if synthesis_cases.len() != profile.synthesis_positives {
        return Err(format!(
            "synthesis must contain exactly {} positives",
            profile.synthesis_positives
        )
        .into());
    }
    if profile.galadriel_reproduction_gate && synthesis_cases[2].prompt != PASSIVE_PROMPT {
        return Err("validation2 synthesis prompt set or order changed".into());
    }
    let mut phase = FreezePhase::Compiling;

    let mut session = open_session(&vindex, &source_path)?;
    let frozen_detector_before = detectors_without_adapter(&session)?;
    let mut callbacks = larql_vindex::SilentLoadCallbacks;
    let mut weights = InferenceWeights::load(&vindex, &config, &mut callbacks)?;

    let mut synthesis_captures = Vec::new();
    for case in &synthesis_cases {
        let capture = forward_capture(&mut session, &mut weights, &tokenizer, &case.prompt)?;
        if capture.adapter_applied || capture.decisions.len() != 1 || !capture.decisions[0].allowed
        {
            return Err(format!("synthesis detector gate failed for {:?}", case.prompt).into());
        }
        synthesis_captures.push(capture);
    }
    let residuals: Vec<Array1<f32>> = synthesis_captures
        .iter()
        .map(|capture| Array1::from(capture.hidden.clone()))
        .collect();
    let compiled = compile_terminal_residual_adapter(
        weights.as_weights(),
        &residuals,
        profile.target_ids[0],
        Array1::from(k1_down).view(),
    )?;
    if profile.galadriel_reproduction_gate
        && (compiled.baseline_margins[0] <= 0.0 || compiled.baseline_margins[1] <= 0.0)
    {
        return Err(
            "K1 no longer selects target ID 8833 on the first two synthesis prompts".into(),
        );
    }
    if profile.galadriel_reproduction_gate
        && (compiled.baseline_margins[2] - PASSIVE_EXPECTED_MARGIN).abs() > PASSIVE_TOLERANCE
    {
        return Err(format!(
            "passive baseline margin {:.8} does not reproduce {:.6} +/- {:.2}",
            compiled.baseline_margins[2], PASSIVE_EXPECTED_MARGIN, PASSIVE_TOLERANCE
        )
        .into());
    }

    let derived = derive_patch(&source, slot, &compiled, &profile)?;
    derived.save(&derived_path)?;
    let reparsed = validate_frozen_patch(
        &derived_path,
        active_checksum.as_deref(),
        weights.as_weights().hidden_size,
    )?;
    let derived_sha256 = sha256_file(&derived_path)?;
    eprintln!("FROZEN_ADAPTER_SHA256={derived_sha256}");
    if phase != FreezePhase::Compiling {
        return Err("adapter freeze phase advanced more than once".into());
    }
    phase = FreezePhase::PatchHashed;

    let evaluation = read_evaluation_suite(&evaluation_path, phase)?;
    let cases = evaluation_cases(&evaluation, profile.target_name, profile.target_ids.len())?;
    if cases.iter().filter(|case| case.class == "positive").count() != profile.evaluation_positives
        || cases.iter().filter(|case| case.class == "control").count()
            != profile.evaluation_controls
    {
        return Err(format!(
            "evaluation must contain {} positives and {} controls",
            profile.evaluation_positives, profile.evaluation_controls
        )
        .into());
    }

    let mut references = Vec::with_capacity(cases.len());
    for case in &cases {
        references.push(forward_capture(
            &mut session,
            &mut weights,
            &tokenizer,
            &case.prompt,
        )?);
    }
    apply_patch(&mut session, &derived_path)?;
    if detectors_without_adapter(&session)? != frozen_detector_before {
        return Err("derived patch changed the frozen detector specification".into());
    }

    let delta_norm = l2_norm(compiled.delta.as_slice().ok_or("delta is not contiguous")?);
    let mut prompt_evidence = Vec::with_capacity(cases.len());
    let mut activation_passed = true;
    for (case, reference) in cases.iter().zip(&references) {
        let adapted = forward_capture(&mut session, &mut weights, &tokenizer, &case.prompt)?;
        let (reference_margin, reference_runner, reference_top) =
            target_margin(&reference.logits, profile.target_ids[0])?;
        let (adapter_margin, adapter_runner, adapter_top) =
            target_margin(&adapted.logits, profile.target_ids[0])?;
        let decisions_equal = decisions_bit_identical(&reference.decisions, &adapted.decisions);
        let logits_equal = logits_bit_identical(&reference.logits, &adapted.logits);
        let delta_fraction = delta_norm / l2_norm(&reference.hidden);
        let passed = if case.class == "positive" {
            adapter_top == profile.target_ids[0]
                && adapter_margin >= EVALUATION_MARGIN
                && adapted.decisions.len() == 1
                && adapted.decisions[0].allowed
                && adapted.adapter_applied
                && delta_fraction <= compiled.options.max_residual_fraction
                && decisions_equal
        } else {
            !adapted.adapter_applied
                && !adapted.decisions.is_empty()
                && adapted.decisions.iter().all(|decision| !decision.allowed)
                && logits_equal
                && decisions_equal
        };
        activation_passed &= passed;
        prompt_evidence.push(PromptEvidence {
            class: case.class,
            name: case.name.clone(),
            prompt: case.prompt.clone(),
            cell: case.cell.clone(),
            fold: case.fold,
            reference_top_id: reference_top,
            reference_runner_id: reference_runner,
            reference_target_margin: reference_margin,
            adapter_top_id: adapter_top,
            adapter_runner_id: adapter_runner,
            adapter_target_margin: adapter_margin,
            reference_decisions: reference.decisions.clone(),
            adapter_decisions: adapted.decisions,
            detector_decisions_bit_identical: decisions_equal,
            adapter_applied: adapted.adapter_applied,
            delta_fraction,
            logits_bit_identical: logits_equal,
            passed,
        });
    }

    let mut cached = Vec::new();
    let mut cached_passed = false;
    if activation_passed {
        let specs = span_specs(&reparsed);
        if specs.len() != 1 || specs[0].target_ids != profile.target_ids {
            return Err("derived patch changed the ActiveSpan specification".into());
        }
        cached_passed = true;
        for case in &cases {
            let ids = prompt_ids(&tokenizer, &weights, &case.prompt)?;
            let patched = session
                .patched_overlay()
                .ok_or("session has no patched vindex")?;
            let generated = weights.generate_patched(
                &tokenizer,
                patched,
                &ids,
                if case.class == "positive" { 4 } else { 1 },
                &specs,
            )?;
            let generated_ids: Vec<_> = generated.generated.iter().map(|(_, id)| *id).collect();
            let detector_allowed = generated
                .context_decisions
                .iter()
                .any(|decision| decision.allowed);
            let passed = if case.class == "positive" {
                detector_allowed
                    && generated_ids.get(..profile.target_ids.len()) == Some(profile.target_ids)
                    && generated.trigger_slot == Some(specs[0].slot)
                    && generated.released_after_span
            } else {
                !generated.context_decisions.is_empty()
                    && generated
                        .context_decisions
                        .iter()
                        .all(|decision| !decision.allowed)
                    && generated_ids.first() != Some(&profile.target_ids[0])
                    && generated.trigger_slot.is_none()
            };
            cached_passed &= passed;
            cached.push(json!({
                "class": case.class,
                "name": case.name,
                "prompt": case.prompt,
                "generated_ids": generated_ids,
                "context_decisions": generated.context_decisions,
                "trigger_slot": generated.trigger_slot,
                "released_after_span": generated.released_after_span,
                "passed": passed,
            }));
        }
    }

    let status = if activation_passed && cached_passed {
        "passed"
    } else {
        "failed"
    };
    Ok(json!({
        "protocol": profile.protocol_id,
        "status": status,
        "phase_reached": if activation_passed { "cached_generation" } else { "activation" },
        "inputs": {
            "vindex": vindex,
            "source_patch": source_path,
            "source_patch_sha256": sha256_file(&source_path)?,
            "synthesis_suite": synthesis_path,
            "synthesis_suite_sha256": sha256_file(&synthesis_path)?,
            "evaluation_suite": evaluation_path,
            "evaluation_suite_sha256": sha256_file(&evaluation_path)?,
            "active_base_checksum": active_checksum,
        },
        "compiled_adapter": {
            "patch": derived_path,
            "sha256": derived_sha256,
            "slot": {"layer": slot.0, "feature": slot.1},
            "target_ids": profile.target_ids,
            "worst_prompt_index": compiled.worst_prompt_index,
            "baseline_runner_ids": compiled.baseline_runner_ids,
            "baseline_margins": compiled.baseline_margins,
            "compiled_runner_ids": compiled.compiled_runner_ids,
            "compiled_margins": compiled.compiled_margins,
            "orthogonal_ratio": compiled.orthogonal_ratio,
            "alpha": compiled.alpha,
            "alpha_max": compiled.alpha_max,
            "delta_norm": delta_norm,
            "options": {
                "target_margin": compiled.options.target_margin,
                "min_orthogonal_ratio": compiled.options.min_orthogonal_ratio,
                "max_residual_fraction": compiled.options.max_residual_fraction,
                "monotonicity_grid_points": compiled.options.monotonicity_grid_points,
                "monotonicity_tolerance": compiled.options.monotonicity_tolerance,
                "bisection_relative_tolerance": compiled.options.bisection_relative_tolerance,
            }
        },
        "activation": {
            "passed": activation_passed,
            "prompts": prompt_evidence,
        },
        "cached_generation": {
            "run": activation_passed,
            "passed": cached_passed,
            "prompts": cached,
        },
    }))
}

fn run_alder_v2_holdout(args: &[String]) -> Result<Value, AnyError> {
    if args.len() != 7 {
        return Err(usage().into());
    }
    let vindex = PathBuf::from(&args[2]);
    let source_path = PathBuf::from(&args[3]);
    let derived_path = PathBuf::from(&args[4]);
    let holdout_path = PathBuf::from(&args[5]);
    let source = VindexPatch::load(&source_path)?;
    let (slot, _) = source_patch_parts(&source, &ALDER_V2_TARGET_IDS)?;
    let config = larql_vindex::load_vindex_config(&vindex)?;
    let configured_arch = larql_vindex::arch_from_vindex_config(&config)
        .ok_or("vindex has no model_config for terminal-adapter preflight")?;
    preflight_terminal_residual_adapter_architecture(configured_arch.as_ref())?;
    let active_checksum = base_checksum_from_manifest(config.checksums.as_ref());
    if source.base_checksum != active_checksum {
        return Err("Alder-v2 source patch is not bound to the active vindex checksum".into());
    }
    let reparsed = validate_frozen_patch(
        &derived_path,
        active_checksum.as_deref(),
        configured_arch.config().hidden_size,
    )?;
    let derived_sha256 = sha256_file(&derived_path)?;
    eprintln!("FROZEN_ADAPTER_SHA256={derived_sha256}");

    // This is the first filesystem access to the sealed holdout.  The derived
    // adapter has already been reparsed, checksum-checked, and hashed above.
    let holdout = read_evaluation_suite(&holdout_path, FreezePhase::PatchHashed)?;
    let cases = evaluation_cases(&holdout, "Ada Lovelace", ALDER_V2_TARGET_IDS.len())?;
    if cases.iter().filter(|case| case.class == "positive").count() != 4
        || cases.iter().filter(|case| case.class == "control").count() != 12
    {
        return Err("Alder-v2 holdout must contain four positives and twelve controls".into());
    }

    let tokenizer = larql_vindex::load_vindex_tokenizer(&vindex)?;
    let mut session = open_session(&vindex, &source_path)?;
    let frozen_detector_before = detectors_without_adapter(&session)?;
    let mut callbacks = larql_vindex::SilentLoadCallbacks;
    let mut weights = InferenceWeights::load(&vindex, &config, &mut callbacks)?;
    let mut references = Vec::with_capacity(cases.len());
    for case in &cases {
        references.push(forward_capture(
            &mut session,
            &mut weights,
            &tokenizer,
            &case.prompt,
        )?);
    }
    apply_patch(&mut session, &derived_path)?;
    if detectors_without_adapter(&session)? != frozen_detector_before {
        return Err("derived patch changed the frozen Alder-v2 detector".into());
    }
    let adapters: Vec<_> = reparsed
        .operations
        .iter()
        .filter_map(|operation| match operation {
            PatchOp::Context {
                residual_adapter: Some(adapter),
                ..
            } => Some(adapter),
            _ => None,
        })
        .collect();
    let [adapter] = adapters.as_slice() else {
        return Err("derived Alder-v2 patch must contain exactly one adapter".into());
    };
    let delta = decode_gate_vector(&adapter.delta_vector_b64)?;
    let delta_norm = l2_norm(&delta);
    let mut prompt_evidence = Vec::with_capacity(cases.len());
    let mut activation_passed = true;
    for (case, reference) in cases.iter().zip(&references) {
        let adapted = forward_capture(&mut session, &mut weights, &tokenizer, &case.prompt)?;
        let (reference_margin, reference_runner, reference_top) =
            target_margin(&reference.logits, ALDER_V2_TARGET_IDS[0])?;
        let (adapter_margin, adapter_runner, adapter_top) =
            target_margin(&adapted.logits, ALDER_V2_TARGET_IDS[0])?;
        let decisions_equal = decisions_bit_identical(&reference.decisions, &adapted.decisions);
        let logits_equal = logits_bit_identical(&reference.logits, &adapted.logits);
        let delta_fraction = delta_norm / l2_norm(&reference.hidden);
        let passed = if case.class == "positive" {
            adapter_top == ALDER_V2_TARGET_IDS[0]
                && adapter_margin >= EVALUATION_MARGIN
                && adapted.decisions.len() == 1
                && adapted.decisions[0].allowed
                && adapted.adapter_applied
                && delta_fraction <= 0.05
                && decisions_equal
        } else {
            !adapted.adapter_applied
                && !adapted.decisions.is_empty()
                && adapted.decisions.iter().all(|decision| !decision.allowed)
                && logits_equal
                && decisions_equal
        };
        activation_passed &= passed;
        prompt_evidence.push(PromptEvidence {
            class: case.class,
            name: case.name.clone(),
            prompt: case.prompt.clone(),
            cell: case.cell.clone(),
            fold: case.fold,
            reference_top_id: reference_top,
            reference_runner_id: reference_runner,
            reference_target_margin: reference_margin,
            adapter_top_id: adapter_top,
            adapter_runner_id: adapter_runner,
            adapter_target_margin: adapter_margin,
            reference_decisions: reference.decisions.clone(),
            adapter_decisions: adapted.decisions,
            detector_decisions_bit_identical: decisions_equal,
            adapter_applied: adapted.adapter_applied,
            delta_fraction,
            logits_bit_identical: logits_equal,
            passed,
        });
    }

    let mut cached = Vec::new();
    let mut cached_passed = false;
    if activation_passed {
        let specs = span_specs(&reparsed);
        if specs.len() != 1 || specs[0].target_ids != ALDER_V2_TARGET_IDS {
            return Err("derived patch changed the Alder-v2 ActiveSpan specification".into());
        }
        cached_passed = true;
        for case in &cases {
            let ids = prompt_ids(&tokenizer, &weights, &case.prompt)?;
            let patched = session
                .patched_overlay()
                .ok_or("session has no patched vindex")?;
            let generated = weights.generate_patched(
                &tokenizer,
                patched,
                &ids,
                if case.class == "positive" { 4 } else { 1 },
                &specs,
            )?;
            let generated_ids: Vec<_> = generated.generated.iter().map(|(_, id)| *id).collect();
            let detector_allowed = generated
                .context_decisions
                .iter()
                .any(|decision| decision.allowed);
            let passed = if case.class == "positive" {
                detector_allowed
                    && generated_ids.get(..ALDER_V2_TARGET_IDS.len())
                        == Some(ALDER_V2_TARGET_IDS.as_slice())
                    && generated.trigger_slot == Some(specs[0].slot)
                    && generated.released_after_span
            } else {
                !generated.context_decisions.is_empty()
                    && generated
                        .context_decisions
                        .iter()
                        .all(|decision| !decision.allowed)
                    && generated_ids.first() != Some(&ALDER_V2_TARGET_IDS[0])
                    && generated.trigger_slot.is_none()
            };
            cached_passed &= passed;
            cached.push(json!({
                "class": case.class,
                "name": case.name,
                "prompt": case.prompt,
                "generated_ids": generated_ids,
                "context_decisions": generated.context_decisions,
                "trigger_slot": generated.trigger_slot,
                "released_after_span": generated.released_after_span,
                "passed": passed,
            }));
        }
    }
    Ok(json!({
        "protocol": ALDER_V2_HOLDOUT_PROTOCOL_ID,
        "status": if activation_passed && cached_passed { "passed" } else { "failed" },
        "phase_reached": if activation_passed { "cached_generation" } else { "activation" },
        "inputs": {
            "vindex": vindex,
            "source_patch": source_path,
            "source_patch_sha256": sha256_file(&source_path)?,
            "derived_patch": derived_path,
            "derived_patch_sha256": derived_sha256,
            "holdout_suite": holdout_path,
            "holdout_suite_sha256": sha256_file(&holdout_path)?,
            "active_base_checksum": active_checksum,
            "slot": {"layer": slot.0, "feature": slot.1},
            "target_ids": ALDER_V2_TARGET_IDS,
        },
        "activation": {"passed": activation_passed, "prompts": prompt_evidence},
        "cached_generation": {"run": activation_passed, "passed": cached_passed, "prompts": cached},
    }))
}

fn main() -> Result<(), AnyError> {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--preflight-only") {
        if args.len() != 3 {
            return Err(usage().into());
        }
        let result = preflight_only(Path::new(&args[2]))?;
        println!("{}", serde_json::to_string_pretty(&result)?);
        return Ok(());
    }
    let alder_v2_dev = args.get(1).map(String::as_str) == Some("--alder-v2-dev");
    let alder_v2_holdout = args.get(1).map(String::as_str) == Some("--alder-v2-holdout");
    let evidence_path = args
        .get(if alder_v2_dev { 7 } else { 6 })
        .map(PathBuf::from);
    let protocol_id = if alder_v2_dev {
        ALDER_V2_PROTOCOL_ID
    } else if alder_v2_holdout {
        ALDER_V2_HOLDOUT_PROTOCOL_ID
    } else {
        PROTOCOL_ID
    };
    let result = if alder_v2_holdout {
        run_alder_v2_holdout(&args)
    } else {
        run(&args)
    };
    match result {
        Ok(evidence) => {
            let path = evidence_path.ok_or(usage())?;
            atomic_write_json(&path, &evidence)?;
            println!("{}", serde_json::to_string_pretty(&evidence)?);
            if evidence.get("status").and_then(Value::as_str) != Some("passed") {
                return Err("terminal residual adapter spike failed its frozen gates".into());
            }
            Ok(())
        }
        Err(error) => {
            if let Some(path) = evidence_path {
                let failure = json!({
                    "protocol": protocol_id,
                    "status": "failed_or_infrastructure_inconclusive",
                    "error": error.to_string(),
                });
                let _ = atomic_write_json(&path, &failure);
            }
            Err(error)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluation_file_is_sealed_before_patch_hash() {
        let error = read_evaluation_suite(
            Path::new("this-file-must-not-be-opened.json"),
            FreezePhase::Compiling,
        )
        .err()
        .expect("gate must reject before filesystem access");
        assert!(error.to_string().contains("sealed"));
    }

    #[test]
    fn target_margin_uses_lowest_id_for_exact_ties() {
        let (margin, runner, top) = target_margin(&[2.0, 2.0, 1.5], 2).unwrap();
        assert_eq!(runner, 0);
        assert_eq!(top, 0);
        assert_eq!(margin, -0.5);
    }

    #[test]
    fn bit_identity_is_bitwise_not_numeric() {
        assert!(logits_bit_identical(&[0.0, 1.0], &[0.0, 1.0]));
        assert!(!logits_bit_identical(&[0.0], &[-0.0]));
    }

    #[test]
    fn sha256_is_deterministic() {
        assert_eq!(sha256_bytes(b"larql"), sha256_bytes(b"larql"));
        assert_ne!(sha256_bytes(b"larql"), sha256_bytes(b"LARQL"));
    }
}
