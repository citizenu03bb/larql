//! E1/E2 research harness for multi-token INSERT targets.
//!
//! This deliberately scores the complete target span.  It does not treat a
//! correct first sub-token as a successful edit.  The `oracle` phase appends
//! the expected continuation IDs after a real first-token hit, modelling the
//! hard-mask semantics of an `ActiveSpan` without conflating them with patch
//! detection.

use std::error::Error;
use std::path::{Path, PathBuf};

use larql_inference::context_detector::{
    ContextDecision, ContextDetectorSpec, ContextRepresentation,
};
use larql_inference::{InferenceWeights, KnnRouteMode};
use larql_lql::{parse, Session};
use serde::{Deserialize, Serialize};

const PROBE_TOP_K: usize = 10;
type AnyError = Box<dyn Error>;

#[derive(Clone, Deserialize)]
struct FactSpec {
    entity: String,
    relation: String,
    target: String,
    expected_tokens: Option<usize>,
    #[serde(default)]
    paraphrases: Vec<String>,
}

#[derive(Deserialize)]
struct ControlSpec {
    name: String,
    prompt: String,
    #[serde(default)]
    cell: Option<String>,
    #[serde(default)]
    fold: Option<usize>,
}

#[derive(Deserialize)]
struct SuiteSpec {
    facts: Vec<FactSpec>,
    #[serde(default)]
    controls: Vec<ControlSpec>,
}

#[derive(Serialize)]
struct TokenStep {
    phase: &'static str,
    step: usize,
    expected_id: u32,
    expected: String,
    returned_top1: String,
    top1: String,
    top1_probability: f64,
    runner_up: Option<String>,
    runner_up_probability: Option<f64>,
    top1_logit_margin: Option<f64>,
    expected_rank: Option<usize>,
    expected_probability: Option<f64>,
    top1_matches: bool,
    knn_override: bool,
    knn_override_target: Option<String>,
    knn_override_cosine: Option<f32>,
    knn_override_layer: Option<usize>,
    inference_ms: f64,
    context_decisions: Vec<ContextDecision>,
    patch_activations: Vec<larql_inference::active_span::PatchActivationEvent>,
    #[serde(skip_serializing)]
    residuals: Vec<(usize, Vec<f32>)>,
    #[serde(skip_serializing)]
    position_residuals: Vec<(usize, Vec<Vec<f32>>)>,
}

#[derive(Serialize)]
struct ResidualCapture {
    class: &'static str,
    cell: String,
    fold: usize,
    name: String,
    prompt: String,
    prompt_ids: Vec<u32>,
    prompt_token_surfaces: Vec<String>,
    top1: String,
    top1_probability: f64,
    runner_up: Option<String>,
    runner_up_probability: Option<f64>,
    top1_logit_margin: Option<f64>,
    expected_probability: Option<f64>,
    context_decisions: Vec<ContextDecision>,
    patch_activations: Vec<larql_inference::active_span::PatchActivationEvent>,
    residuals: Vec<(usize, Vec<f32>)>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    position_residuals: Vec<(usize, Vec<Vec<f32>>)>,
}

#[derive(Serialize)]
struct SpanSummary {
    phase: &'static str,
    prompt: String,
    target: String,
    target_ids: Vec<u32>,
    first_token_hit: bool,
    exact_native_span: bool,
    evaluated_steps: usize,
    oracle_armed: bool,
    emitted_exact_span: bool,
    trailing_top1: Option<String>,
    looped_to_first_target_token: bool,
    trigger_slot: Option<larql_inference::active_span::PatchSlotId>,
    trigger_gate_score: Option<f32>,
    trigger_gate_cosine: Option<f32>,
    trigger_activation: Option<f32>,
    trigger_contribution_norm: Option<f32>,
}

#[derive(Serialize)]
struct CachedSpanSummary {
    phase: &'static str,
    prompt: String,
    target_ids: Vec<u32>,
    generated_ids: Vec<u32>,
    generated_tokens: Vec<String>,
    first_token_hit: bool,
    emitted_exact_span: bool,
    released_after_span: bool,
    trigger_slot: Option<larql_inference::active_span::PatchSlotId>,
    context_decisions: Vec<ContextDecision>,
}

#[derive(Serialize)]
struct CachedValidationSummary {
    phase: &'static str,
    positive_total: usize,
    positive_passed: usize,
    negative_total: usize,
    negative_passed: usize,
    failed_positive_prompts: Vec<String>,
    failed_negative_prompts: Vec<String>,
    all_passed: bool,
}

#[derive(Default)]
struct SuiteOptions {
    mode: String,
    save_patch: Option<PathBuf>,
    apply_patch: Option<PathBuf>,
    activation_only: bool,
    install_only: bool,
    residual_out: Option<PathBuf>,
    position_residuals: bool,
    context_detector: Option<PathBuf>,
    cached_smoke: bool,
    cached_validation: bool,
}

#[derive(Deserialize)]
struct FrozenContextDetector {
    layer: usize,
    representation: ContextRepresentation,
    center: Vec<f32>,
    entity_direction: Vec<f32>,
    relation_direction: Vec<f32>,
    entity_threshold: f32,
    relation_threshold: f32,
}

fn usage() -> &'static str {
    "usage:\n  multitoken_insert_baseline <vindex> <entity> <relation> <target> [compose|knn|--native-only]\n  multitoken_insert_baseline <vindex> --suite <suite.json> [compose|knn] [--save-patch <file.vlp>] [--apply-patch <file.vlp>] [--activation-only] [--residual-out <file.json>] [--position-residuals] [--context-detector <file.json>] [--cached-smoke | --cached-validation] [--install-only]"
}

fn canonical_prompt(entity: &str, relation: &str) -> String {
    let relation_words = relation.replace(['-', '_'], " ");
    format!("The {relation_words} of {entity} is")
}

fn surface_eq(left: &str, right: &str) -> bool {
    left.trim() == right.trim()
}

fn target_tokens(
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    target: &str,
) -> Result<Vec<u32>, AnyError> {
    let encoding = tokenizer
        .encode(format!(" {target}"), false)
        .map_err(|error| std::io::Error::other(error.to_string()))?;
    Ok(encoding.get_ids().to_vec())
}

fn prompt_tokens(
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

fn infer_step(
    session: &mut Session,
    weights: &mut InferenceWeights,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    ids: &[u32],
    expected_id: u32,
    phase: &'static str,
    step: usize,
    capture_positions: bool,
    context_detectors: &[ContextDetectorSpec],
) -> Result<TokenStep, AnyError> {
    let expected = tokenizer
        .decode(&[expected_id], true)
        .map_err(|error| std::io::Error::other(error.to_string()))?;
    let patched = session
        .patched_overlay_mut()
        .ok_or("session has no vindex overlay")?;
    let embedded_context_detectors: Vec<ContextDetectorSpec> = patched
        .context_detectors()
        .into_iter()
        .map(Into::into)
        .collect();
    let context_detectors = if context_detectors.is_empty() {
        embedded_context_detectors.as_slice()
    } else {
        context_detectors
    };
    let (result, position_residuals, context_decisions) = if capture_positions {
        let captured = weights.infer_patched_with_positions(
            tokenizer,
            patched,
            Some(&patched.knn_store),
            ids,
            PROBE_TOP_K,
            &KnnRouteMode::Legacy,
        );
        (
            captured.infer,
            captured.position_residuals,
            captured.context_decisions,
        )
    } else if !context_detectors.is_empty() {
        let captured = weights.infer_patched_contextual(
            tokenizer,
            patched,
            Some(&patched.knn_store),
            ids,
            PROBE_TOP_K,
            &KnnRouteMode::Legacy,
            context_detectors,
        );
        (captured.infer, Vec::new(), captured.context_decisions)
    } else {
        (
            weights.infer_patched(
                tokenizer,
                patched,
                Some(&patched.knn_store),
                ids,
                PROBE_TOP_K,
                &KnnRouteMode::Legacy,
            ),
            Vec::new(),
            Vec::new(),
        )
    };
    let mut patch_activations = larql_inference::active_span::patch_activation_events(
        weights.as_weights(),
        patched,
        &result.residuals,
    );
    // `patch_activation_events` reconstructs potential overlay activations
    // from residuals. Contextually rejected payloads did not execute and must
    // not be reported — or allowed to arm an ActiveSpan — as real events.
    patch_activations.retain(|event| {
        let Some(_) = context_detectors
            .iter()
            .find(|spec| spec.payload == event.slot)
        else {
            return true;
        };
        context_decisions
            .iter()
            .find(|decision| decision.payload == event.slot)
            .is_some_and(|decision| decision.allowed)
    });

    let (returned_top1, _) = result
        .predictions
        .first()
        .cloned()
        .unwrap_or_else(|| (String::new(), 0.0));
    let (top1, top1_probability) = result
        .model_top1
        .clone()
        .unwrap_or_else(|| (returned_top1.clone(), 0.0));
    let model_predictions = if result.knn_override.is_some() {
        result.predictions.get(1..).unwrap_or_default()
    } else {
        result.predictions.as_slice()
    };
    let expected_at = model_predictions
        .iter()
        .enumerate()
        .find(|(_, (token, _))| surface_eq(token, &expected));
    let runner_up = model_predictions
        .iter()
        .find(|(token, _)| !surface_eq(token, &top1))
        .cloned();
    let top1_logit_margin = runner_up
        .as_ref()
        .and_then(|(_, probability)| probability_logit_margin(top1_probability, *probability));
    let top1_matches = surface_eq(&top1, &expected);
    let knn_override = result.knn_override.clone();

    Ok(TokenStep {
        phase,
        step,
        expected_id,
        expected,
        returned_top1,
        top1_matches,
        top1,
        top1_probability,
        runner_up: runner_up.as_ref().map(|(token, _)| token.clone()),
        runner_up_probability: runner_up.map(|(_, probability)| probability),
        top1_logit_margin,
        expected_rank: expected_at.map(|(rank, _)| rank + 1),
        expected_probability: expected_at.map(|(_, (_, probability))| *probability),
        knn_override: knn_override.is_some(),
        knn_override_target: knn_override.as_ref().map(|value| value.token.clone()),
        knn_override_cosine: knn_override.as_ref().map(|value| value.cosine),
        knn_override_layer: knn_override.as_ref().map(|value| value.layer),
        inference_ms: result.walk_ms,
        context_decisions,
        patch_activations,
        residuals: result.residuals,
        position_residuals,
    })
}

fn probability_logit_margin(top1: f64, runner_up: f64) -> Option<f64> {
    if top1.is_finite() && runner_up.is_finite() && top1 > 0.0 && runner_up > 0.0 {
        Some(top1.ln() - runner_up.ln())
    } else {
        None
    }
}

fn load_weights(
    vindex: &Path,
    config: &larql_vindex::VindexConfig,
) -> Result<InferenceWeights, AnyError> {
    let mut callbacks = larql_vindex::SilentLoadCallbacks;
    Ok(InferenceWeights::load(vindex, config, &mut callbacks)?)
}

fn install_fact(session: &mut Session, fact: &FactSpec, mode_sql: &str) -> Result<(), AnyError> {
    let insert_sql = format!(
        "INSERT INTO EDGES (entity, relation, target) VALUES ({}, {}, {}) MODE {mode_sql};",
        serde_json::to_string(&fact.entity)?,
        serde_json::to_string(&fact.relation)?,
        serde_json::to_string(&fact.target)?,
    );
    let insert = parse(&insert_sql)?;
    for line in session.execute(&insert)? {
        eprintln!("{line}");
    }
    Ok(())
}

fn probe_control(
    session: &mut Session,
    vindex: &Path,
    config: &larql_vindex::VindexConfig,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    control: &ControlSpec,
    fact: &FactSpec,
    expected_id: u32,
    phase: &'static str,
) -> Result<(), AnyError> {
    let mut weights = load_weights(vindex, config)?;
    let ids = prompt_tokens(tokenizer, &weights, &control.prompt)?;
    let result = infer_step(
        session,
        &mut weights,
        tokenizer,
        &ids,
        expected_id,
        phase,
        0,
        false,
        &[],
    )?;
    println!(
        "{}",
        serde_json::json!({
            "kind": "control_probe",
            "phase": phase,
            "control": control.name,
            "prompt": control.prompt,
            "target_entity": fact.entity,
            "target": fact.target,
            "probe": result,
        })
    );
    Ok(())
}

fn run_native_phase(
    session: &mut Session,
    vindex: &Path,
    config: &larql_vindex::VindexConfig,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    prompt: &str,
    target: &str,
    target_ids: &[u32],
    phase: &'static str,
) -> Result<SpanSummary, AnyError> {
    let mut weights = load_weights(vindex, config)?;
    let mut ids = prompt_tokens(tokenizer, &weights, prompt)?;
    let mut first_token_hit = false;
    let mut exact = true;
    let mut evaluated_steps = 0;

    for (step, expected_id) in target_ids.iter().copied().enumerate() {
        let result = infer_step(
            session,
            &mut weights,
            tokenizer,
            &ids,
            expected_id,
            phase,
            step,
            false,
            &[],
        )?;
        if step == 0 {
            first_token_hit = result.top1_matches;
        }
        evaluated_steps += 1;
        println!("{}", serde_json::to_string(&result)?);
        if !result.top1_matches {
            exact = false;
            break;
        }
        // Appending the known expected ID is safe only after the model chose
        // the same token surface.  This avoids decode→re-tokenize drift.
        ids.push(expected_id);
    }

    Ok(SpanSummary {
        phase,
        prompt: prompt.to_string(),
        target: target.to_string(),
        target_ids: target_ids.to_vec(),
        first_token_hit,
        exact_native_span: exact && evaluated_steps == target_ids.len(),
        evaluated_steps,
        oracle_armed: false,
        emitted_exact_span: false,
        trailing_top1: None,
        looped_to_first_target_token: false,
        trigger_slot: None,
        trigger_gate_score: None,
        trigger_gate_cosine: None,
        trigger_activation: None,
        trigger_contribution_norm: None,
    })
}

fn run_oracle_phase(
    session: &mut Session,
    vindex: &Path,
    config: &larql_vindex::VindexConfig,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    prompt: &str,
    target: &str,
    target_ids: &[u32],
    span_specs: &[larql_inference::active_span::ActiveSpanSpec],
) -> Result<SpanSummary, AnyError> {
    let phase = "post_insert_oracle";
    let mut weights = load_weights(vindex, config)?;
    let mut ids = prompt_tokens(tokenizer, &weights, prompt)?;
    let mut evaluated_steps = 0;
    let mut native_exact = true;
    let mut first_token_hit = false;
    let mut active_span: Option<larql_inference::active_span::ActiveSpan> = None;
    let mut trigger_gate_score = None;
    let mut trigger_gate_cosine = None;
    let mut trigger_activation = None;
    let mut trigger_contribution_norm = None;

    for (step, expected_id) in target_ids.iter().copied().enumerate() {
        let result = infer_step(
            session,
            &mut weights,
            tokenizer,
            &ids,
            expected_id,
            phase,
            step,
            false,
            &[],
        )?;
        if step == 0 {
            first_token_hit = result.top1_matches;
            if !first_token_hit {
                println!("{}", serde_json::to_string(&result)?);
                return Ok(SpanSummary {
                    phase,
                    prompt: prompt.to_string(),
                    target: target.to_string(),
                    target_ids: target_ids.to_vec(),
                    first_token_hit: false,
                    exact_native_span: false,
                    evaluated_steps: 1,
                    oracle_armed: false,
                    emitted_exact_span: false,
                    trailing_top1: None,
                    looped_to_first_target_token: false,
                    trigger_slot: None,
                    trigger_gate_score: None,
                    trigger_gate_cosine: None,
                    trigger_activation: None,
                    trigger_contribution_norm: None,
                });
            }
            active_span = larql_inference::active_span::ActiveSpan::try_arm(
                expected_id,
                span_specs,
                &result.patch_activations,
            );
            let Some(ref span) = active_span else {
                println!("{}", serde_json::to_string(&result)?);
                return Ok(SpanSummary {
                    phase,
                    prompt: prompt.to_string(),
                    target: target.to_string(),
                    target_ids: target_ids.to_vec(),
                    first_token_hit: true,
                    exact_native_span: false,
                    evaluated_steps: 1,
                    oracle_armed: false,
                    emitted_exact_span: false,
                    trailing_top1: None,
                    looped_to_first_target_token: false,
                    trigger_slot: None,
                    trigger_gate_score: None,
                    trigger_gate_cosine: None,
                    trigger_activation: None,
                    trigger_contribution_norm: None,
                });
            };
            if let Some(event) = result
                .patch_activations
                .iter()
                .find(|event| event.slot == span.slot())
            {
                trigger_gate_score = Some(event.gate_score);
                trigger_gate_cosine = event.gate_cosine;
                trigger_activation = Some(event.activation);
                trigger_contribution_norm = event.contribution_norm;
            }
        } else {
            let span = active_span
                .as_mut()
                .expect("step > 0 only after the slot armed an ActiveSpan");
            if span.expected_next() != Some(expected_id) {
                return Err(format!(
                    "ActiveSpan expected {:?}, suite expected {expected_id}",
                    span.expected_next()
                )
                .into());
            }
            // The E1 harness scores one full forward per token. Appending the
            // exact ID models the same hard-mask selection as the cached E2
            // decoder; `observe_emitted` validates and advances the state.
            span.observe_emitted(expected_id);
        }
        native_exact &= result.top1_matches;
        evaluated_steps += 1;
        println!("{}", serde_json::to_string(&result)?);

        // Step 0 was selected naturally.  For continuation steps this append
        // is the result an exact-one-token hard mask would select.
        ids.push(expected_id);
    }

    // The emitter must be disarmed now.  One unmasked step detects the common
    // failure where the target immediately loops back to its first sub-token.
    let first_id = *target_ids.first().ok_or("target tokenized to zero IDs")?;
    let trailing = infer_step(
        session,
        &mut weights,
        tokenizer,
        &ids,
        first_id,
        "post_span_unmasked",
        target_ids.len(),
        false,
        &[],
    )?;
    let trailing_top1 = trailing.top1.clone();
    let looped = trailing.top1_matches;
    println!("{}", serde_json::to_string(&trailing)?);

    Ok(SpanSummary {
        phase,
        prompt: prompt.to_string(),
        target: target.to_string(),
        target_ids: target_ids.to_vec(),
        first_token_hit,
        exact_native_span: native_exact,
        evaluated_steps,
        oracle_armed: true,
        emitted_exact_span: true,
        trailing_top1: Some(trailing_top1),
        looped_to_first_target_token: looped,
        trigger_slot: active_span.as_ref().map(|span| span.slot()),
        trigger_gate_score,
        trigger_gate_cosine,
        trigger_activation,
        trigger_contribution_norm,
    })
}

fn run_cached_phase(
    session: &mut Session,
    inference_weights: &mut InferenceWeights,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    prompt: &str,
    target_ids: &[u32],
    span_specs: &[larql_inference::active_span::ActiveSpanSpec],
    phase: &'static str,
    max_tokens: usize,
) -> Result<CachedSpanSummary, AnyError> {
    let prompt_ids = prompt_tokens(tokenizer, inference_weights, prompt)?;
    let patched = session
        .patched_overlay()
        .ok_or("session has no vindex overlay")?;
    let result = inference_weights.generate_patched(
        tokenizer,
        patched,
        &prompt_ids,
        max_tokens,
        span_specs,
    )?;
    let generated = result.generated;
    let generated_ids: Vec<_> = generated.iter().map(|(_, id)| *id).collect();
    let generated_tokens: Vec<_> = generated.into_iter().map(|(token, _)| token).collect();
    let emitted_exact_span = generated_ids.get(..target_ids.len()) == Some(target_ids);

    Ok(CachedSpanSummary {
        phase,
        prompt: prompt.to_string(),
        target_ids: target_ids.to_vec(),
        first_token_hit: generated_ids.first() == target_ids.first(),
        emitted_exact_span,
        released_after_span: emitted_exact_span && result.released_after_span,
        trigger_slot: result.trigger_slot,
        generated_ids,
        generated_tokens,
        context_decisions: result.context_decisions,
    })
}

fn cached_summary_passes(summary: &CachedSpanSummary, expect_positive: bool) -> bool {
    let detector_allowed = summary
        .context_decisions
        .iter()
        .any(|decision| decision.allowed);
    let detector_rejected = !summary.context_decisions.is_empty()
        && summary
            .context_decisions
            .iter()
            .all(|decision| !decision.allowed);

    if expect_positive {
        detector_allowed
            && summary.first_token_hit
            && summary.emitted_exact_span
            && summary.released_after_span
            && summary.trigger_slot.is_some()
    } else {
        detector_rejected
            && !summary.first_token_hit
            && !summary.emitted_exact_span
            && summary.trigger_slot.is_none()
    }
}

fn open_session(vindex: &Path) -> Result<Session, AnyError> {
    let mut session = Session::new();
    let use_stmt = parse(&format!(r#"USE "{}";"#, vindex.display()))?;
    for line in session.execute(&use_stmt)? {
        eprintln!("{line}");
    }
    Ok(session)
}

fn mode_sql(mode: &str) -> Result<&'static str, AnyError> {
    match mode.to_ascii_lowercase().as_str() {
        "compose" => Ok("COMPOSE"),
        "knn" => Ok("KNN"),
        _ => Err(usage().into()),
    }
}

fn parse_suite_options(args: &[String]) -> Result<SuiteOptions, AnyError> {
    let mut options = SuiteOptions {
        mode: "compose".into(),
        ..SuiteOptions::default()
    };
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "compose" | "knn" => options.mode = args[index].clone(),
            "--save-patch" => {
                index += 1;
                options.save_patch =
                    Some(PathBuf::from(args.get(index).ok_or_else(|| {
                        std::io::Error::other("--save-patch requires a path")
                    })?));
            }
            "--apply-patch" => {
                index += 1;
                options.apply_patch =
                    Some(PathBuf::from(args.get(index).ok_or_else(|| {
                        std::io::Error::other("--apply-patch requires a path")
                    })?));
            }
            "--activation-only" => options.activation_only = true,
            "--position-residuals" => options.position_residuals = true,
            "--context-detector" => {
                index += 1;
                options.context_detector =
                    Some(PathBuf::from(args.get(index).ok_or_else(|| {
                        std::io::Error::other("--context-detector requires a path")
                    })?));
            }
            "--residual-out" => {
                index += 1;
                options.residual_out =
                    Some(PathBuf::from(args.get(index).ok_or_else(|| {
                        std::io::Error::other("--residual-out requires a path")
                    })?));
            }
            "--install-only" => options.install_only = true,
            "--cached-smoke" => options.cached_smoke = true,
            "--cached-validation" => options.cached_validation = true,
            _ => return Err(usage().into()),
        }
        index += 1;
    }
    if options.save_patch.is_some() && options.apply_patch.is_some() {
        return Err("--save-patch and --apply-patch are mutually exclusive".into());
    }
    if options.install_only && options.save_patch.is_none() && options.apply_patch.is_none() {
        return Err("--install-only requires --save-patch or --apply-patch".into());
    }
    if (options.cached_smoke || options.cached_validation) && options.apply_patch.is_none() {
        return Err("--cached-smoke/--cached-validation requires --apply-patch".into());
    }
    if options.cached_smoke && options.cached_validation {
        return Err("--cached-smoke and --cached-validation are mutually exclusive".into());
    }
    if options.residual_out.is_some() && !options.activation_only {
        return Err("--residual-out requires --activation-only".into());
    }
    if options.position_residuals && !options.activation_only {
        return Err("--position-residuals requires --activation-only".into());
    }
    if options.context_detector.is_some() && !options.activation_only {
        return Err("--context-detector currently requires --activation-only".into());
    }
    if options.context_detector.is_some() && options.position_residuals {
        return Err("--context-detector and --position-residuals are mutually exclusive".into());
    }
    Ok(options)
}

fn execute_statement(session: &mut Session, sql: &str) -> Result<(), AnyError> {
    let statement = parse(sql)?;
    for line in session.execute(&statement)? {
        eprintln!("{line}");
    }
    Ok(())
}

fn span_specs_from_patch(
    patch: &larql_vindex::VindexPatch,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
) -> Result<Vec<larql_inference::active_span::ActiveSpanSpec>, AnyError> {
    let mut specs = Vec::new();
    let mut persisted_slots = std::collections::HashSet::new();
    for operation in &patch.operations {
        let larql_vindex::PatchOp::Context {
            layer,
            feature,
            target_ids,
            ..
        } = operation
        else {
            continue;
        };
        if target_ids.len() > 1 {
            persisted_slots.insert((*layer, *feature));
            specs.push(larql_inference::active_span::ActiveSpanSpec {
                slot: larql_inference::active_span::PatchSlotId {
                    layer: *layer,
                    feature: *feature,
                },
                target_ids: target_ids.clone(),
            });
        }
    }
    for operation in &patch.operations {
        let larql_vindex::PatchOp::Insert {
            layer,
            feature,
            target,
            down_meta,
            ..
        } = operation
        else {
            continue;
        };
        if persisted_slots.contains(&(*layer, *feature)) {
            continue;
        }
        let ids = target_tokens(tokenizer, target)?;
        if ids.len() < 2 {
            continue;
        }
        if let Some(meta) = down_meta {
            if ids.first().copied() != Some(meta.top_token_id) {
                return Err(format!(
                    "patch slot L{layer}/F{feature} target {target:?} tokenizes to {:?}, but patch records first ID {}",
                    ids, meta.top_token_id
                )
                .into());
            }
        }
        specs.push(larql_inference::active_span::ActiveSpanSpec {
            slot: larql_inference::active_span::PatchSlotId {
                layer: *layer,
                feature: *feature,
            },
            target_ids: ids,
        });
    }
    Ok(specs)
}

fn load_context_detectors(
    path: Option<&Path>,
    span_specs: &[larql_inference::active_span::ActiveSpanSpec],
) -> Result<Vec<ContextDetectorSpec>, AnyError> {
    let Some(path) = path else {
        return Ok(Vec::new());
    };
    let frozen: FrozenContextDetector = serde_json::from_slice(&std::fs::read(path)?)?;
    let [span] = span_specs else {
        return Err(format!(
            "--context-detector requires exactly one multi-token payload, found {}",
            span_specs.len()
        )
        .into());
    };
    if frozen.layer >= span.slot.layer {
        return Err(format!(
            "detector layer L{} must precede payload L{}",
            frozen.layer, span.slot.layer
        )
        .into());
    }
    Ok(vec![ContextDetectorSpec {
        detector_layer: frozen.layer,
        representation: frozen.representation,
        center: frozen.center,
        entity_direction: frozen.entity_direction,
        relation_direction: frozen.relation_direction,
        entity_threshold: frozen.entity_threshold,
        relation_threshold: frozen.relation_threshold,
        payload: span.slot,
    }])
}

fn probe_activation_only(
    session: &mut Session,
    weights: &mut InferenceWeights,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    prompt: &str,
    fact: &FactSpec,
    expected_id: u32,
    class: &'static str,
    cell: &str,
    fold: usize,
    name: &str,
    capture_positions: bool,
    context_detectors: &[ContextDetectorSpec],
) -> Result<ResidualCapture, AnyError> {
    let ids = prompt_tokens(tokenizer, weights, prompt)?;
    let prompt_token_surfaces = ids
        .iter()
        .map(|id| {
            tokenizer
                .decode(&[*id], true)
                .unwrap_or_else(|_| format!("<{id}>"))
        })
        .collect();
    let mut result = infer_step(
        session,
        weights,
        tokenizer,
        &ids,
        expected_id,
        "post_patch_activation",
        0,
        capture_positions,
        context_detectors,
    )?;
    let residuals = std::mem::take(&mut result.residuals);
    let position_residuals = std::mem::take(&mut result.position_residuals);
    let patch_activations = result.patch_activations.clone();
    let context_decisions = result.context_decisions.clone();
    let top1 = result.top1.clone();
    let top1_probability = result.top1_probability;
    let runner_up = result.runner_up.clone();
    let runner_up_probability = result.runner_up_probability;
    let top1_logit_margin = result.top1_logit_margin;
    let expected_probability = result.expected_probability;
    println!(
        "{}",
        serde_json::json!({
            "kind": "activation_probe",
            "class": class,
            "name": name,
            "prompt": prompt,
            "target_entity": fact.entity,
            "target": fact.target,
            "probe": result,
        })
    );
    Ok(ResidualCapture {
        class,
        cell: cell.to_string(),
        fold,
        name: name.to_string(),
        prompt: prompt.to_string(),
        prompt_ids: ids,
        prompt_token_surfaces,
        top1,
        top1_probability,
        runner_up,
        runner_up_probability,
        top1_logit_margin,
        expected_probability,
        context_decisions,
        patch_activations,
        residuals,
        position_residuals,
    })
}

fn run_suite(vindex: &Path, suite_path: &Path, options: &SuiteOptions) -> Result<(), AnyError> {
    let mode_sql = mode_sql(&options.mode)?;
    let suite: SuiteSpec = serde_json::from_slice(&std::fs::read(suite_path)?)?;
    if suite.facts.is_empty() {
        return Err("suite must contain at least one fact".into());
    }

    let config = larql_vindex::load_vindex_config(vindex)?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(vindex)?;
    let tokenized: Vec<(FactSpec, Vec<u32>)> = suite
        .facts
        .iter()
        .cloned()
        .map(|fact| {
            let ids = target_tokens(&tokenizer, &fact.target)?;
            if ids.is_empty() {
                return Err(format!("target {:?} tokenized to zero IDs", fact.target).into());
            }
            if let Some(expected) = fact.expected_tokens {
                if ids.len() != expected {
                    return Err(format!(
                        "target {:?} expected T={expected}, tokenizer produced T={} ({ids:?})",
                        fact.target,
                        ids.len()
                    )
                    .into());
                }
            }
            Ok((fact, ids))
        })
        .collect::<Result<_, AnyError>>()?;
    println!(
        "{}",
        serde_json::json!({
            "kind": "suite",
            "phase": "start",
            "mode": mode_sql,
            "apply_patch": options.apply_patch,
            "activation_only": options.activation_only,
            "install_only": options.install_only,
            "residual_out": options.residual_out,
            "position_residuals": options.position_residuals,
            "context_detector": options.context_detector,
            "cached_smoke": options.cached_smoke,
            "cached_validation": options.cached_validation,
            "n_facts": tokenized.len(),
            "targets": tokenized.iter().map(|(fact, ids)| serde_json::json!({
                "entity": fact.entity,
                "relation": fact.relation,
                "target": fact.target,
                "ids": ids,
            })).collect::<Vec<_>>(),
        })
    );
    if std::env::var_os("LARQL_SUITE_TOKENIZE_ONLY").is_some() {
        return Ok(());
    }
    let mut session = open_session(vindex)?;

    // Measure every positive and negative before any edit is installed.
    if options.apply_patch.is_none() && !options.install_only && !options.activation_only {
        for (fact, ids) in &tokenized {
            let canonical = canonical_prompt(&fact.entity, &fact.relation);
            for prompt in std::iter::once(&canonical).chain(fact.paraphrases.iter()) {
                let summary = run_native_phase(
                    &mut session,
                    vindex,
                    &config,
                    &tokenizer,
                    prompt,
                    &fact.target,
                    ids,
                    "pre_insert_native",
                )?;
                println!("{}", serde_json::to_string(&summary)?);
            }
            for control in &suite.controls {
                probe_control(
                    &mut session,
                    vindex,
                    &config,
                    &tokenizer,
                    control,
                    fact,
                    ids[0],
                    "pre_insert_control",
                )?;
            }
        }
    }

    let span_specs = if let Some(patch_path) = &options.apply_patch {
        let patch = larql_vindex::VindexPatch::load(patch_path)?;
        let legacy_specs = span_specs_from_patch(&patch, &tokenizer)?;
        execute_statement(
            &mut session,
            &format!("APPLY PATCH {};", serde_json::to_string(patch_path)?),
        )?;
        let persisted_specs = session.active_span_specs();
        for operation in &patch.operations {
            let larql_vindex::PatchOp::Context {
                layer,
                feature,
                target_ids,
                ..
            } = operation
            else {
                continue;
            };
            if !persisted_specs.iter().any(|spec| {
                spec.slot.layer == *layer
                    && spec.slot.feature == *feature
                    && spec.target_ids == *target_ids
            }) {
                return Err(format!(
                    "applied contextual payload L{layer}/F{feature} did not expose its persisted ActiveSpan specification"
                )
                .into());
            }
        }
        if persisted_specs.is_empty() {
            legacy_specs
        } else {
            persisted_specs
        }
    } else {
        if let Some(patch_path) = &options.save_patch {
            execute_statement(
                &mut session,
                &format!("BEGIN PATCH {};", serde_json::to_string(patch_path)?),
            )?;
        }
        // Install all facts before post-measurement. This is the material N>1
        // distinction: edits coexist in one PatchedVindex/session.
        for (fact, _) in &tokenized {
            install_fact(&mut session, fact, mode_sql)?;
        }
        let specs = session.active_span_specs();
        if options.save_patch.is_some() {
            execute_statement(&mut session, "SAVE PATCH;")?;
        }
        specs
    };

    let context_detectors =
        load_context_detectors(options.context_detector.as_deref(), &span_specs)?;

    if options.cached_smoke || options.cached_validation {
        let (fact, target_ids) = tokenized.first().ok_or("cached smoke requires one fact")?;
        let mut cached_weights = load_weights(vindex, &config)?;
        let canonical = canonical_prompt(&fact.entity, &fact.relation);
        let positive_prompts: Vec<&str> = if options.cached_smoke {
            vec![canonical.as_str()]
        } else {
            std::iter::once(canonical.as_str())
                .chain(fact.paraphrases.iter().map(String::as_str))
                .collect()
        };
        let controls: Vec<&ControlSpec> = if options.cached_smoke {
            vec![suite
                .controls
                .first()
                .ok_or("cached smoke requires at least one control")?]
        } else {
            suite.controls.iter().collect()
        };

        let mut failed_positive_prompts = Vec::new();
        for prompt in &positive_prompts {
            let summary = run_cached_phase(
                &mut session,
                &mut cached_weights,
                &tokenizer,
                prompt,
                target_ids,
                &span_specs,
                "cached_positive",
                target_ids.len() + 1,
            )?;
            if !cached_summary_passes(&summary, true) {
                failed_positive_prompts.push((*prompt).to_string());
            }
            println!("{}", serde_json::to_string(&summary)?);
        }

        let mut failed_negative_prompts = Vec::new();
        for control in &controls {
            let summary = run_cached_phase(
                &mut session,
                &mut cached_weights,
                &tokenizer,
                &control.prompt,
                target_ids,
                &span_specs,
                "cached_negative",
                if options.cached_smoke {
                    target_ids.len() + 1
                } else {
                    1
                },
            )?;
            if !cached_summary_passes(&summary, false) {
                failed_negative_prompts.push(control.prompt.clone());
            }
            println!("{}", serde_json::to_string(&summary)?);
        }

        let validation = CachedValidationSummary {
            phase: if options.cached_smoke {
                "cached_smoke_summary"
            } else {
                "cached_validation_summary"
            },
            positive_total: positive_prompts.len(),
            positive_passed: positive_prompts.len() - failed_positive_prompts.len(),
            negative_total: controls.len(),
            negative_passed: controls.len() - failed_negative_prompts.len(),
            all_passed: failed_positive_prompts.is_empty() && failed_negative_prompts.is_empty(),
            failed_positive_prompts,
            failed_negative_prompts,
        };
        let all_passed = validation.all_passed;
        println!("{}", serde_json::to_string(&validation)?);
        if !all_passed {
            return Err("cached validation failed".into());
        }
        return Ok(());
    }

    if options.install_only {
        return Ok(());
    }

    if options.activation_only {
        // A full E4B load is the dominant memory and latency cost. Reuse one
        // immutable weight set across the suite instead of reloading 12 GiB
        // for every independent prompt capture.
        let mut weights = load_weights(vindex, &config)?;
        let mut captures = Vec::new();
        for (fact, ids) in &tokenized {
            let canonical = canonical_prompt(&fact.entity, &fact.relation);
            for (index, prompt) in std::iter::once(&canonical)
                .chain(fact.paraphrases.iter())
                .enumerate()
            {
                let name = if index == 0 {
                    "canonical".to_string()
                } else {
                    format!("paraphrase-{index}")
                };
                captures.push(probe_activation_only(
                    &mut session,
                    &mut weights,
                    &tokenizer,
                    prompt,
                    fact,
                    ids[0],
                    "positive",
                    "11",
                    index,
                    &name,
                    options.position_residuals,
                    &context_detectors,
                )?);
            }
            for control in &suite.controls {
                captures.push(probe_activation_only(
                    &mut session,
                    &mut weights,
                    &tokenizer,
                    &control.prompt,
                    fact,
                    ids[0],
                    "control",
                    control.cell.as_deref().unwrap_or("control"),
                    control.fold.unwrap_or(0),
                    &control.name,
                    options.position_residuals,
                    &context_detectors,
                )?);
            }
        }
        if let Some(path) = &options.residual_out {
            let file = std::fs::File::create(path)?;
            serde_json::to_writer(std::io::BufWriter::new(file), &captures)?;
            eprintln!("Saved residual captures: {}", path.display());
        }
        return Ok(());
    }

    for (fact, ids) in &tokenized {
        let canonical = canonical_prompt(&fact.entity, &fact.relation);
        for prompt in std::iter::once(&canonical).chain(fact.paraphrases.iter()) {
            let post = run_native_phase(
                &mut session,
                vindex,
                &config,
                &tokenizer,
                prompt,
                &fact.target,
                ids,
                "post_insert_native",
            )?;
            println!("{}", serde_json::to_string(&post)?);
            if ids.len() > 1 {
                let oracle = run_oracle_phase(
                    &mut session,
                    vindex,
                    &config,
                    &tokenizer,
                    prompt,
                    &fact.target,
                    ids,
                    &span_specs,
                )?;
                println!("{}", serde_json::to_string(&oracle)?);
            }
        }
        for control in &suite.controls {
            probe_control(
                &mut session,
                vindex,
                &config,
                &tokenizer,
                control,
                fact,
                ids[0],
                "post_insert_control",
            )?;
        }
    }
    Ok(())
}

fn main() -> Result<(), AnyError> {
    let args: Vec<String> = std::env::args().collect();
    if args.get(2).map(String::as_str) == Some("--suite") {
        if args.len() < 4 {
            return Err(usage().into());
        }
        let options = parse_suite_options(&args[4..])?;
        return run_suite(Path::new(&args[1]), Path::new(&args[3]), &options);
    }
    if args.len() < 5 || args.len() > 6 {
        return Err(usage().into());
    }

    let vindex = PathBuf::from(&args[1]);
    let entity = &args[2];
    let relation = &args[3];
    let target = &args[4];
    let mode = args.get(5).map(String::as_str).unwrap_or("compose");
    let native_only = mode == "--native-only";
    let mode_sql = if native_only {
        "COMPOSE"
    } else {
        mode_sql(mode)?
    };

    let config = larql_vindex::load_vindex_config(&vindex)?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(&vindex)?;
    let target_ids = target_tokens(&tokenizer, target)?;
    if target_ids.is_empty() {
        return Err("target tokenized to zero IDs".into());
    }
    let prompt = canonical_prompt(entity, relation);

    eprintln!(
        "case: entity={entity:?} relation={relation:?} target={target:?} ids={target_ids:?} mode={mode_sql}"
    );

    let mut session = open_session(&vindex)?;

    let pre = run_native_phase(
        &mut session,
        &vindex,
        &config,
        &tokenizer,
        &prompt,
        target,
        &target_ids,
        "pre_insert_native",
    )?;
    println!("{}", serde_json::to_string(&pre)?);

    if native_only {
        return Ok(());
    }

    install_fact(
        &mut session,
        &FactSpec {
            entity: entity.clone(),
            relation: relation.clone(),
            target: target.clone(),
            expected_tokens: None,
            paraphrases: Vec::new(),
        },
        mode_sql,
    )?;

    let post = run_native_phase(
        &mut session,
        &vindex,
        &config,
        &tokenizer,
        &prompt,
        target,
        &target_ids,
        "post_insert_native",
    )?;
    println!("{}", serde_json::to_string(&post)?);

    if target_ids.len() > 1 {
        let span_specs = session.active_span_specs();
        let oracle = run_oracle_phase(
            &mut session,
            &vindex,
            &config,
            &tokenizer,
            &prompt,
            target,
            &target_ids,
            &span_specs,
        )?;
        println!("{}", serde_json::to_string(&oracle)?);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_inference::active_span::PatchSlotId;

    fn decision(allowed: bool) -> ContextDecision {
        ContextDecision {
            payload: PatchSlotId {
                layer: 31,
                feature: 1625,
            },
            detector_layer: 6,
            entity_margin: if allowed { 1.0 } else { -1.0 },
            relation_margin: if allowed { 1.0 } else { -1.0 },
            allowed,
        }
    }

    fn summary(positive: bool) -> CachedSpanSummary {
        CachedSpanSummary {
            phase: "test",
            prompt: "prompt".into(),
            target_ids: vec![1, 2, 3],
            generated_ids: if positive {
                vec![1, 2, 3, 4]
            } else {
                vec![9, 8, 7, 6]
            },
            generated_tokens: Vec::new(),
            first_token_hit: positive,
            emitted_exact_span: positive,
            released_after_span: positive,
            trigger_slot: positive.then_some(PatchSlotId {
                layer: 31,
                feature: 1625,
            }),
            context_decisions: vec![decision(positive)],
        }
    }

    #[test]
    fn cached_validation_accepts_strict_positive_and_negative_shapes() {
        assert!(cached_summary_passes(&summary(true), true));
        assert!(cached_summary_passes(&summary(false), false));
        assert!(!cached_summary_passes(&summary(true), false));
        assert!(!cached_summary_passes(&summary(false), true));
    }

    #[test]
    fn cached_validation_requires_a_patch_and_excludes_smoke() {
        let no_patch = parse_suite_options(&["--cached-validation".into()]);
        assert!(no_patch.is_err());

        let both = parse_suite_options(&[
            "--apply-patch".into(),
            "patch.vlp".into(),
            "--cached-smoke".into(),
            "--cached-validation".into(),
        ]);
        assert!(both.is_err());

        let validation = parse_suite_options(&[
            "--apply-patch".into(),
            "patch.vlp".into(),
            "--cached-validation".into(),
        ])
        .expect("cached validation options");
        assert!(validation.cached_validation);
        assert!(!validation.cached_smoke);
    }

    #[test]
    fn probability_margin_matches_the_softmax_logit_difference() {
        let margin = probability_logit_margin(0.4, 0.2).expect("positive probabilities");
        assert!((margin - 2.0_f64.ln()).abs() < 1e-12);
        assert_eq!(probability_logit_margin(0.0, 0.2), None);
        assert_eq!(probability_logit_margin(0.4, f64::NAN), None);
    }
}
