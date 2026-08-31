//! Compile an exact product-bounded surface manifest without loading weights.
//!
//! The command prints the complete `CertifiedPatch | UncompilableEdit`
//! decision as JSON. Exit code 0 means certified; exit code 2 is a structured
//! refusal. The output manifest is written only after in-memory certification.

use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use larql_lql::product_surface_compiler::{
    compile_product_surface_context, ProductSurfaceCompilationStage, ProductSurfaceCompileOutcome,
    ProductSurfaceCompileRequest, UncompilableEdit, UncompilableEditCode,
};

fn usage() -> &'static str {
    "usage: compile_product_surface_context <vindex> <adapter.vlp> <surfaces.json> <output.json>"
}

fn refusal(
    stage: ProductSurfaceCompilationStage,
    code: UncompilableEditCode,
    reason: impl Into<String>,
    retryable: bool,
) -> ProductSurfaceCompileOutcome {
    ProductSurfaceCompileOutcome::refused(UncompilableEdit::new(stage, code, reason, retryable))
}

fn commit_certified_bytes(output: &std::path::Path, bytes: &[u8]) -> Result<(), UncompilableEdit> {
    match std::fs::read(output) {
        Ok(existing) if existing == bytes => return Ok(()),
        Ok(_) => {
            return Err(UncompilableEdit::new(
                ProductSurfaceCompilationStage::Commit,
                UncompilableEditCode::OutputCommitFailed,
                format!(
                    "refusing to overwrite existing non-identical output {}",
                    output.display()
                ),
                false,
            ));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(UncompilableEdit::new(
                ProductSurfaceCompilationStage::Commit,
                UncompilableEditCode::OutputCommitFailed,
                format!("cannot inspect output {}: {error}", output.display()),
                true,
            ));
        }
    }

    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    let mut temporary = tempfile::NamedTempFile::new_in(parent).map_err(|error| {
        UncompilableEdit::new(
            ProductSurfaceCompilationStage::Commit,
            UncompilableEditCode::OutputCommitFailed,
            format!(
                "cannot create temporary output beside {}: {error}",
                output.display()
            ),
            true,
        )
    })?;
    temporary.write_all(bytes).map_err(|error| {
        UncompilableEdit::new(
            ProductSurfaceCompilationStage::Commit,
            UncompilableEditCode::OutputCommitFailed,
            format!("cannot write certified manifest bytes: {error}"),
            true,
        )
    })?;
    temporary.as_file().sync_all().map_err(|error| {
        UncompilableEdit::new(
            ProductSurfaceCompilationStage::Commit,
            UncompilableEditCode::OutputCommitFailed,
            format!("cannot sync certified manifest bytes: {error}"),
            true,
        )
    })?;
    temporary.persist_noclobber(output).map_err(|error| {
        UncompilableEdit::new(
            ProductSurfaceCompilationStage::Commit,
            UncompilableEditCode::OutputCommitFailed,
            format!(
                "cannot atomically commit certified manifest to {}: {}",
                output.display(),
                error.error
            ),
            error.error.kind() != std::io::ErrorKind::AlreadyExists,
        )
    })?;
    Ok(())
}

fn run(args: &[String]) -> ProductSurfaceCompileOutcome {
    if args.len() != 5 {
        return refusal(
            ProductSurfaceCompilationStage::Request,
            UncompilableEditCode::InvalidInvocation,
            usage(),
            false,
        );
    }
    let vindex = PathBuf::from(&args[1]);
    let source_patch = PathBuf::from(&args[2]);
    let surface_bank = PathBuf::from(&args[3]);
    let output = PathBuf::from(&args[4]);
    let outcome = compile_product_surface_context(ProductSurfaceCompileRequest {
        vindex: &vindex,
        source_patch: &source_patch,
        surface_bank: &surface_bank,
    });
    let Some(certified) = outcome.certified_patch() else {
        return outcome;
    };
    let bytes = match certified.manifest_bytes() {
        Ok(bytes) => bytes,
        Err(refusal) => return ProductSurfaceCompileOutcome::refused(refusal),
    };
    if let Err(refusal) = commit_certified_bytes(&output, &bytes) {
        return ProductSurfaceCompileOutcome::refused(refusal);
    }
    outcome
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let outcome = run(&args);
    match serde_json::to_string_pretty(&outcome) {
        Ok(json) => println!("{json}"),
        Err(error) => {
            eprintln!("failed to serialize compiler outcome: {error}");
            return ExitCode::FAILURE;
        }
    }
    match outcome {
        ProductSurfaceCompileOutcome::CertifiedPatch { .. } => ExitCode::SUCCESS,
        ProductSurfaceCompileOutcome::UncompilableEdit { .. } => ExitCode::from(2),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_invocation_is_a_structured_non_retryable_refusal() {
        let outcome = run(&["compile_product_surface_context".into()]);
        assert!(matches!(
            outcome,
            ProductSurfaceCompileOutcome::UncompilableEdit {
                refusal: UncompilableEdit {
                    stage: ProductSurfaceCompilationStage::Request,
                    code: UncompilableEditCode::InvalidInvocation,
                    retryable: false,
                    ..
                }
            }
        ));
    }

    #[test]
    fn commit_is_idempotent_and_never_overwrites_different_bytes() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("manifest.json");
        commit_certified_bytes(&output, b"certified").unwrap();
        commit_certified_bytes(&output, b"certified").unwrap();

        let refusal = commit_certified_bytes(&output, b"different").unwrap_err();
        assert_eq!(refusal.stage, ProductSurfaceCompilationStage::Commit);
        assert_eq!(refusal.code, UncompilableEditCode::OutputCommitFailed);
        assert!(!refusal.retryable);
        assert_eq!(std::fs::read(output).unwrap(), b"certified");
    }
}
