//! `MetadataStore` — owns down-meta heap/mmap state and per-feature
//! overrides (INSERT/DELETE-side mutations).
//!
//! Carved out of `VectorIndex` in the 2026-04-25 reorg.

use std::collections::HashMap;
use std::sync::Arc;

use crate::index::types::{DownMetaMmap, FeatureMeta};

#[derive(Clone)]
pub struct MetadataStore {
    /// Per-layer, per-feature output token metadata (heap mode).
    pub down_meta: Vec<Option<Vec<Option<FeatureMeta>>>>,
    /// Mmap'd down_meta.bin (zero-copy mode).
    pub down_meta_mmap: Option<Arc<DownMetaMmap>>,
    /// Down vector overrides — `(layer, feature) → hidden_size f32`.
    pub down_overrides: HashMap<(usize, usize), Vec<f32>>,
    /// Up vector overrides — same shape; written by INSERT.
    pub up_overrides: HashMap<(usize, usize), Vec<f32>>,
    /// Per-slot gate threshold τ — `(layer, feature) → τ`. When present,
    /// the inserted slot's activation becomes `silu(gate·x − τ)·(up·x)`,
    /// killing the weak-positive SiLU tail that otherwise leaks the edit
    /// onto held-out prompts (see the 2026-06-02 amplification diagnostic).
    /// Only meaningful for inserted slots; native neurons are untouched.
    pub gate_thresholds: HashMap<(usize, usize), f32>,
}

impl MetadataStore {
    pub fn empty(num_layers: usize) -> Self {
        Self {
            down_meta: vec![None; num_layers],
            down_meta_mmap: None,
            down_overrides: HashMap::new(),
            up_overrides: HashMap::new(),
            gate_thresholds: HashMap::new(),
        }
    }
}
