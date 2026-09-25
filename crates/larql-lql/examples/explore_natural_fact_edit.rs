//! Interactive exploration worker: one model load, at most 128 full-prefix forwards.
use larql_inference::{
    forward::LayerHook,
    hidden_to_raw_logits,
    vindex::{predict_kquant_hidden_with_ffn_hooked, WalkFfn},
    InferenceWeights,
};
use ndarray::{Array2, Axis};
use serde::Deserialize;
use serde_json::json;
use std::{
    error::Error,
    io::{BufRead, Write},
    path::Path,
};
type E = Box<dyn Error>;
#[derive(Deserialize)]
struct Request {
    ids: Vec<u32>,
    layer: Option<usize>,
    position: Option<usize>,
    delta: Option<Vec<f32>>,
    #[serde(default)]
    gain: f32,
}
struct Hook {
    layer: Option<usize>,
    position: Option<usize>,
    delta: Option<Vec<f32>>,
    gain: f32,
    captures: Vec<(usize, Vec<f32>)>,
    applied: bool,
}
impl LayerHook for Hook {
    fn on_post_layer(&mut self, layer: usize, h: &mut Array2<f32>) {
        if self.layer == Some(layer) {
            if let (Some(position), Some(delta)) = (self.position, self.delta.as_ref()) {
                if position < h.nrows() {
                    for (i, v) in h.row_mut(position).iter_mut().enumerate() {
                        *v += self.gain * delta[i];
                    }
                    self.applied = true;
                }
            }
        }
        if [31, 41].contains(&layer) {
            self.captures.push((layer, h.row(h.nrows() - 1).to_vec()));
        }
    }
}
fn emit(v: serde_json::Value) -> Result<(), E> {
    let mut o = std::io::stdout().lock();
    write!(o, "EXP\t")?;
    serde_json::to_writer(&mut o, &v)?;
    writeln!(o)?;
    o.flush()?;
    Ok(())
}
fn main() -> Result<(), E> {
    let a: Vec<_> = std::env::args().skip(1).collect();
    if a.len() != 1 {
        return Err("usage: explore_natural_fact_edit VINDEX".into());
    }
    let path = Path::new(&a[0]);
    let cfg = larql_vindex::load_vindex_config(path)?;
    if cfg.hidden_size != 2560 || cfg.num_layers != 42 || cfg.vocab_size != 262144 {
        return Err("unexpected model".into());
    }
    let mut w = InferenceWeights::load(path, &cfg, &mut larql_vindex::SilentLoadCallbacks)?;
    let (weights, index) = w.as_quantised_parts_mut().ok_or("Q4K required")?;
    emit(json!({"type":"ready","max_forwards":128}))?;
    for (ordinal, line) in std::io::stdin().lock().lines().enumerate() {
        if ordinal >= 128 {
            return Err("session forward budget exhausted".into());
        }
        let r: Request = serde_json::from_str(&line?)?;
        if r.ids.is_empty()
            || r.ids.len() > 256
            || r.ids.iter().any(|&v| v >= 262144)
            || !r.gain.is_finite()
        {
            return Err("invalid request".into());
        }
        if let Some(ref d) = r.delta {
            if d.len() != 2560
                || d.iter().any(|v| !v.is_finite())
                || !matches!(r.layer, Some(31 | 41))
                || r.position.is_none_or(|p| p >= r.ids.len())
            {
                return Err("invalid intervention".into());
            }
        } else if r.layer.is_some() || r.position.is_some() {
            return Err("incomplete intervention".into());
        }
        let mut hook = Hook {
            layer: r.layer,
            position: r.position,
            delta: r.delta,
            gain: r.gain,
            captures: vec![],
            applied: false,
        };
        let wr: &larql_models::ModelWeights = unsafe { &*(weights as *const _) };
        let ffn = WalkFfn::new_unlimited(wr, index);
        let h = predict_kquant_hidden_with_ffn_hooked(weights, &r.ids, index, &ffn, &mut hook)?;
        let last = h.row(h.nrows() - 1).to_owned().insert_axis(Axis(0));
        let logits = hidden_to_raw_logits(weights, &last);
        if logits.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite logits".into());
        }
        emit(
            json!({"type":"forward","ordinal":ordinal,"logits":logits,"captures":hook.captures,"applied":hook.applied}),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn intervention_stays_at_its_original_position_as_prefix_grows() {
        let mut hook = Hook {
            layer: Some(31),
            position: Some(1),
            delta: Some(vec![2.0, -1.0]),
            gain: 0.5,
            captures: vec![],
            applied: false,
        };
        let mut h = Array2::zeros((4, 2));
        hook.on_post_layer(30, &mut h);
        assert_eq!(h.sum(), 0.0);
        hook.on_post_layer(31, &mut h);
        assert_eq!(h.row(1).to_vec(), vec![1.0, -0.5]);
        assert_eq!(h.row(3).to_vec(), vec![0.0, 0.0]);
        assert!(hook.applied);
    }
}
