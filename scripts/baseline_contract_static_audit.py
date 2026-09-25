"""No inference: inspect source headers, manifests, and three quantized planes."""
import hashlib
import json
from pathlib import Path
import struct
import sys
sys.path.insert(0,'/home/pat/llama.cpp-mtp/gguf-py')
from gguf import GGUFReader
ROOT=Path(__file__).resolve().parents[1]
V=Path('/mnt/models/larql-vindexes/gemma-4-e4b-upstream-20260811-q4k.vindex')
HF=Path('/home/pat/hf-models/gemma-4-E4B-it')
G=Path('/mnt/models/hf-models/gemma4-e4b-gguf/gemma-4-E4B-it-Q4_K_M.gguf')
def sha(b):return hashlib.sha256(b).hexdigest()
metadata={}
for n in ['chat_template.jinja','tokenizer.json','tokenizer_config.json','generation_config.json']:
    a=(HF/n).read_bytes();b=(V/n).read_bytes()
    metadata[n]=dict(source_sha256=sha(a),vindex_sha256=sha(b),identical=a==b)
    assert a==b
with open(HF/'model.safetensors','rb') as f:
    size=struct.unpack('<Q',f.read(8))[0]; header=json.loads(f.read(size))
prefix='model.language_model.'
source={k.removeprefix(prefix):v['shape'] for k,v in header.items() if k.startswith(prefix)}
manifest={}
for n in ['weight_manifest.json','attn_weights_kquant_manifest.json','interleaved_kquant_manifest.json']:
    manifest.update({x['key']:x['shape'] for x in json.loads((V/n).read_text())})
missing={k:v for k,v in source.items() if k not in manifest}
mismatch={k:[s,manifest[k]] for k,s in source.items() if k in manifest and s!=manifest[k]}
extra=sorted(set(manifest)-set(source))
assert missing=={'embed_tokens.weight':[262144,2560]} and not mismatch and not extra
assert (V/'embeddings.bin').stat().st_size==262144*2560*2
reader=GGUFReader(str(G),'r')
tensors={t.name:t for t in reader.tensors}
planes=[]
for manifest_name,filename,key,gkey in [
 ('attn_weights_kquant_manifest.json','attn_weights_kquant.bin','layers.0.self_attn.q_proj.weight','blk.0.attn_q.weight'),
 ('interleaved_kquant_manifest.json','interleaved_kquant.bin','layers.0.mlp.gate_proj.weight','blk.0.ffn_gate.weight'),
 ('interleaved_kquant_manifest.json','interleaved_kquant.bin','layers.0.mlp.down_proj.weight','blk.0.ffn_down.weight')]:
    m=next(x for x in json.loads((V/manifest_name).read_text()) if x['key']==key)
    with open(V/filename,'rb') as f:f.seek(m['offset']);a=f.read(m['length'])
    t=tensors[gkey];b=t.data.tobytes()
    planes.append(dict(vindex_key=key,gguf_key=gkey,vindex_format=m['format'],gguf_type=t.tensor_type.name,
        vindex_sha256=sha(a),gguf_sha256=sha(b),encoded_bytes_equal=a==b))
report=dict(model_forwards_executed=0,metadata=metadata,source_language_tensors=len(source),manifest_tensors=len(manifest),
    separate_embedding=missing,embedding_bytes=(V/'embeddings.bin').stat().st_size,shape_mismatches=mismatch,extra_manifest_keys=extra,
    quantized_plane_comparison=planes,gguf_metadata={k:v.contents() for k,v in reader.fields.items()
       if k.startswith('general.') and ('name' in k or 'repo_url' in k or 'architecture' in k or 'quantized_by' in k or 'file_type' in k)},
    limitation='Structural coverage and sampled encoded-weight comparison; not numerical forward validation.')
p=ROOT/'docs/research-data/baseline-contract-audit-v1/static-audit.json'
p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
