"""Phase 16 AW seed=13 GPU reproduction (Kaggle script kernel)."""
import os, sys, json, shutil, subprocess, hashlib, datetime, glob, time, pathlib

INPUT = pathlib.Path('/kaggle/input/cara-finsent-phase16-aw-bundle')
WORK  = pathlib.Path('/kaggle/working')
REPO  = WORK / 'repo'
REPO.mkdir(parents=True, exist_ok=True)
for sub in ['data/processed/gold','src/cara_finsent','scripts','results','models','logs']:
    (REPO/sub).mkdir(parents=True, exist_ok=True)

print('=== Stage repo from dataset ===', flush=True)
INPUT_ROOT = pathlib.Path('/kaggle/input')
candidates = sorted(INPUT_ROOT.glob('*'))
print('INPUT_ROOT children:', [c.name for c in candidates], flush=True)
INPUT = next((c for c in candidates if 'phase16' in c.name.lower() or 'aw' in c.name.lower()), candidates[0] if candidates else INPUT_ROOT)
print('Using INPUT =', INPUT, flush=True)
print('INPUT contents:', sorted(p.name for p in INPUT.iterdir()), flush=True)
# Kaggle stored folders as .zip archives (--dir-mode zip). Extract them.
import zipfile
EXTRACT = WORK / 'extracted'
EXTRACT.mkdir(parents=True, exist_ok=True)
for z in INPUT.glob('*.zip'):
    with zipfile.ZipFile(z) as zf:
        zf.extractall(EXTRACT / z.stem)
    print('extracted', z.name, '->', EXTRACT / z.stem, flush=True)
# Kaggle may also auto-extract -> copy directly from INPUT.
SOURCES = [EXTRACT, INPUT]
for ROOT in SOURCES:
    for src in ROOT.rglob('*.csv'):
        shutil.copy2(src, REPO/'data/processed/gold'/src.name)
    for src in ROOT.rglob('cara_finsent/*.py'):
        shutil.copy2(src, REPO/'src/cara_finsent'/src.name)
    sroot = ROOT / 'scripts'
    if sroot.exists():
        for src in sroot.rglob('*.py'):
            shutil.copy2(src, REPO/'scripts'/src.name)
    for src in ROOT.rglob('11f_*.py'):
        shutil.copy2(src, REPO/'scripts'/src.name)
    for src in ROOT.rglob('41_*.py'):
        shutil.copy2(src, REPO/'scripts'/src.name)
print('data:', sorted(p.name for p in (REPO/'data/processed/gold').iterdir()))
print('src :', sorted(p.name for p in (REPO/'src/cara_finsent').iterdir()))
print('scr :', sorted(p.name for p in (REPO/'scripts').iterdir()))

print('=== Pip install ===', flush=True)
# Detect GPU compute capability; preinstalled torch (>=2.6) drops sm_60 (Tesla P100).
# Pin torch 2.4.1+cu121 which still supports sm_60 / sm_70 / sm_75 / sm_80 / sm_86 / sm_90.
try:
    import torch as _t
    _cap = _t.cuda.get_device_capability(0) if _t.cuda.is_available() else (0,0)
except Exception:
    _cap = (0,0)
print('detected cuda capability:', _cap, flush=True)
if _cap and _cap[0] < 7:
    print('Pinning torch==2.4.1+cu121 for sm_60 (P100) compatibility', flush=True)
    subprocess.check_call([sys.executable, '-m', 'pip', '-q', 'install',
                           '--index-url', 'https://download.pytorch.org/whl/cu121',
                           'torch==2.4.1', 'torchvision==0.19.1'])
subprocess.check_call([sys.executable, '-m', 'pip', '-q', 'install',
                       'transformers==4.49.0', 'datasets>=3.0,<4.0',
                       'accelerate>=1.0,<2.0', 'scikit-learn>=1.5,<2.0',
                       'pandas', 'numpy'])

# Re-import torch in a clean subprocess to avoid in-process docstring conflict.
torch_info = subprocess.check_output([sys.executable, '-c',
    'import torch,json;print(json.dumps({"version":torch.__version__,'
    '"cuda":torch.cuda.is_available(),'
    '"name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))'],
    text=True).strip()
ti = json.loads(torch_info)
print('torch', ti['version'], 'cuda?', ti['cuda'], ti['name'], flush=True)

os.chdir(REPO)
sys.path.insert(0, str(REPO/'src'))

env_path = WORK/'phase16_aw_seed13_environment.txt'
with env_path.open('w') as fh:
    fh.write(f'python={sys.version}\n')
    fh.write(f'torch={ti["version"]}\n')
    fh.write(f'cuda_available={ti["cuda"]}\n')
    fh.write(f'gpu_name={ti["name"]}\n')
    fh.write(subprocess.check_output([sys.executable,'-m','pip','freeze'], text=True))
print('env ->', env_path, flush=True)

print('=== Train AW seed=13 ===', flush=True)
log_path = WORK/'phase16_aw_seed13_gpu_train_log.txt'
cmd = [sys.executable, 'scripts/11f_train_finbert_agreement_weighted.py',
       '--seed','13','--weight_schedule','linear',
       '--num_epochs','3','--batch_size','16']
print('CMD:', ' '.join(cmd), flush=True)
t0 = time.time()
with log_path.open('w') as fh:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end='', flush=True)
        fh.write(line)
    rc = proc.wait()
print('returncode', rc, 'elapsed', round(time.time()-t0,1), flush=True)
assert rc == 0, 'training failed'

print('=== Locate val-best AW checkpoint ===', flush=True)
ckpt_root = sorted(glob.glob('models/finbert_agreement_weighted_linear_*'))[-1]
print('ckpt_root', ckpt_root, flush=True)
best_subdir = ckpt_root
for s in sorted(glob.glob(f'{ckpt_root}/checkpoint-*/trainer_state.json')):
    d = json.load(open(s))
    print(s, 'best_metric=', d.get('best_metric'),
          'best_model_checkpoint=', d.get('best_model_checkpoint'), flush=True)
    bmc = d.get('best_model_checkpoint')
    if bmc and os.path.isdir(bmc):
        best_subdir = bmc
print('best_subdir', best_subdir, flush=True)

print('=== Generate val + test predictions via script 41 ===', flush=True)
cmd2 = [sys.executable, 'scripts/41_generate_val_test_predictions.py',
        '--model_family','agreement_weighted','--checkpoint',best_subdir,
        '--seed','13','--batch_size','16']
print('CMD:', ' '.join(cmd2), flush=True)
rc2 = subprocess.call(cmd2)
print('returncode', rc2, flush=True)
assert rc2 == 0

val_csvs  = sorted(glob.glob('results/**/finbert_agreement_weighted_val_predictions_*.csv', recursive=True))
test_csvs = sorted(glob.glob('results/**/finbert_agreement_weighted_test_predictions_*.csv', recursive=True))
summ_csvs = sorted(glob.glob('results/**/finbert_agreement_weighted_summary_*.csv', recursive=True))
print('val ', val_csvs, flush=True)
print('test', test_csvs, flush=True)
print('summ', summ_csvs, flush=True)
assert val_csvs and test_csvs and summ_csvs
shutil.copy2(val_csvs[-1],  WORK/'phase16_aw_seed13_val_predictions.csv')
shutil.copy2(test_csvs[-1], WORK/'phase16_aw_seed13_test_predictions.csv')

import pandas as pd, numpy as np
from sklearn.metrics import accuracy_score, f1_score
test_df = pd.read_csv(WORK/'phase16_aw_seed13_test_predictions.csv')
val_df  = pd.read_csv(WORK/'phase16_aw_seed13_val_predictions.csv')
label_col = 'label' if 'label' in test_df.columns else 'true_label'
pred_col  = 'pred_label' if 'pred_label' in test_df.columns else 'prediction'
y_true_t = test_df[label_col].values; y_pred_t = test_df[pred_col].values
y_true_v = val_df[label_col].values;  y_pred_v = val_df[pred_col].values
test_macro = float(f1_score(y_true_t, y_pred_t, average='macro', zero_division=0))
test_acc   = float(accuracy_score(y_true_t, y_pred_t))
val_macro  = float(f1_score(y_true_v, y_pred_v, average='macro', zero_division=0))
val_acc    = float(accuracy_score(y_true_v, y_pred_v))
print(f'TEST macro_f1={test_macro:.4f} acc={test_acc:.4f}', flush=True)
print(f'VAL  macro_f1={val_macro:.4f} acc={val_acc:.4f}', flush=True)
print('GATE pass (>=0.8817)?', test_macro >= 0.8817, flush=True)

pd.DataFrame([
    {'phase':16,'seed':13,'split':'test','accuracy':test_acc,'macro_f1':test_macro,
     'pass_gate_0p8817': test_macro >= 0.8817},
    {'phase':16,'seed':13,'split':'val','accuracy':val_acc,'macro_f1':val_macro,
     'pass_gate_0p8817': None},
]).to_csv(WORK/'phase16_aw_seed13_metrics.csv', index=False)

def sha256(p):
    h = hashlib.sha256()
    with open(p,'rb') as fh:
        for c in iter(lambda: fh.read(8192), b''): h.update(c)
    return h.hexdigest()

split_path = REPO/'data/processed/gold/latest_gold_phrasebank_split.csv'
summ_row = pd.read_csv(summ_csvs[-1]).iloc[0].to_dict()
manifest = {
    'phase': 16,
    'kaggle_kernel_slug': 'ranafarazahmed/cara-finsent-phase-16-aw-seed-13-gpu-rerun',
    'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'cuda_available': bool(torch.cuda.is_available()),
    'seed': 13,
    'base_model': 'ProsusAI/finbert',
    'split_file': str(split_path),
    'split_sha256': sha256(split_path),
    'label_remap': [1, 2, 0],
    'native_id2label': {0:'positive', 1:'negative', 2:'neutral'},
    'agreement_weight_schedule': 'linear',
    'epochs': 3, 'batch_size': 16, 'max_length': 128,
    'learning_rate': 2e-5, 'warmup_steps': 100,
    'selected_checkpoint_rule': 'load_best_model_at_end on val macro_f1',
    'best_checkpoint': best_subdir,
    'val_macro_f1': val_macro, 'val_accuracy': val_acc,
    'test_macro_f1': test_macro, 'test_accuracy': test_acc,
    'pass_gate_0p8817': bool(test_macro >= 0.8817),
    'training_summary_csv': summ_csvs[-1],
    'training_summary_row': {k:(float(v) if isinstance(v,(int,float,np.floating,np.integer)) else str(v))
                             for k,v in summ_row.items()},
    'generated_at_utc': datetime.datetime.utcnow().isoformat()+'Z',
}
with open(WORK/'phase16_aw_seed13_manifest.json','w') as fh:
    json.dump(manifest, fh, indent=2, default=str)

print('=== Final artefacts ===', flush=True)
for p in sorted(WORK.glob('phase16_aw_seed13_*')):
    print(p, p.stat().st_size, flush=True)
