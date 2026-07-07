# Evaluation Instructions - Get Full Metrics Table

## Critical Bug Fixed ✅

**Problem**: Your evaluation was running with random uninitialized weights because:
1. The code used `safetensors.load_file()` on PyTorch `.pth` files (wrong loader)
2. Errors were silently caught and logged, allowing execution to continue

**Solution**: 
- Replaced `load_file` with `torch.load` in `runner.py` and `full_model.py`
- Added proper error handling that crashes loudly if checkpoint loading fails
- Added support for nested checkpoint structures

## Now Run Evaluation on All 4 Datasets

### Step 1: Pull the fixes on your remote server

```bash
ssh ariyan@10.8.1.109
cd /home/ariyan/UCOD-DPL-2
git pull origin main
```

You should see these updates:
- `engine/runner/runner.py` - Fixed checkpoint loading
- `models/modules/full_model.py` - Fixed checkpoint loading  
- `eval_all_datasets.sh` - New evaluation script

### Step 2: Verify your checkpoint exists

```bash
# Check available checkpoints
ls -lh work_dir/uscod/ADF_dinov2/UCOD-ADF/ckp/

# You should see epoch25.pth or similar
```

### Step 3: Update the script if needed

If your checkpoint has a different name (e.g., `best.pth` instead of `epoch25.pth`), edit the script:

```bash
nano eval_all_datasets.sh
# Change this line:
# CHECKPOINT="work_dir/uscod/ADF_dinov2/UCOD-ADF/ckp/epoch25.pth"
# To match your actual checkpoint filename
```

### Step 4: Run evaluation on all 4 datasets

```bash
./eval_all_datasets.sh
```

This will automatically evaluate on:
1. **CHAMELEON** (76 images)
2. **TE-CAMO** (250 images) - CAMO-Test  
3. **TE-COD10K** (2026 images) - COD10K-Test
4. **NC4K** (4121 images)

### Step 5: Extract metrics from logs

The output will show 5 metrics for each dataset:
- **S_m** ↑ (S-measure)
- **F_β^w** ↑ (Weighted F-measure)
- **F_β^m** ↑ (Mean F-measure)
- **E_φ** ↑ (E-measure)
- **M** ↓ (MAE - Mean Absolute Error)

Look for lines like:
```
Sm: 0.XXX | Fw: 0.XXX | Fm: 0.XXX | Em: 0.XXX | MAE: 0.XXX
```

### Alternative: Run manually one by one

If the script has issues, run each evaluation individually:

```bash
# CHAMELEON
python eval.py --config configs/uscod/ADF_dinov2.py \
  --load_from work_dir/uscod/ADF_dinov2/UCOD-ADF/ckp/epoch25.pth \
  --opts dataset_cfg.valset_cfg.DATASET CHAMELEON

# TE-CAMO (CAMO-Test)
python eval.py --config configs/uscod/ADF_dinov2.py \
  --load_from work_dir/uscod/ADF_dinov2/UCOD-ADF/ckp/epoch25.pth \
  --opts dataset_cfg.valset_cfg.DATASET TE-CAMO

# TE-COD10K (COD10K-Test)
python eval.py --config configs/uscod/ADF_dinov2.py \
  --load_from work_dir/uscod/ADF_dinov2/UCOD-ADF/ckp/epoch25.pth \
  --opts dataset_cfg.valset_cfg.DATASET TE-COD10K

# NC4K
python eval.py --config configs/uscod/ADF_dinov2.py \
  --load_from work_dir/uscod/ADF_dinov2/UCOD-ADF/ckp/epoch25.pth \
  --opts dataset_cfg.valset_cfg.DATASET NC4K
```

### Expected Output Format

For your paper table, you need these columns:

| Dataset | S_m ↑ | F_β^w ↑ | F_β^m ↑ | E_φ ↑ | M ↓ |
|---------|-------|---------|---------|-------|-----|
| CHAMELEON (76) | ? | ? | ? | ? | ? |
| CAMO-Test (250) | ? | ? | ? | ? | ? |
| COD10K-Test (2026) | ? | ? | ? | ? | ? |
| NC4K (4121) | ? | ? | ? | ? | ? |

The evaluation script will fill in all the "?" values!

## Troubleshooting

### If checkpoint loading still fails:

Check the checkpoint structure:
```bash
python3 -c "
import torch
ckpt = torch.load('work_dir/uscod/ADF_dinov2/UCOD-ADF/ckp/epoch25.pth', map_location='cpu', weights_only=False)
print('Type:', type(ckpt))
if isinstance(ckpt, dict):
    print('Keys:', list(ckpt.keys()))
"
```

If it shows nested keys like `{'model_state_dict': ...}`, the fix should handle it automatically.

### If datasets are not found:

Verify dataset paths:
```bash
ls datasets/RefCOD/
# Should show: CHAMELEON  NC4K  TE-CAMO  TE-COD10K  TR-CAMO  TR-COD10K

# Check images exist:
find datasets/RefCOD/TE-CAMO/im -type f | wc -l  # Should be ~250
find datasets/RefCOD/TE-COD10K/im -type f | wc -l  # Should be ~2026
```

## Summary

✅ Fixed critical bug (safetensors → torch.load)
✅ Added proper error handling  
✅ Created automated evaluation script
✅ Pushed all changes to GitHub

**Next**: Run `./eval_all_datasets.sh` on your server and collect the metrics!
