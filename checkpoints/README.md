# Checkpoints

No trained checkpoints are included in this repository.

When you run the training scripts, checkpoints are saved here by default:

```text
checkpoints/
  clip_finetuned_10_epochs.pth
  clip_finetuned_50_epochs.pth
  clip_finetuned_50_epochs_low_lr.pth
  pla_clip.pth
```

These files are ignored by Git. The scripts start from the public Hugging Face
model `openai/clip-vit-base-patch16`.

TODO: Add public checkpoint links only if trained weights are released later.
