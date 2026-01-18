python train_DPO.py \
  --model_name google/gemma-2-9b \
  --data_path Data/SimpLegal-Synth/PreferenceData_Skilled.json \
  --quant_bits 4 \
  --output_dir dpo-gemma-legal
