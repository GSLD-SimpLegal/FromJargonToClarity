python DPO_Inference.py \
  --model "model name" \
  --peft_path "model peft path(trained dpo model)" \
  --input_file "test file path (.json)" \
  --grade "target grade (skilled|intermediate|basic)" \
  --hf_token hf_xxxxxxxxxxxxxxxxx \
  --output_prefix "output file name"