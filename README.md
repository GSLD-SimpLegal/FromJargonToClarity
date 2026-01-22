# FromJargonToClarity: 
Code Base and Dataset for our manuscript titled "From Jargon to Clarity: Bridging Understanding Through Graded Simplification of Legal Data". 

## Dataset Structure

#### GSLD/Data/AllGenerationData
Contains all model-generated legal text simplifications across different grade levels:

| Grade Level   | File Path                                                         |
|---------------|-------------------------------------------------------------------|
| Basic         | GSLD/Data/AllGenerationData/Basic_grade_generatedSimplifications.xlsx |
| Intermediate  | GSLD/Data/AllGenerationData/Intermediate_grade_generatedSimplifications.xlsx |
| Skilled       | GSLD/Data/AllGenerationData/Skilled_grade_generatedSimplifications.xlsx |


#### GSLD/Data/SimpLegal-Pref
Contains human-vetted gold-standard simplifications for different grade levels.

#### GSLD/Data/SimpLegal-Synth
This synthetic dataset is created by pairing each rejected simplification with a randomly sampled accepted simplification of the same source paragraph. This design increases coverage of diverse failure cases while reducing the risk of overfitting to narrow error patterns.

## Scripts Structure

#### GSLD/Scripts

This directory contains all training, inference, and generation pipelines used for graded legal text simplification experiments.

| Script Name                  | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| DPO_trainer.py              | Trains a graded legal simplification model using Direct Preference Optimization (DPO) with LoRA-based parameter-efficient fine-tuning |
| DPO_Inference.py            | Runs inference using a trained DPO-aligned PEFT adapter to generate simplified legal text |
| TextSimplification_ZS_IGIP.py| Implements zero-shot (ZS) and Iterative Guided Improvement Prompting (IGIP) strategies for legal text simplification |


### Shell Scripts
| Script Name               | Description                                      |
|---------------------------|--------------------------------------------------|
| run_TextSimplification.sh | Executes zero-shot and IGIP-based simplification experiments |
| run_DPOInference.sh       | Runs batch inference using trained DPO adapters  |
| run_DPOtrain.sh           | Executes DPO training for graded legal simplification |



