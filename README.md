# FromJargonToClarity: 
Code Base and Dataset for our manuscript titled "From Jargon to Clarity: Bridging Understanding Through Graded Simplification of Legal Data". 

## Dataset Structure

#### GSLD/Data/AllGenerationData
Contains all model-generated legal text simplifications across different grade levels:

| Grade Level   | File Path                                                         |
|---------------|-------------------------------------------------------------------|
| Basic         | AllGenerationData/Basic_grade_generatedSimplifications.xlsx |
| Intermediate  | AllGenerationData/Intermediate_grade_generatedSimplifications.xlsx |
| Skilled       | AllGenerationData/Skilled_grade_generatedSimplifications.xlsx |


#### GSLD/Data/SimpLegal-Pref
Contains human-vetted gold-standard simplifications for different grade levels.

#### GSLD/Data/SimpLegal-Synth
This synthetic dataset is constructed by pairing each rejected simplification with a randomly sampled accepted simplification from the same source paragraph. This strategy increases coverage of diverse failure cases while reducing the risk of overfitting to narrow error patterns. The dataset includes both training and test preference data for all three grades, Skilled, Intermediate, and Basic. The training files are provided in a compressed format.

## Scripts Structure

#### GSLD/Scripts

This directory contains all training, inference, and generation pipelines used for graded legal text simplification experiments.

| Script Name                  | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| DPO_trainer.py              | Trains a graded legal simplification model using Direct Preference Optimization (DPO) with LoRA-based parameter-efficient fine-tuning |
| DPO_Inference.py            | Runs inference using a trained DPO-aligned model to generate simplified legal text |
| TextSimplification_ZS_IGIP.py| Implements zero-shot (ZS) and Iterative Guided Improvement Prompting (IGIP) strategies for legal text simplification |


### Shell Scripts
| Script Name               | Description                                      |
|---------------------------|--------------------------------------------------|
| run_DPOtrain.sh           | Executes DPO training for graded legal simplification |
| run_DPOInference.sh       | Runs batch inference using trained DPO model  |
| run_TextSimplification.sh | Executes zero-shot and IGIP-based simplification experiments |

## Recommended Execution Workflow

The typical experimental workflow for graded legal text simplification follows three main stages:

1. **Train the DPO Model**  
   Use `DPO_trainer.py` (or `run_DPOtrain.sh`) to fine-tune the base language model with preference data for a target simplification grade.

2. **Run DPO-Based Inference**  
   Use `DPO_Inference.py` (or `run_DPOInference.sh`) to generate simplified outputs using the Direct Preference Optimization.

3. **Run Prompt-Based Baselines**  
   Use `TextSimplification_ZS_IGIP.py` (or `run_TextSimplification.sh`) to generate zero-shot and IGIP baseline outputs for comparison.

This workflow enables direct comparison between prompt-based methods and preference-optimized models.

---

