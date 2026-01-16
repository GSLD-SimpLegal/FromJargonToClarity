# FromJargonToClarity
Code Base and Dataset for our manuscript titled "From Jargon to Clarity: Bridging Understanding Through Graded Simplification of Legal Data". 

## Dataset Structure

### Data/AllGenerationData
Contains all model-generated legal text simplifications across different grade levels:

- **Basic Grade**  
  `Data/AllGenerationData/Basic_grade_generatedSimplifications.xlsx`

- **Intermediate Grade**  
  `Data/AllGenerationData/Intermediate_grade_generatedSimplifications.xlsx`

- **Skilled Grade**  
  `Data/AllGenerationData/Skilled_grade_generatedSimplifications.xlsx`

### Data/SimpLegal-Pref
Contains human-vetted gold-standard simplifications for different grade levels.

### Data/SimpLegal-Synth
This synthetic dataset is created by pairing each rejected simplification with a randomly sampled accepted simplification of the same source paragraph. This design increases coverage of diverse failure cases while reducing the risk of overfitting to narrow error patterns.







