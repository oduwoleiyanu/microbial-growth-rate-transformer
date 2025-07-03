# microbial-growth-rate-transformer

Scripts and data associated with the manuscript: Transformer-based Prediction of Microbial Growth Rates from Genomic Data

## Overview

The code implements a transformer-based model for predicting the minimum doubling time of microbial genomes directly from genomic sequences. This includes:

- Loading the transformer model(https://github.com/Hoarfrost-Lab/LookingGlassv2.0)
- Finetuning the model on Madin et al., dataset ((https://figshare.com/collections/A_synthesis_of_bacterial_and_archaeal_phenotypic_trait_data/4843290/3)
 containing the empirical minimum doubling time and genome id
- Training and evaluating the finetuned model
- Incorporating metadata (e.g., growth temperature)


