# Gene_Signature_Discovery


## Description: 
   - Based on microRNA, mRNA quantification data and other clinical data of the patients with a breast cancer subtype, we want to build a model by machine learning to predict 5-year survival rate and tumor stage. If those labels cannot be predicted accurately, we want to do a data mining by clustering to find out if the data can be classified into other subclasses and the genes that distinguish the subclasses. Furthermore, we want to find out the biological pathways those genes are involved in and make hypotheses about microRNA- mRNA pair signature.

    - Writen report can be found here: https://github.com/timothyfisherphd/Gene_Signature_Discovery/blob/main/Gene_Signature_Discovery.pdf
## Usage
1. Open Terminal or Command Window to excute following code: 
./run.sh

2. Using output of gene list then run gene names through /Pathway_Analysis.  Open Python3 and run script below 
./GO_enrichment_analysis.py

## Output
- Shown in written summary "./Pathway_Analysis/go_annotation_results.docx" for all clusters or indivudal in './Pathway_Analysis/go_cluster_results'
