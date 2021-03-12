#!/usr/bin/env python
# coding: utf-8

# In[6]:


##  Retrieving GO annotations and import packages ###
import Bio.UniProt.GOA as GOA
from goatools import obo_parser # Import the OBO parser from GOATools
import wget
import os
from ftplib import FTP

# Downloading the obo file

go_obo_url = 'http://purl.obolibrary.org/obo/go/go-basic.obo'
data_folder = os.getcwd() + '/data' # Creating a new directory named data to store obo file.


# Check if the file exists already
if(not os.path.isfile(data_folder+'/go-basic.obo')):
    go_obo = wget.download(go_obo_url, data_folder+'/go-basic.obo')
else:
    go_obo = data_folder+'/go-basic.obo' # Store the go obo file in the variable go_obo
    
# To create dictionary of GO terms.
go = obo_parser.GODag(go_obo)


# In[7]:


# Download the reduced GAF file containing gene association of only humans. #

'''
Human UniProt-GOA GAF file
ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/goa_human.gaf.gz
'''

human_url = '/pub/databases/GO/goa/HUMAN/goa_human.gaf.gz'
human_fn = human_url.split('/')[-1]

# Check if the file exists already
human_gaf = os.path.join(data_folder, human_fn)
if(not os.path.isfile(human_gaf)):
    # Login to FTP server
    ebi_ftp = FTP('ftp.ebi.ac.uk')
    ebi_ftp.login() # To loing in as random person... Ghose
    
    # Download
    with open(human_gaf,'wb') as human_fp:
        ebi_ftp.retrbinary('RETR {}'.format(human_url), human_fp.write)
        
    # Logout from FTP server
    ebi_ftp.quit()


# In[8]:


import gzip
# File is a gunzip file, so we need to open it in this way
with gzip.open(human_gaf, 'rt') as human_gaf_fp:
    human_funcs = {}  # Initialize the dictionary of functions
    
    # Iterate on each function using Bio.UniProt.GOA library.
    for entry in GOA.gafiterator(human_gaf_fp):
        uniprot_id = entry.pop('DB_Object_ID')
        human_funcs[uniprot_id] = entry


# In[9]:


### GO enrichment analysis ###
from goatools.go_enrichment import GOEnrichmentStudy

# The reference population from the observed Human GOA file. 
pop = human_funcs.keys()

# The Swiss-Prot gene IDs from mygene_code.py results
study = ['P03905','P00846','P03891','P00395','P00403','P00156']#cluster5_c0c1

# To create a dictionary of genes with their 
# UniProt ID as a key and their set of GO annotations as the values.
assoc = {}
for x in human_funcs:
    if x not in assoc:
        assoc[x] = set()
    assoc[x].add(str(human_funcs[x]['GO_ID']))

## The GO terms that are most significantly enriched. 
# Methods for GOEnrichmentStudy
methods = ["bonferroni", "sidak", "holm", "fdr"]




# Running GOEnrichmentStudy
g = GOEnrichmentStudy(pop, assoc, go,
                         propagate_counts=True,
                         alpha=0.05,
                         methods=methods)
g_res = g.run_study(study)
    


# In[10]:


g.print_results(g_res, min_ratio=None, pval=0.05)


# In[ ]:




