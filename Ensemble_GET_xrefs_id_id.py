#!/usr/bin/python3
import requests, sys
 
server = "https://rest.ensembl.org"
ext = "/xrefs/id/ENSG00000140937.12?"
 
r = requests.get(server+ext, headers={ "Content-Type" : "application/json"})
 
if not r.ok:
  r.raise_for_status()
  sys.exit()
 
decoded = r.json()
print(repr(decoded))