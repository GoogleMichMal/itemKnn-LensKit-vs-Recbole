### This directory contains our "reproduction" results 
Here's what we did:

1. We adjusted the nDCG@10 calculation for LensKit, since it differed from the RecBole implementation 
2. We used RecBole to split the data in train/test (80/20), converted the splitted sets in a csv and used these to run the LensKit algorithm. This made sure, that both algorithms used exactly the same data.
3. We adjusted the LensKit similarity matrix calculation, since if differed from the RecBole implementation. (just in the file plit_itemknn_modified)

The files listed in this directory don't contain our code. The Code can be found in the "Code" directory. If you are just interested in plotting our result, run the files contained in this directory.
