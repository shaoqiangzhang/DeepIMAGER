# DeepIMAGER
a suoervised deep learning model for predicting cell-type-specific gene regulatory networks from scRNA-seq and ChIP-seq.


Code is tested using Python 3.8

Runtime environment: 

tensorflow              2.4.1

tensorflow-gpu          2.4.1 

numba                   0.51.2

numpy                   1.23.5

scikit-learn             1.2.2 

## STEP 1: Generate input for DeepIMAGER

### #1  Code: generate_input_realdata.py

### #2  Input: Gene expression profile and the benchmark, etc.

### #3  Parameters:
out_dir: Indicate the path for output.

expr_file: The file of the gene expression profile. Can be h5 or csv file, the format please refer the example data.

pairs_for_predict_file: The file of the training gene pairs and their labels.

geneName_map_file: The file to map the name of gene in expr_file to the pairs_for_predict_file

flag_load_from_h5: Is the expr_file is a h5 file. True or False.

flag_load_split_batch_pos: Is there a file that indicate the position in pairs_for_predict_file to divide pairs into different TFs.

TF_divide_pos_file: File that indicate the position in pairs_for_predict_file to divide pairs into different TFs.

TF_num: To generate representation for this number of TFs. Should be a integer that equal or samller than the number of TFs in the pairs_for_predict_file.

TF_order_random: If the TF_num samller than the number of TFs in the pairs_for_predict_file, we need to indicate TF_order_random, if TF_order_random=True, then the code will generate representation for randomly selected TF_num TFs.

top_or_random: Decide how to select the neighbor images. Can be set as "top_cov","top_corr", "random"

get_abs: Select neighbor images by considering top value or top absolute value.
