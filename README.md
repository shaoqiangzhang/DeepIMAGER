# DeepIMAGER
A supervised deep-learning model for predicting cell-specific gene regulatory networks from scRNA-seq and ChIP-seq data.


Code is tested using Python 3.8

## Runtime environment: 

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

### #4  Command for each cell type:

python3.8 generate_input_realdata.py -out_dir bonemarrow_representation -expr_file ../data_evaluation/bonemarrow/bone_marrow_cell.h5 -pairs_for_predict_file ../data_evaluation/bonemarrow/gold_standard_for_TFdivide -geneName_map_file ../data_evaluation/bonemarrow/sc_gene_list.txt -flag_load_from_h5 True -flag_load_split_batch_pos True -TF_divide_pos_file ../data_evaluation/bonemarrow/whole_gold_split_pos -TF_num 13

python3.8 generate_input_realdata.py -out_dir dendritic_representation -expr_file ../data_evaluation/dendritic/dendritic_cell.h5 -pairs_for_predict_file ../data_evaluation/dendritic/gold_standard_dendritic_whole.txt -geneName_map_file ../data_evaluation/dendritic/sc_gene_list.txt -flag_load_from_h5 True -flag_load_split_batch_pos True -TF_divide_pos_file ../data_evaluation/dendritic/dendritic_divideTF_pos -TF_num 16

python3.8 generate_input_realdata.py -out_dir hESC_representation -expr_file ../data_evaluation/single_cell_type/hESC/ExpressionData.csv -pairs_for_predict_file ../data_evaluation/single_cell_type/training_pairshESC.txt -geneName_map_file ../data_evaluation/single_cell_type/hESC_geneName_map.txt -flag_load_from_h5 False -flag_load_split_batch_pos True -TF_divide_pos_file ../data_evaluation/single_cell_type/training_pairshESC.txtTF_divide_pos.txt -TF_num 18 -TF_order_random True

python3.8 generate_input_realdata.py -out_dir mESC_2_representation -expr_file ../data_evaluation/single_cell_type/mESC/ExpressionData.csv -pairs_for_predict_file ../data_evaluation/single_cell_type/training_pairsmESC.txt -geneName_map_file ../data_evaluation/single_cell_type/mESC_geneName_map.txt -flag_load_from_h5 False -flag_load_split_batch_pos True -TF_divide_pos_file ../data_evaluation/single_cell_type/training_pairsmESC.txtTF_divide_pos.txt -TF_num 18 -TF_order_random True

 python3.8 generate_input_realdata.py -out_dir mHSC_GM_representation -expr_file ../data_evaluation/single_cell_type/mHSC-GM/ExpressionData.csv -pairs_for_predict_file ../data_evaluation/single_cell_type/training_pairsmHSC_GM.txt -geneName_map_file ../data_evaluation/single_cell_type/mHSC_GM_geneName_map.txt -flag_load_from_h5 False -flag_load_split_batch_pos True -TF_divide_pos_file ../data_evaluation/single_cell_type/training_pairsmHSC_GM.txtTF_divide_pos.txt -TF_num 18 -TF_order_random True

python3.8 generate_input_realdata.py -out_dir mHSC_L_representation -expr_file ../data_evaluation/single_cell_type/mHSC-L/ExpressionData.csv -pairs_for_predict_file ../data_evaluation/single_cell_type/training_pairsmHSC_L.txt -geneName_map_file ../data_evaluation/single_cell_type/mHSC_L_geneName_map.txt -flag_load_from_h5 False -flag_load_split_batch_pos True -TF_divide_pos_file ../data_evaluation/single_cell_type/training_pairsmHSC_L.txtTF_divide_pos.txt -TF_num 18 -TF_order_random True

## STEP 2: TF-aware three-fold Cross-validation for DeepIMAGER
 
 Code: DeepIMAGER.py

### #1 Input: the output of the STEP 1.

### #2 Parameters:

num_batches: Since in STEP 1, we divide training pairs by TFs, and representation for one TF is included in one batch. Here the num_batches should be the number of TF or the number of x file

data_path: The path that includes x file, y file and z file, which is generated in the last step.

output_dir: Indicate the path for output.

cross_validation_fold_divide_file: A file that indicate how to divide the x file into three-fold. .

to_predict: True or False. Default is False, then the code will do cross-validation evaluation. If set to True, we need to indicate weight_path for a trained model and the code will do prediction based on the trained model.

weight_path: The path for a trained model.

### #3 Command example:

python3.8 DeepIMAGER.py -num_batches 13 -data_path bonemarrow_representation/version11/ -output_dir boneMarrow -cross_validation_fold_divide_file cross_validation_fold_divide.txt

python3.8 DeepIMAGER.py -num_batches 18 -data_path mHSC_L_representation/version11/ -output_dir mHSC_L_test -cross_validation_fold_divide_file cross_validation_fold_divide.txt

(You need to modify cross_validation_fold_divide.txt yourself)

## STEP 3: Predict use trained DeepIMAGER

### #1  Input: Trained model from STEP2, Representation for other pairs generated by STEP1.

### #2  DeepIMAGER.py can also predict GRN using a trained model, for example:

python3.8  DeepIMAGER.py -to_predict True -num_batches 18 -data_path  mHSC_L_representation/version12/ -output_dir predict_test/ -weight_path model.h5/dedriticL.h5

### #3  Example model, Using the same dataset for training and prediction, trained by bonemarrow.h5, tested by bonemarrow

python3.8 predict_use_model.py -data_path mDCS_representation/version0/ -num_batches 13 -weight_path model.h5/dendritic.h5 -output_dir predict_mdcs/

### #4  Example model, Using the different dataset for training and prediction, trained by mHSC_L.h5, tested by mESC_2

python3.8 predict_use_model.py -data_path mESC_2_representation/version11/ -num_batches 18 -weight_path mHSC_L_keras_cnn_trained_model_shallow.hdf5 -output_dir train_mHSC_L.test_mESC_2/


