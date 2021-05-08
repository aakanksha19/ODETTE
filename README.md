# ODETTE
Open Domain EvenT Trigger Extractor (ODETTE) using ADA (Adversarial Domain Adaptation)

This repository contains the code for the ACL 2020 paper: "[Towards Open Domain Event Trigger Identification using Adversarial Domain Adaptation](https://www.aclweb.org/anthology/2020.acl-main.681.pdf)".

## Code requirements
- python 3.6 
- library requirements in requirements.txt
- Stanford CoreNLP models. After downloading the models, please edit the “models_dir” variable in line 332 of event_dataset.py to your model folder. I have left a TODO note there so that it’s easy to search.

## Running the code
### Data preprocessing
This code requires two datasets (source and target). The source dataset should be in this format:

folder/<br/>
...file1.tsv<br/>
...file2.tsv<br/>
...<br/>
...filen.tsv<br/>
...train_ids.txt<br/>
...test_ids.txt<br/>
...dev_ids.txt<br/>
 
Each file (file1...filen) should be in the same tsv format as David Bamman's [Litbank data](https://github.com/dbamman/litbank/tree/master/events/tsv), i.e. a 2-column tsv where the first column of every line contains a word and the second column contains a tag indicating whether it is an event or not (EVENT/O). Sentences in the original document should be separated by a newline in the tsv. A sample file would look like this:
 
Today, O<br/>
he, O<br/>
joined, EVENT<br/>
Google, O

He, O<br/>
is, O<br/>
an, O<br/>
engineer, O<br/>
 
The train_ids, dev_ids and test_ids simply consist of lists of filenames (WITHOUT .tsv extension) which belong to the train, dev and test set respectively. A sample train_ids file would look like this:<br/>
file1<br/>
file2<br/>
file3<br/>
 
The target dataset should be in this format:<br/>
target_folder<br/>
...file1.txt<br/>
...file2.txt<br/>
...<br/>
...filen.txt<br/>
 
Each file should contain a single document with every tokenized sentence in the document appearing on a new line.
 
Our ACL 2020 paper used TimeBank and LitBank as source/target in various experimental settings. LitBank is already available in the required format and our work used the same splits. For TimeBank, you will need to access the corpus from the LDC and preprocess it to convert it into the source format. The timebank_processing folder contains the preprocessing script (timebank_preprocessor.py) in which you just need to change the root_dir and out_dir variables to point to your TimeBank corpus folder and the output folder you want the processed files to be stored in. The train/dev/test splits for TimeBank used in the paper (train_ids.txt, dev_ids.txt, test_ids.txt) are also included in the same folder. This should reduce the possibility of any discrepancies arising due to differences in data processing.

### Running the system
Instructions to run the system differ slightly based on the type of model you want to test (BERT vs non-BERT)

#### Running non-BERT models:
```>>> python tester.py --data_dir <SOURCE_FOLDER> --train_file <TRAIN_FILENAME> --dev_file <DEV_FILENAME> --test_file <TEST_FILENAME> --model <MODEL_TYPE> --model_path <MODEL_SAVE_PATH> --do_train --do_eval``` (No adaptation)
 
```>>> python da_tester_new.py --data_dir <SOURCE_FOLDER> --target_dir <TARGET_FOLDER> --train_file <TRAIN_FILENAME> --dev_file <DEV_FILENAME> --test_file <TEST_FILENAME> --model <MODEL_TYPE> --model_path <MODEL_SAVE_PATH> --adv_coeff <LAMBDA_VALUE> --do_train --do_eval``` (With adaptation)
 
These commands train the specified non-BERT model on the labeled source dataset provided via --data_dir, and evaluate it on the source dataset. To include adaptation, specify the target dataset via --target_dir as in the second command (reminder: target dataset only consists of raw text files). To evaluate the performance of a trained model on the target dataset, re-run the same command without the --do_train flag, but provide the target dataset in the --data_dir flag. Evaluation is always run on the dataset provided via --data_dir. In the target evaluation setting (no --do_train and providing target dataset in --data_dir) for da_tester_new.py, it does not matter which folder is provided as --target_dir as long as it is a valid path. This dataset will be loaded but never used.
 
<b>Important notes about flags:</b>
- --model takes one of {word, delex, pos, bert-bilstm, bert-mlp}
- --seed can be used to specify a random seed
- --emb_size can be used to change word embedding size (default is 100)
- --emb_file can be used to provide the path to a pretrained word embedding file
- --bidir can be used to specify whether bidirectionality is used. By default, the model is bidirectional, but if the bidir flag is used, the model is unidirectional. (Sorry this is slightly unintuitive!)
- For all other flags, the values set by default are the ones used in the paper
 
#### Running BERT models:
<b>Step 1: Extracting and dumping BERT embeddings for dataset:</b><br/>
```python dump_bert_features.py --data_dir <SOURCE_FOLDER> --target_dir <TARGET_FOLDER> --train_file <TRAIN_FILENAME> --dev_file <DEV_FILENAME> --test_file <TEST_FILENAME> --save_path <SAVE_DIR> --suffix <DATASET_NAME> --model bert```
 
This code will create BERT embeddings for all sentences in your source and target datasets. <SAVE_DIR> is the directory where the created BERT embeddings will be dumped as pytorch tensors. <DATASET_NAME> is simply an identifier used to label the dumped BERT feature files. For a non domain adapted setting, simply ignore the --target_dir flag.<br/>
<b>Important note:</b> Since the folder provided in --target_dir only contains raw texts, you need to run this step twice in a DA setting. In the first run, --data_dir contains source and --target_dir contains unlabeled target data. In the second run, --data_dir contains labeled target data and --target_dir can contain any valid folder of raw texts. The second run is needed in order to create a labeled target dataset dump to evaluate the adapted model on the target data.
 
<b>Step 2: Train a domain-adapted model:</b><br/>
```python da_tester_new.py --data_dir <SOURCE_FOLDER> --target_dir <TARGET_FOLDER> --train_file <TRAIN_FILENAME> --dev_file <DEV_FILENAME> --test_file <TEST_FILENAME> --save_path <SAVE_DIR> --suffix <DATASET_NAME> --model bert-bilstm --model_path <MODEL_SAVE_PATH>  --adv_coeff <LAMBDA_VALUE> --do_train --do_eval``` (No adaptation)
 
```python da_tester_new.py --data_dir <SOURCE_FOLDER> --target_dir <TARGET_FOLDER> --train_file <TRAIN_FILENAME> --dev_file <DEV_FILENAME> --test_file <TEST_FILENAME> --save_path <SAVE_DIR> --suffix <DATASET_NAME> --model bert-bilstm --model_path <MODEL_SAVE_PATH>  --adv_coeff <LAMBDA_VALUE> --do_train --do_eval``` (With adaptation)
 
These commands will train a model with BERT features. All instructions and flags are similar to the ones used for non-BERT models. The only difference is that dumped BERT features must be provided here.

If you face any issues with the code or with reproducing our results, please contact anaik@cs.cmu.edu

If you find our code useful, please cite the following paper:
```
@inproceedings{naik-rose-2020-towards,
    title = "Towards Open Domain Event Trigger Identification using Adversarial Domain Adaptation",
    author = "Naik, Aakanksha  and
      Rose, Carolyn",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.681",
    doi = "10.18653/v1/2020.acl-main.681",
    pages = "7618--7624",
    abstract = "We tackle the task of building supervised event trigger identification models which can generalize better across domains. Our work leverages the adversarial domain adaptation (ADA) framework to introduce domain-invariance. ADA uses adversarial training to construct representations that are predictive for trigger identification, but not predictive of the example{'}s domain. It requires no labeled data from the target domain, making it completely unsupervised. Experiments with two domains (English literature and news) show that ADA leads to an average F1 score improvement of 3.9 on out-of-domain data. Our best performing model (BERT-A) reaches 44-49 F1 across both domains, using no labeled target data. Preliminary experiments reveal that finetuning on 1{\%} labeled data, followed by self-training leads to substantial improvement, reaching 51.5 and 67.2 F1 on literature and news respectively.",
}
```
