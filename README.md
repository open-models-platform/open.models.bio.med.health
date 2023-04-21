
OpenModels aims to revolutionize the healthcare industry by leveraging large models and training them, or aligning them with existing large models, for various applications, including:

• Intelligent Diagnosis and Treatment: OpenModels will develop advanced models for accurate diagnosis and personalized treatment plans by analyzing patient data, medical histories, and current conditions. The models will be trained using professional data and expert feedback from the medical industry.

• Bioinformatics: OpenModels will use large models to analyze complex biological data, such as genomics, proteomics, and metabolomics, to advance our understanding of diseases and potential therapeutic targets.

• Medical Natural Language Processing: OpenModels will enhance medical NLP models for processing clinical texts, enabling automatic extraction of key information, summarization, and translation of medical documents.

• Medical Image Recognition and Image Processing: OpenModels will employ advanced models to analyze medical images for improved diagnosis, monitoring, and treatment planning. The models will be trained using high-quality image datasets and expert feedback.

• Machine Learning and Medical Deep Learning: OpenModels will utilize cutting-edge machine learning and deep learning techniques to develop models that can predict disease progression, identify new drug targets, and optimize treatment strategies.

• Intelligent Medical Voice Input and Recognition Technology: OpenModels will develop voice recognition models tailored for medical applications, enabling efficient voice-to-text transcription and voice-activated assistance in clinical settings.

• Hospital Information Management System: OpenModels will integrate large models into hospital information management systems to streamline data management, automate processes, and enhance decision-making.

• Artificial Intelligence Drug Synthesis System: OpenModels will leverage large models to design and synthesize new drugs with optimal efficacy and safety profiles.

• Virtual Diagnosis and Treatment System: OpenModels will create virtual platforms to simulate patient consultations, diagnose conditions, and recommend treatment plans, offering medical professionals a valuable training tool.

• Intelligent Health Management: OpenModels will develop personalized health management models to track individual health data, provide lifestyle recommendations, and facilitate preventative care.

• Virtual Nursing: OpenModels will create virtual nursing models to support patient care remotely, offering personalized assistance and monitoring.

• Telemedicine: OpenModels will enhance telemedicine platforms by incorporating large models for improved diagnosis, treatment, and patient engagement.

• Disease Prediction and Forecasting System: OpenModels will use large models to predict disease outbreaks and develop early warning systems for public health preparedness.

• Surgical and Rehabilitation Robots: OpenModels will train models to control surgical and rehabilitation robots, enhancing precision and patient outcomes.

• Electronic Medical Record Applications: OpenModels will develop models for efficient and secure management of electronic medical records, facilitating seamless data sharing and collaboration.

• Healthcare Collaboration Technology: OpenModels will create collaborative platforms to connect healthcare professionals, enabling information sharing, consultation, and coordinated care.

• Healthcare Computer Games: OpenModels will develop educational and therapeutic healthcare games using large models, promoting patient engagement and understanding.

• Computer-Aided Detection and Diagnosis: OpenModels will refine models for computer-aided detection and diagnosis, enhancing the accuracy and efficiency of medical imaging analysis.

• Epidemiology Modeling: OpenModels will develop models to simulate and analyze disease spread, informing public health interventions and policies.

• Genetic Counseling: OpenModels will create models for personalized genetic counseling, helping patients understand their genetic risks and potential treatment options.

• Health Informatics Education: OpenModels will develop educational tools and resources for health informatics students and professionals, utilizing large models for enhanced learning experiences.

• Health Information Systems: OpenModels will integrate large models into health information systems, optimizing data management, analysis, and decision-making.

• Health Risk Assessment and Modeling: OpenModels will develop models for assessing individual and population health risks, informing preventative measures and resource allocation.

• Health Software Architecture, Frameworks, Design: OpenModels will create robust and flexible software solutions for healthcare applications, using large models to enable seamless integration and customization.

• Health Informatics: OpenModels will advance health informatics by using large models to process and analyze complex healthcare data, enabling better decision-making and improved patient outcomes.

• Intelligent Medical Management Systems: OpenModels will develop AI-driven medical management systems to optimize hospital workflows, resource allocation, and patient care.

• Healthcare Software: OpenModels will create innovative healthcare software solutions by harnessing the power of large models, addressing diverse medical needs and applications.

• Medical Insurance Fraud Detection: OpenModels will employ large models to detect and prevent medical insurance fraud, ensuring the integrity of healthcare systems and protecting consumers.

• Mobile Health and Sensor Applications: OpenModels will develop mobile health applications and sensor-based solutions using large models, facilitating remote monitoring, personalized care, and patient engagement.

For each application, OpenModels will utilize specialized data and collect expert feedback from the medical industry. This feedback will be used to train the models through Reinforcement Learning with Human Feedback (RLHF) to ensure high performance in the corresponding medical scenarios. By leveraging large models in these areas, OpenModels aims to revolutionize the healthcare landscape, improving patient care, optimizing resource utilization, and accelerating medical innovation.


-  **Intelligent Diagnosis and Treatment**
-  **Bioinformatics**
-  **Medical Natural Language Processing**
-  **Medical Image Recognition and Image Processing**
-  **Machine Learning and Medical Deep Learning**
-  **Intelligent Medical Voice Input and Recognition Technology**
-  **Hospital Information Management System**
-  **Artificial Intelligence Drug Synthesis System**
-  **Virtual Diagnosis and Treatment System**
-  **Intelligent Health Management**
-  **Virtual Nursing**
-  **Telemedicine**
-  **Disease Prediction and Forecasting System**
-  **Surgical and Rehabilitation Robots**
-  **Electronic Medical Record Applications**
-  **Healthcare Collaboration Technology**
-  **Healthcare Computer Games**
-  **Computer-Aided Detection and Diagnosis**
-  **Epidemiology Modeling**
-  **Genetic Counseling**
-  **Health Informatics Education**
-  **Health Information Systems**
-  **Health Risk Assessment and Modeling**
-  **Health Software Architecture, Frameworks, Design**
-  **Health Informatics**
-  **Intelligent Medical Management Systems**
-  **Healthcare Software**
-  **Medical Insurance Fraud Detection**
-  **Mobile Health and Sensor Applications**



# OpenBioMed
This repository holds OpenBioMed, an open-source toolkit for multi-modal representation learning in AI-driven biomedical research. Our focus is on multi-modal information, e.g. knowledge graphs and biomedical texts for drugs, proteins, and single cells, as well as a wide range of applications, including drug-target interaction prediction, molecular property prediction, cell-type prediction, molecule-text retrieval, molecule-text generation, and drug-response prediction. **Researchers can compose a large number of deep learning models including LLMs like BioMedGPT-1.6B and CellLM to facilitate downstream tasks.** We provide easy-to-use APIs and commands to accelerate life science research.

## News!

- [04/23] 🔥The pre-alpha BioMedGPT model and OpenBioMed are available!

## Features

- **3 different modalities for drugs, proteins, and cell-lines**: molecular structure, knowledge graphs, and biomedical texts. We present a unified and easy-to-use pipeline to load, process, and fuse the multi-modal information.
- **BioMedGPT-1.6B, including other 20 deep learning models**, ranging from CNNs and GNNs to Transformers. **BioMedGPT-1.6B** is a pre-trained multi-modal molecular foundation model with 1.6B parameters that associates 2D molecular graphs with texts. We also present **CellLM**, a single cell foundation model with 50M parameters.  
The checkpoints of BioMedGPT-1.6B and CellLM can be downloaded from [here](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg) (password is 7a6b). You can test the performance of BioMedGPT-1.6B on molecule-text retrieval by running scripts/mtr/run.sh, or test the performance of CellLM on cell type classification by running scripts/ctc/train.sh.
- **8 downstream tasks** including AIDD tasks like drug-target interaction prediction and molecule property training, as wel as cross-modal tasks like molecule captioning and text-based molecule generation.  
- **20+ datasets** that are most popular in AI-driven biomedical research. Reproductible benchmarks with abundant model combinations and comprehensive evaluations are provided.
- **3 knowledge graphs** with extensive domain expertise. We present **BMKGv1**, a knowledge graph containing 6,917 drugs, 19,992 proteins, and 2,223,850 relationships with text descriptions. We offer APIs to load and process these graphs and link drugs and proteins based on structural information.

## Installation

To support basic usage of OpenBioMed, run the following command:

```bash
conda create -n OpenBioMed python=3.8
conda activate OpenBioMed
conda install -c conda-forge rdkit
pip install torch

# for torch_geometric, please follow instructions at https://github.com/pyg-team/pytorch_geometric to install the correct version of pyg
pip install torch_cluster-1.6.0-cp38-cp38-linux_x86_64.whl
pip install torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
pip install torch_sparse-0.6.14-cp38-cp38-linux_x86_64.whl
pip install torch_spline_conv-1.2.1-cp38-cp38-linux_x86_64.whl
pip install torch-geometric

pip install transformers 
pip install ogb
```

**Note** that additional packages may be required for specific downstream tasks.

## Quick Start

Here, we provide a quick example of training DeepDTA for drug-target interaction prediction on the Davis dataset. For more models, datasets, and tasks, please refer to our [scripts](./open_biomed/scripts) and [documents](./docs).

### Step 1: Data Preparation

Install the Davis dataset [here](https://github.com/hkmztrk/DeepDTA/tree/master/data) and run the following:

```
mkdir datasets
cd datasets
mkdir dti
mv [your_path_of_davis] ./dti/davis
```

### Step 2: Training and Evaluation

Run:

```bash
cd ../open_biomed
bash scripts/dti/train_baseline_regression.sh
```

The results will look like the following (running takes around 40 minutes on an NVIDIA A100 GPU):

```bash
INFO - __main__ - MSE: 0.2198, Pearson: 0.8529, Spearman: 0.7031, CI: 0.8927, r_m^2: 0.6928
```

## Contact Us

As a pre-alpha version release, we are looking forward to user feedback to help us improve our framework. If you have any questions or suggestions, please open an issue or contact [dair@air.tsinghua.edu.cn](mailto:dair@air.tsinghua.edu.cn).


## Cite Us

If you find our open-sourced code & models helpful to your research, please also consider star🌟 and cite📑 this repo. Thank you for your support!
```
@misc{OpenBioMed_code,
  author={Yizhen, Luo and Kai, Yang and Massimo, Hong and Xingyi, Liu and Suyuan, Zhao and Jiahuan, Zhang and Yushuai, Wu and Zaiqing, Nie},
  title={Code of OpenBioMed},
  year={2023},
  howpublished = {\url{https://github.com/BioFM/OpenBioMed.git}}
}
```

## Contributing

If you encounter problems using OpenBioMed, feel free to create an issue! 
