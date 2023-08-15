# Adapting-LEAF
The PCEN Adapting experiments under two kinds of noisy environments for INTERSPEECH 2023 Paper.

**Paper title**: What is Learnt by the LEArnable Front-end (LEAF)? Adapting Per-Channel Energy Normalisation (PCEN) to Noisy Conditions

**Authors**: Hanyu Meng, Dr Vidhyasaharan Sethu, Prof Eliathamby Ambikairajah

**Insititution**: The University of New South Wales

## Introduction
This is the source code for adapting PCEN layer in LEAF under noisy environment.

## Experimental Setups and Results
**1. Four model to be trained**
<p align="center">
    <img src="Image/experiment_process_leaf.png" width="500" height="400">
</p>

* **Clean Trained**: Trained on the entire noise-free training set (baseline). 
* **Noisy Trained**: Trained on the noisy version of the entire training data. 
* **Before Adapt**: Trained on the noise-free training set without including adaptation data.
* **PCEN Adapt**: The BA model with the PCEN layer was adapted using the noisy adaptation data. 

As for the backend, we use the same back-end (EfficientNetB0) as the original LEAF paper.

**2. Speech Processing Task: Emotion Recognition**

**3. Dataset and Partition**

We applied the CREMA-D Dataset with the following partition:
<p align="center">
    <img src="Image/data_partition.png" width="500" height="400">
</p>

**4. Noise Setups**
1. We varying the Signal-to-Noise Ratio (SNR): from 0 to 20 dB with 5dB increment (We trained all 4 models under 0 dB, 5 dB, 10 dB, 15 dB, 20 dB SNR correspondingly).
2. Two representative classes of noise were chosed
    * **Gaussian Noise (Stationary Noise)**
    * **Babble Noise (Non-stationary Noise)**
        * We create the babble noise environment by randomly select 3 speech samples from the **MUSAN** dataset and mixed them up.
        * The availablity of MUSAN dataset: [Download](https://www.openslr.org/17/)
        
**5. Results and Analysis**

We evaluate the accuracy of the overall model accuracy under noisy environments (adding different kinds and level of noise to the test set).
The accuracy of all 4 models under different kinds/levels or noise is shown as follow:
<p align="center">
    <img src="Image/noise_result_new.png" width="500" height="350">
</p>

**Analysis:** 
1. Gaussian Noise Adaptation
    * Noisy training data helps the model learn the pattern of noise and improves its robustness.
    * Adapting the PCEN layer with a small amount of noisy data, the impact of noise on accuracy can be mitigated.

2. Babble Noise Adaptation
    * Training with noisy data is not as effective under babble noise conditions.
        - The reason might due to that compared with gaussian noise, babble noise has greater similarity between noise and speech.
    * Adapting PCEN with babble noise might be effective in allowing the model to be used under non-stationary noisy conditions. 

## Getting Started
### Prequest
    - python=3.8 
    - setuptools==59.5.0
    - numpy==1.23.3
    - tqdm
    - tensorboard
    - pytorch==1.10.0
    - tensorflow-datasets
    - efficientnet_pytorch
    - PySoundFile
    - soundfile

### Installation
```bash
git clone https://github.com/Hanyu-Meng/Adapting-LEAF.git
```
## Configrations
1. **Creating Noisy Dataset**
    * **Create Noisy Dataset with Different Level of Gaussian Noise**

    1. Change the dataset directory and new path in [create_noisy_dataset.py](PCEN_Adapting/Noisy_dataset_create/create_noisy_dataset.py) to your dataset directory and the directory to store the noisy dataset.

    2. Open [dataset_pre_process.py](PCEN_Adapting/Noisy_dataset_create/dataset_pre_process.py), change exec
    ```bash
    exec = "/your_environment_directory /current_path/create_noisy_dataset_dataset.py"
    ```
    3. Run [dataset_pre_process.py](PCEN_Adapting/Noisy_dataset_create/dataset_pre_process.py)
    ```bash
    python3 dataset_pre_process.py
    ```
    * **Create Noisy Dataset with Different Level of Babble Noise**

    1. Change the dataset directory, new path, and dir in [create_babble_noise_dataset.py](PCEN_Adapting/Noisy_dataset_create/create_babble_noise_dataset.py) to your dataset directory, the directory to store the noisy dataset, and your directory for MUSAN speech subset.
    ```bash
    directory = "/your_path_for_CREMA_D"
    path_new = "/your_path_for_different_level_of_Babble_CREMA_D"
    dir = "/your_path_for_MUSAN_SPEECH"
    ```
    2. Run [dataset_pre_process.py](PCEN_Adapting/Noisy_dataset_create/dataset_pre_process.py)
    ```bash
    python3 dataset_pre_process.py
    ```

2. **Baseline Models Training**

    * **Clean Trained**
        * Gaussian Noise
        ```bash
        python3 main.py --data-set 'CREMAD_SEN_90' --noise_test True  ---babble_test False --noise False --level {noise_level}
        ```
        * Babble Noise
        ```bash
        python3 main.py --data-set 'CREMAD_SEN_90'  --noise_test False --babble_test True --babble False --level {noise_level}
        ```
    * **Noisy Trained**
        * Gaussian Noise
        ```bash
        python3 main.py --data-set 'CREMAD_SEN_90' --noise_test True  ---babble_test False --noise True --level {noise_level}
        ```
        * Babble Noise
        ```bash
        python3 main.py --data-set 'CREMAD_SEN_90'  --noise_test False --babble_test True --babble True --level {noise_level}
        ```
    Alternatively, you can modified the batch-job script [batch_job_noisy_train.py](Job_scripts/batch_job_noisy_train.py) to run the experiments, it can automatically vary the noise level and store the models.

3. **Adaption Models Training**

    * **Before Adapt (BA)**
        * Gaussian Noise
        ```bash
        python3 main.py --data-set 'CREMAD' --noise_test True  ---babble_test False --noise True --level {noise_level} --tune False
        ```
        * Babble Noise
        ```bash
        python3 main.py --data-set 'CREMAD'  --noise_test False --babble_test True --babble True  --level {noise_level} --tune False
        ```
    * **PCEN Adapt (PA)**
        * Gaussian Noise
        ```bash
        python3 main.py --data-set 'CREMAD' --resume {path_to_corresponding_BA_model} --noise_test True  ---babble_test False --noise True --level {noise_level} --tune True
        ```
        * Babble Noise
        ```bash
        python3 main.py --data-set 'CREMAD' --resume {path_to_corresponding_BA_model}  --noise_test False --babble_test True --babble True --level {noise_level} --tune True
        ```
    There are also have batch_job files [batch_job_before_adapt.py](Job_scripts/batch_job_before_adapt.py) and [batch_job_pcen_adapt.py](Job_scripts/batch_job_pcen_adapt.py) for reference, you can modify the job scripts according to your settings.

## File Structures
```bash
├── Adapting-LEAF
│   ├── Job_script
│   │   ├── batch_job.py
│   │   ├── batch_job_before_adapt.py
│   ├── Noise_dataset_create
│   │   ├── image1.png
│   │   ├── image2.jpg
│   ├── Pre-trained models
│   │   ├── Babble Noise
│   │       ├── babble_BA_0db
│   │           ├── args.txt
│   │           ├── final_metrics.txt
│   │           ├── net_checkpoint.pth
│   │           ├── net_last_model.pth
│   │       ├── .....
│   │ 
│   │   ├── Gaussian Noise
│   │       ├── BA_0db
│   │           ├── args.txt
│   │           ├── final_metrics.txt
│   │           ├── net_checkpoint.pth
│   │           ├── net_last_model.pth
│   │       ├── .....
│   │   
│   └── datasets
│       ├── __init__.py
│       ├── crema_d.py
│   └── model
│       ├── __init__.py
│       ├── leaf.py
│   └── result_analysis
│       ├── ....
│   └── engine.py
│   └── main.py
│   └── utils.py
```
## Pretained Models

## References

## Acknowledgement


