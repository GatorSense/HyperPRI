# HyperPRI
HyperPRI - **Hyper**spectral **P**lant **R**oot **I**magery

This Github Repo contains source code used to demonstrate how the hyperspectral data included within the HyperPRI dataset improves binary segmentation performance for a deep learning segmentation model.

==This code's public release is a work-in-progress and will be cleaned up following submissions to bioRxiv and Elsevier's COMPAG journal==

[HyperPRI Dataset](https://doi.org/10.7910/DVN/MAYDHT)
- Oct-15-2023: Initial upload/release of dataset
- Mar-25-2024: Included hyperspectral data for the viewing pane's material (Lexan)
- Jun-21-2024: Set aside rhizobox 40 as test data (dates: Aug-15 and Aug-24)

Preprint: [bioRxiv 2023.09.29.559614v3](https://www.biorxiv.org/content/10.1101/2023.09.29.559614v3)

YouTube: [Dataset Video](https://youtu.be/T1D1MBxySlI)

## Why use the HyperPRI dataset?
Data in HyperPRI **enhances plant science analyses** and provides **challenging features for machine learning** models.
- Hyperspectral data can supplement root analysis
- Study root traits across time, from seedling to reproductively mature
- Thin object features: 1-3 pixels wide
- High correlation between the high-resolution channels of hyperspectral data

## Computer Vision Tasks
There are a number of related CV tasks for this dataset:
- Compute root characteristics (length, diameter, angle, count, system architecture, hyperspectral)
- Determine root turnover
- Observe drought resiliency and response
- Compare multiple physical and hyperspectral plant traits across time
- Investigate texture analysis techniques
- Segment roots vs. soil

## HyperPRI Dataset Information
- Hyperspectral Data (400 – 1000 nm, every 2 nm)
- Temporal Data: Imaged for 14 or 15 timesteps across two months
  - Drought: Aug-06 to Aug-19, 78 - 91 days after planting (stage R6)
  - Drought: Jun-28 to Jul-21, 39 – 62 days after planting (stage V7 - V9)
- Fully-annotated segmentation masks
  - Includes annotations for peanut nodules and pegs
- Box weights at each time stamp
  - Baseline Measurements: Empty box, dry soil, wet soil
- 32 Peanut (Arachis hypogaea) rhizoboxes – 358 images
- 32 Sweet Corn (Zea mays) rhizoboxes – 390 images


# Running the Code Out of the Box
The primary Python packages used are [PyTorch](https://pytorch.org/), [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/), and related utilities. See the `environment.yml` file for specific versions.

1. Create a Conda virtual environment (ideally) with the packages requested in the provided `environment.yml` file.
    - Additional instructions for using the YAML file may be found on the [Conda site](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

1. Dataset directory setup. The full directory path from the base `HyperPRI/` repository is `Datasets/HyperPRI/`.
    - If using, place the JSON/CSV splits data in a `Datasets/HyperPRI/data_splits` subdirectory.
    - Per plant type (eg. Peanut, Sweet Corn), place them in a `Datasets/HyperPRI/{Peanut, SweetCorn}_968x608` subdirectory which hosts 3 of its own subdirectories: `hsi_files`, `mask_files`, and `rgb_files`. As the names suggest, the HSI `.dat` and `.hdr` files should be in `hsi_files`, and the PNG mask/image files should be in the `mask_files` and `rgb_files` subdirectories, respectively.
      - Please note that the paper only used a `Peanut_968x608` subdirectory.

1. Across all model training, the following holds:
    - `kfold_train.py`: Set `start_split` and `num_splits` to 0 and 5, respectively. Set the `n_seeds` to the number of training seeds desired per model.
    - `src/Experiments/params_HyperPRI.py`: Batch Size of 2. Adam optimization with 0.001 LR, standard $\beta$ values, and no weight decay. `num_classes=1` (binary).

1. For each model, the following architecture parameters were used for the paper. Henceforth, the `.../params_HyperPRI.py` file is referred to as "Parameters":
    - UNET:
      - Parameters: `n_channels=3`.
      - `kfold_train.py`: Set `dataset` equal to `"RGB"`.
      - Everything else should be hardcoded to get ~31.0M parameters.
    - SpectralUNET:
      - Parameters: `n_channels=238`, `patch_size=(608, 700)`, `augment=True`, `spectral_bn_size=1650`. The `hsi_lo` and `hsi_hi` values should be 25 and 263, respectively. This approximately corresponds to 450nm and 926nm on the EM spectrum.
      - `kfold_train.py`: Set `dataset` equal to `"HSI"`. Set `MODEL_SHARD` to `True`. This will require at least 2 GPUs to train due to the size of features when inputting multiple images. If a single GPU is desired, set `MODEL_SHARD` to `False` and decrease the value of `patch_size` in the Parameters until the training fits.
    - CubeNET-64:
      - Parameters: `n_channels=238`, `patch_size=(608, 968)`, `augment=False`, `cube_featmaps=64`. The `hsi_lo` and `hsi_hi` values should be 25 and 263, respectively. This approximately corresponds to 450nm and 926nm on the EM spectrum.
      - `kfold_train.py`: Set `dataset` equal to `"HSI"`. Set `MODEL_SHARD` to `False`.

1. Run `kfold_train.py` individually for each set of parameters in the previous step.

1. After training is finished, the models should be saved in their respective directories. Provided this is so, the `kfold_validate.py` file is all set-up and prepared for running out of the box. If segmentation maps of the dataset for all three models is requested, change the `segmaps` list to be `[True, True, True]`.

For any issues and additional questions, please direct them to [changspencer](https://github.com/changspencer).

# Training Notes & Baseline Performance
*Comet-ML/Tensorboard Logging*: If certain loggers are undesired, they can be commented out starting in the `src/PLTrainer.py > train_net` method. It will be up to the user to trace all places where the logger(s) may be disrupted through removing their instantiation/definition.

*SpectralUNET Training:* To train SpectralUNET with 1650 neurons in each layer with our coding setup and memory constraints, we had to randomly crop the hyperspectral cubes' height and width to $608\times 700$. We expect that even if the additional 268 width-wise pixels were included, the model's performance would still be subpar compared to UNET and CubeNET.

**Validation Data**
|Metric|UNET|SpectralUNET|CubeNET-64|
| :---: | :---: | :---: | :---: |
| **BCE Loss** | 0.080 (0.015) | 0.146 (0.022) | **0.077 (0.014)** |
| **DICE**     | 0.838 (0.015) | 0.717 (0.044) | **0.844 (0.013)** |
| **+IOU**     | 0.721 (0.022) | 0.561 (0.053) | **0.730 (0.019)** |
| **AP**       | 0.919 (0.013) | 0.781 (0.048) | **0.923 (0.012)** |

**Test Data**
|Metric|UNET|SpectralUNET|CubeNET-64|
| :---: | :---: | :---: | :---: |
| **Pix Acc**  | 0.733 (0.123) | 0.751 (0.114) | **0.898 (0.134)** |
| **DICE**     | 0.162 (0.053) | 0.161 (0.064) | **0.471 (0.206)** |
| **+IOU**     | 0.089 (0.031) | 0.089 (0.039) | **0.329 (0.163)** |
| **AP**       | 0.226 (0.079) | 0.220 (0.083) | **0.610 (0.109)** |

*Note:* Metrics shown are the mean across all splits with standard deviation in parentheses. Dataset splits are described in the JSON files located at the dataset URL above.
