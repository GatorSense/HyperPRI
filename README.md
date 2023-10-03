# HyperPRI
HyperPRI - **Hyper**spectral **P**lant **R**oot **I**magery

This Github Repo contains source code used to demonstrate how the hyperspectral data included within the HyperPRI dataset improves binary segmentation performance for a deep learning segmentation model.

==This code's public release is a work-in-progress and will be cleaned up following submissions to bioRxiv and Elsevier's COMPAG journal==

DataDryad Links: (Pending approval on DataDryad)
- [Peanut (Arachis hypogaea) Data](https://doi.org/10.5061/dryad.xksn02vmg)
- [Sweet Corn (_Zea mays_) Data](https://doi.org/10.5061/dryad.t76hdr868)

Preprint: [bioRxiv 2023.09.29.559614](https://www.biorxiv.org/content/10.1101/2023.09.29.559614v1)

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

# Methodology and Performance - Summary
(To be paraphrased from preprint manuscript)
