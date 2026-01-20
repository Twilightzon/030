# PODD: Pareto-Optimized Dual-Teacher Distillation for Fair Graph Learning under Distribution Shifts

This repository contains the official PyTorch implementation for the paper "PODD: Pareto-Optimized Dual-Teacher Distillation for Fair Graph Learning under Distribution Shifts".

## 📋 Overview

PODD is a plug-and-play framework designed to tackle the challenge of maintaining both fairness and utility in Graph Neural Networks (GNNs) under distribution shifts. It employs a dual-teacher distillation architecture combined with a meta-learning-based Pareto optimization mechanism.

**Key Components:**
- **Fairness Teacher ($T_{fair}$)**: Debiases representations via adversarial refinement (compatible with SOTA fair GNNs).
- **Distribution Shift Teacher ($T_{shift}$)**: Captures stable patterns via simulated perturbations (Gaussian noise & Edge dropout).
- **Pareto Optimization**: Dynamically resolves gradient conflicts between fairness and distribution shifts objectives.
- **Information Bottleneck**: Filters out task-irrelevant noise in the student model.

## 🛠️ Dependencies

* python>=3.7
* torch==2.0.1
* torch-geometric==2.3.1
* torch-scatter==2.1.1
* numpy==1.24.4
* scikit-learn==1.3.0

## Project Structure


├── dataset/               
│   ├── bail/
│   ├── credit/
│   └── pokec/
├── models/                
│   ├── _init_.py         
│   ├── gcn.py          
│   ├── gin.py   
│   ├── meta_fairness.py  
│   └── podd.py/    
├── train.py        
├── utils.py       
├── run.sh                 
└── README.md


🧩 Plug-and-Play Framework PODD is designed as a model-agnostic, plug-and-play framework that can be seamlessly integrated with existing Fair GNNs (e.g., FairSIN, FairGB, NIFTY) or standard GNN backbones (e.g., GCN, GIN). It functions as a wrapper that enhances the base model's resistance against distribution shifts without requiring modifications to the model's internal architecture. When adapting PODD to other models, be sure to save the trained model first before entering our training phase.




## Running

The ```run.sh``` includes details to reproduce experimental results in the paper:

```
bash run.sh
```





















