#  Anomaly Detection in IoT Devices

Anomaly detection in Internet of Things (IoT) devices is critical for identifying unusual patterns that may indicate security breaches, faults, or other operational issues. Given the large volume and diversity of data generated in IoT environments, distinguishing between normal and abnormal behaviors requires advanced machine learning techniques.

This project benchmarks multiple anomaly detection models to evaluate their performance on real-world IoT data.

---

##  Project Structure

```
iot-ad/
│
├── models/                 # Training scripts and model definitions
├── results/                # Evaluation metrics and visualizations
│   ├── ae/                 # Autoencoder results
│   ├── iforest/            # Isolation Forest results
│   ├── ocsvm/              # One-Class SVM results
│   └── openmax/            # OpenMax results
│
├── best_base_model.pth     # Pretrained model weights (PyTorch)
├── README.md               # Project documentation
├── .gitignore
└── .gitattributes
```

---

##  Models Used

The following models are implemented and evaluated:

- **Isolation Forest (iForest)**  
  A tree-based ensemble method that isolates anomalies rather than profiling normal data.

- **Autoencoder (AE)**  
  Neural networks trained to reconstruct input data. Anomalies have higher reconstruction error.

- **One-Class SVM (OC-SVM)**  
  Learns a decision boundary around the normal class.

- **OpenMax**  
  A deep learning extension for open set recognition, useful when unknown anomalies appear.

---

##  Dataset

The project uses the **[CIC IoT Dataset 2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html)**, which includes benign and malicious behaviors across various smart home IoT devices. The dataset is already preprocessed for this project.


---

##  Evaluation & Results

Each model is evaluated using:

- **Cross-validation** metrics (`all_cv_results.json`)
- **Final test performance** (`final_test_evaluation.json`)
- **Best hyperparameters** (`best_params.json`)
- **Visualizations** (`pdf/` folders with comparative plots)

Examples include:
- ROC curves
- Confusion matrices
- Reconstruction error distributions (for Autoencoders)

---

##  Example Outputs

Evaluation reports (in JSON and PDF) are saved in the `results/` directory. For example:

- `results/ae/final_test_evaluation.json`  
- `results/iforest/pdf/best_vs_worst_configs.pdf`

---


##  License

This project is open-source.
