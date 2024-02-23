# Real Time Anomaly Segmentation
In the context of existing deep networks, their performance diminishes significantly when confronted with unknown or anomalous objects that were not part of their training data. The ability to detect such out-of-distribution (OoD) objects is crucial, especially in applications like autonomous driving and various computer vision challenges such as continual learning and open-world scenarios.

In this project, the objective was to develop compact anomaly segmentation models capable of real-time deployment. The aim was to create models that could run efficiently on small devices, aligning with the memory constraints typically encountered in edge applications. Specifically, the focus was on designing models suitable for deployment on smart cameras with limited onboard processing capacity.

# Dataset
For training the model, we use the Cityscapes dataset.

# Tasks

## Task 1
Here various anomaly inferences are done using a pre-trained ERF-Net model and anomaly segmentation test dataset provided.

| Model | Method   | Dataset              | AUPRC Score | FPR@TPR95     |
|-------|----------|----------------------|-------------|---------------|
| erfnet| msp      | RoadAnomaly21        | 14.585      | 95.090        |
| erfnet| msp      | RoadObstacle21       | 0.720       | 94.769        |
| erfnet| msp      | FS_LostFound_full    | 0.257       | 95.829        |
| erfnet| msp      | fs_static            | 1.982       | 95.259        |
| erfnet| msp      | RoadAnomaly          | 9.427       | 95.301        |
| erfnet| maxlogit | RoadAnomaly21        | 15.240      | 93.813        |
| erfnet| maxlogit | RoadObstacle21       | 0.972       | 85.607        |
| erfnet| maxlogit | FS_LostFound_full    | 0.227       | 97.079        |
| erfnet| maxlogit | fs_static            | 1.726       | 94.056        |
| erfnet| maxlogit | RoadAnomaly          | 9.436       | 94.731        |
| erfnet| maxentropy| RoadAnomaly21        | 14.488      | 95.214        |
| erfnet| maxentropy| RoadObstacle21       | 0.765       | 94.510        |

## Task 2
Temperature scaling is a method for confidence calibration for any classifier which could result in improving anomaly segmentation capabilities of a network. Here try a simple grid search for temperatuers.

| Model | Method | Dataset        | Temperature | AUPRC Score | FPR@TPR95      |
|-------|--------|----------------|-------------|-------------|----------------|
| erfnet| msp    | RoadAnomaly21  | 1.0         | 14.585      | 95.090         |
| erfnet| msp    | RoadAnomaly21  | 0.5         | 14.675      | 95.052         |
| erfnet| msp    | RoadAnomaly21  | 0.75        | 14.626      | 95.072         |
| erfnet| msp    | RoadAnomaly21  | 1.1         | 14.570      | 95.098         |
| erfnet| msp    | RoadObsticle21 | 1.0         | 0.721       | 94.769         |
| erfnet| msp    | RoadObsticle21 | 0.5         | 0.699       | 94.886         |
| erfnet| msp    | RoadObsticle21 | 0.75        | 0.710       | 94.827         |
| erfnet| msp    | RoadObsticle21 | 1.1         | 0.725       | 94.745         |

## Task 3
The cityscapes dataset comprises 19 recognized category classes along with a void category representing the background. In this segment, we'll treat the void class as an anomaly and train both the ENet and BiSeNet networks accordingly. Subsequently, we'll conduct anomaly inference by exclusively focusing on the output related to the Void class.

| Model   | Method | Dataset            | AUPRC Score | FPR@TPR95      |
|---------|--------|--------------------|-------------|----------------|
| enet    | void   | RoadAnomaly21      | 14.594      | 95.134         |
| enet    | void   | RoadObstacle21     | 0.672       | 95.118         |
| enet    | void   | FS_LostFound_full  | 0.275       | 95.210         |
| enet    | void   | fs_static          | 2.094       | 94.834         |
| enet    | void   | RoadAnomaly        | 9.737       | 95.034         |
| erfnet  | void   | RoadAnomaly21      | 14.687      | 95.119         |
| erfnet  | void   | RoadObstacle21     | 0.675       | 95.015         |
| erfnet  | void   | FS_LostFound_full  | 0.277       | 95.307         |
| erfnet  | void   | fs_static          | 1.784       | 95.887         |
| erfnet  | void   | RoadAnomaly        | 9.664       | 95.161         |
| bisenet | void   | RoadAnomaly21      | 17.408      | 94.451         |
| bisenet | void   | RoadObstacle21     | 0.685       | 95.010         |
| bisenet | void   | FS_LostFound_full  | 0.282       | 95.384         |
| bisenet | void   | fs_static          | 1.756       | 95.244         |
| bisenet | void   | RoadAnomaly        | 12.024      | 94.532         |

# Conclusion
Based on the results presented in the tables, several key conclusions can be drawn:

1. **Model Performance:** The performance of different models (erfnet) varies significantly depending on the method employed (msp, maxlogit, maxentropy) and the dataset used (RoadAnomaly21, RoadObstacle21, FS_LostFound_full, fs_static, RoadAnomaly). This indicates the importance of selecting the appropriate combination of model and method for specific datasets.

2. **Anomaly Detection Ability:** Across different datasets, models exhibit varying degrees of success in detecting anomalies, as evidenced by the AUPRC scores. For instance, the AUPRC scores for the RoadAnomaly21 dataset are generally higher compared to other datasets, suggesting that models perform better at detecting anomalies in this particular scenario.

3. **False Positive Rate:** The FPR@TPR95 metric provides insights into the false positive rate of the models. Lower FPR@TPR95 values indicate better performance in terms of minimizing false positives, which is crucial for real-world applications where accurate anomaly detection is paramount.

4. **Effectiveness of Methods:** The choice of method (msp, maxlogit, maxentropy) also influences model performance. For instance, models utilizing the maxlogit method generally achieve higher AUPRC scores and lower false positive rates compared to other methods across multiple datasets.

In conclusion, the results highlight the importance of careful model selection and method implementation for effective anomaly detection in various real-world scenarios. Further research could focus on optimizing models and methods to improve overall performance and address specific challenges encountered in anomaly detection tasks.

# Project information
This project repository is created as part of Advanced Machine Learning course at Polytechnic University of Turin. Mr. [Shyam Nandan Rai](https://github.com/shyam671) was the tutor of this project and provided the sample code.