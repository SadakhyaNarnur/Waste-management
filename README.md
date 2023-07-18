# Waste-management
Multi-Object Detection for Recyclable Waste Management

Introduction:
The misclassification of waste and overuse of landfills pose a significant threat to the
ecosystem and have bred sanitary and health problems for mankind and wildlife alike. Recycling
solid waste is a critical step in reducing the harmful impacts of waste accumulation; however,
recycling first requires sorting waste material, which is complex and costly. To facilitate the
sorting process, this study proposes an intelligent waste classification system that uses deep
learning technology to detect and classify objects in waste images into six categories: cardboard,
glass, metal, paper, plastic, and trash.

Summary:
The preprocessing steps include SVD and image resizing to
increase model speed, normalization to improve model accuracy, and various data augmentation
techniques, such as adding noise and rotation, to increase model robustness. The resulting data
was fed to four supervised Deep Learning object detection algorithms: R-FCN, Faster R-CNN,
SSD, and YOLOv5. Results show YOLOv5 had the highest mAP rate at 86.2% while also being
the fastest, with inference speed of 7 ms per image. This study provides a mechanism to
categorize recyclable waste with high average precision and high speed that could reduce both
the toxic waste in our landfills and the cost of waste management.
