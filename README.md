# <RHEED_Video_Analysis Using PCA & K-means Clustsering>

## Description

This project aims to utilize RHEED_videos in characterizing thin films grown by molecular deposition.
Principle Component Analysis(PCA) and K-means Clustering are used to apply an unsupervised learning and categorize the data by its statistical importance.
This project provides interactive python notebooks(PCA.ipynb & Kmeans.ipynb) that were used in applying PCA and K-means Clustering, which is applicable to other videos as well.
The provided python file(PCA_Kmeans.py) is the collection of defined functions used in the process.

## How to Use

To apply PCA and K-means CLustering:

Change (name = "YOUR_FILE_NAME") in box[2] of PCA.ipynb and Kmeans.ipynb to your according file name.
The code accepts mp4 files by default although this may be modified.

First run PCA.ipynb then Kmeans.ipynb.
Running the PCA file will automatically save the following contents.
| File Name                   | Content                                           |
|-----------------------------|---------------------------------------------------|
| 00_YOUR_FILE_NAME_eVal.txt  |   First 20 eigenvalues of the covariance matrix   |
| 00_YOUR_FILE_NAME_eVec.txt  |   First 20 eigenvectors of the covariance matrix  |
| 00_YOUR_FILE_NAME_M.txt     |   Mean of the data set                            |

PCA.ipynb will show various graphs and images such as the mean of the data or eigenvectors in image form, the variance of each frame in eigen vector direction, the mean intensity variation in time of a selected area, etc. All of which can be saved in image and\or csv format.

Once you run the PCA.ipynb for a video running the code in box[3] and box[4] is no longer necessary. This process may be time consuming thus the auto-saved files will save and restore the computed values. The Kmeans.ipynb also benefits from this method so you should always run PCA.ipynb at least once before running Kmeans.ipynb.

The Kmeans.ipynb wiil apply K-means Clustering to the video frames for k values 2-6 and output the cluster distribution, mean images of clusters and difference between clusters in image form. As the PCA.ipynb these images and graphs can be saved in image and\or csv format.

K-means Clustering may also be time consuming and the output may have different forms since K-means Clustering is a process pf finding a local minimum. This problem is dealt by determining specific cluster means rather than allowing an arbitrary choice. Box[13] in Kmeans.ipynb allows you to save computed cluster mean by selecting the cluster "ic" to be saved as cluster "mc". AFTER SAVING ALL CLUSTER MEANS UNMUTE "np.savetxt..." and this will save the cluster means. You can use the cluster means like box[12].

## Credits
