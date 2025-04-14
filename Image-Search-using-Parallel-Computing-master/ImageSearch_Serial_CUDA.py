import os
import time
import cv2
import cupy as cp
import matplotlib.pyplot as plt
from FeatureVectors import FeatureVectors
from QuerySearch import QuerySearch


def extractFeatureVectors(image_path):
    # Extracts feature vectors for input image using GPU
    try:
        # Load and preprocess the image on CPU
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (500, 500))
        
        # Extract feature vectors using GPU
        featureVectors = FeatureVectors(image)
        vectors = featureVectors.getFeatureVector()

        imageName = image_path.split("/")[-1]
        return [imageName, vectors]
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return [image_path.split("/")[-1], []]


def ImageSearch(queryImage):
    # Performs Image Search using Query image with GPU acceleration
    image_db_path = "Image_Database/"
    image_paths = []
    for img in os.listdir(image_db_path):
        image_paths.append(image_db_path+img)

    features = {}
    for image in image_paths:
        imageName, vector = extractFeatureVectors(image)
        if vector:  # Only add if vectors were successfully extracted
            features[imageName] = vector

    queryImage_path = image_db_path+queryImage
    imageName, queryVector = extractFeatureVectors(queryImage_path)

    search = QuerySearch(queryVector, features)
    
    # Perform search with GPU acceleration
    results = search.performSearch()
    results.sort(key=lambda res: res[1])

    # Clear GPU memory after search completes
    cp.get_default_memory_pool().free_all_blocks()
    
    return results