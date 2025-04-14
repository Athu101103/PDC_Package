import math
import cupy as cp
import numpy as np


class QuerySearch:
    """ Searches similarity of query image vector with extracted vectors from database using GPU acceleration"""

    def __init__(self, queryFeature, features):
        self.features = features
        # Convert queryFeature to GPU array for faster computation
        self.queryFeature = queryFeature
        self.queryFeature_gpu = cp.asarray(queryFeature, dtype=cp.float32)

    def __cosine_similarity(self, queryVector_gpu, vector):
        # Computes cosine similarity between two vectors using GPU
        eps = 1e-10
        vector_gpu = cp.asarray(vector, dtype=cp.float32)
        
        dot_product = cp.sum(queryVector_gpu * vector_gpu).get()
        
        queryMagnitude = cp.sqrt(cp.sum(queryVector_gpu**2)).get()
        vectorMagnitude = cp.sqrt(cp.sum(vector_gpu**2)).get()
        magnitude = (queryMagnitude * vectorMagnitude) + eps
        
        if not magnitude:
            return 0
            
        cosine_similarity = round(dot_product / magnitude, 5)
        return cosine_similarity

    def __chi2_distance(self, queryVector_gpu, vector):
        # Computes chi-square distance between two vectors using GPU
        eps = 1e-10
        vector_gpu = cp.asarray(vector, dtype=cp.float32)
        
        # GPU implementation of chi-square distance
        numerator = (queryVector_gpu - vector_gpu) ** 2
        denominator = queryVector_gpu + vector_gpu + eps
        dists = numerator / denominator
        
        chi2_distance = 0.5 * cp.sum(dists).get()
        chi2_distance = round(chi2_distance, 5)
        return chi2_distance

    def __SAD_distance(self, queryVector_gpu, vector):
        # Computes SAD distance between two vectors using GPU
        vector_gpu = cp.asarray(vector, dtype=cp.float32)
        dists = cp.abs(queryVector_gpu - vector_gpu)
        SAD_distance = cp.sum(dists).get()
        return SAD_distance

    def performSearch(self):
        # Return similarity indices of queryVector and database images using GPU
        searchSimilarityScores = []

        # Pre-load query vector to GPU
        queryVector_gpu = self.queryFeature_gpu
        
        # Process in batches to maximize GPU throughput
        batch_size = 50
        images = list(self.features.keys())
        results = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_vectors = [self.features[img] for img in batch_images]
            
            # Calculate similarity scores for batch
            for j, image in enumerate(batch_images):
                vector = batch_vectors[j]
                # Using chi2_distance as in the original code
                similarity = self.__chi2_distance(queryVector_gpu, vector)
                searchSimilarityScores.append((image, similarity))
        
        # Free GPU memory
        del queryVector_gpu
        cp.get_default_memory_pool().free_all_blocks()
        
        return searchSimilarityScores