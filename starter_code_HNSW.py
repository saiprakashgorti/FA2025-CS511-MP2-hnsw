import faiss
import numpy as np
import h5py

def load_hdf5_dataset(filename: str):
    """
    Load dataset from ann-benchmarks HDF5 format.
    Returns: (train_vectors, query_vectors, groundtruth)
    - train_vectors: numpy array of training vectors
    - query_vectors: numpy array of query vectors  
    - groundtruth: numpy array with groundtruth neighbor indices (first column for Recall@1)
    """
    with h5py.File(filename, 'r') as f:
        train = np.array(f['train'], dtype=np.float32)
        test = np.array(f['test'], dtype=np.float32)
        neighbors = np.array(f['neighbors'], dtype=np.int32)
        # For Recall@1, use first neighbor
        groundtruth = neighbors[:, 0:1]  # Shape: (n_queries, 1)
    return train, test, groundtruth

def evaluate_hnsw():
    # Path to SIFT dataset HDF5 file
    dataset_file = 'resources/sift-128-euclidean.hdf5'
    
    # Load dataset from HDF5
    print("Loading dataset from HDF5 file...")
    database_vectors, query_vectors, groundtruth = load_hdf5_dataset(dataset_file)
    print(f"Loaded {len(database_vectors)} database vectors with dimension {database_vectors.shape[1]}")
    print(f"Loaded {len(query_vectors)} query vectors")
    
    # Get the first query vector
    query_vector = query_vectors[0:1]  # First query vector, reshape to (1, d)
    
    # Initialize HNSW index
    d = database_vectors.shape[1]  # Dimension (128 for SIFT)
    M = 16
    efConstruction = 200
    efSearch = 200
    
    print(f"Building HNSW index with M={M}, efConstruction={efConstruction}, efSearch={efSearch}...")
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = efConstruction
    index.hnsw.efSearch = efSearch
    
    # Convert to float32 if needed (FAISS requires float32)
    database_vectors = database_vectors.astype(np.float32)
    query_vector = query_vector.astype(np.float32)
    
    # Add vectors to index
    print("Adding vectors to index...")
    index.add(database_vectors)
    print(f"Index built with {index.ntotal} vectors")
    
    # Perform search
    print("Performing search...")
    k = 10
    distances, indices = index.search(query_vector, k)
    
    # Write results to output.txt
    output_file = './output.txt'
    print(f"Writing results to {output_file}...")
    with open(output_file, 'w') as f:
        for idx in indices[0]:
            f.write(f"{idx}\n")
    
    print(f"Top 10 nearest neighbor indices: {indices[0].tolist()}")
    print("Done!")

if __name__ == "__main__":
    evaluate_hnsw()
