import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
import numpy as np
from skimage.measure import regionprops
from scipy.spatial.distance import cdist
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF

class NucleusFeatureExtractor:
    def __init__(self, feature_dim=512):
        # Initialize ResNet models for both DAPI and marker channels
        self.dapi_model = resnet18(pretrained=True)
        self.marker_model = resnet18(pretrained=True)
        
        # Remove the final fully connected layer
        self.dapi_model = nn.Sequential(*list(self.dapi_model.children())[:-1])
        self.marker_model = nn.Sequential(*list(self.marker_model.children())[:-1])
        
        # Freeze the models
        for param in self.dapi_model.parameters():
            param.requires_grad = False
        for param in self.marker_model.parameters():
            param.requires_grad = False
            
        self.feature_dim = feature_dim
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])
        
        self.feature_names = [
            'deep_features_dapi',
            'deep_features_marker',
            'area',
            'perimeter',
            'eccentricity',
            'solidity'
        ]
    
    def extract_patch(self, image, bbox, padding=10):
        """Extract and preprocess a patch from the image."""
        min_row, min_col, max_row, max_col = bbox
        
        # Add padding
        min_row = max(0, min_row - padding)
        min_col = max(0, min_col - padding)
        max_row = min(image.shape[0], max_row + padding)
        max_col = min(image.shape[1], max_col + padding)
        
        patch = image[min_row:max_row, min_col:max_col]
        
        # Convert to PIL Image and replicate to 3 channels if grayscale
        patch_pil = Image.fromarray(patch)
        if len(patch.shape) == 2:
            patch_pil = Image.fromarray(np.stack([patch] * 3, axis=-1))
            
        # Apply transformations
        patch_tensor = self.transform(patch_pil)
        return patch_tensor.unsqueeze(0)
    
    def extract_features(self, image_sample):
        """Extract features from each nucleus in the image.
        
        Args:
            image_sample (ImageSample): Instance containing DAPI and marker images
            
        Returns:
            features (np.ndarray): Feature matrix (n_nuclei x n_features)
            centroids (np.ndarray): Nucleus centroids for building the graph
        """
        if image_sample.dapi_multi_mask is None:
            raise ValueError("No nucleus mask available")
            
        # Get region properties for each nucleus
        props = regionprops(image_sample.dapi_multi_mask)
        
        # Process each nucleus
        features = []
        centroids = []
        
        for prop in props:
            # Extract patches for both channels
            dapi_patch = self.extract_patch(image_sample.dapi, prop.bbox)
            
            # Get deep features for DAPI
            with torch.no_grad():
                dapi_features = self.dapi_model(dapi_patch)
                dapi_features = dapi_features.squeeze().cpu().numpy()
            
            # Get marker features if available
            if image_sample.marker is not None:
                marker_channel = image_sample.marker[:, :, 0] if image_sample.marker.ndim == 3 else image_sample.marker
                marker_patch = self.extract_patch(marker_channel, prop.bbox)
                
                with torch.no_grad():
                    marker_features = self.marker_model(marker_patch)
                    marker_features = marker_features.squeeze().cpu().numpy()
            else:
                marker_features = np.zeros(self.feature_dim)
            
            # Combine all features
            nucleus_features = np.concatenate([
                dapi_features,
                marker_features,
                [prop.area,
                 prop.perimeter,
                 prop.eccentricity,
                 prop.solidity]
            ])
            
            features.append(nucleus_features)
            centroids.append(prop.centroid)
        
        return np.array(features), np.array(centroids)

class NucleusGraphBuilder:
    def __init__(self, max_distance=50):
        self.max_distance = max_distance
        
    def build_graph(self, features, centroids):
        """Build a graph from nucleus features and positions.
        
        Args:
            features (np.ndarray): Nucleus features
            centroids (np.ndarray): Nucleus centroids
            
        Returns:
            Data: PyTorch Geometric graph data object
        """
        # Calculate pairwise distances between nuclei
        distances = cdist(centroids, centroids)
        
        # Create edges between nearby nuclei
        edges = []
        edge_weights = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                if distances[i, j] <= self.max_distance:
                    edges.append([i, j])
                    edges.append([j, i])  # Add both directions
                    edge_weights.extend([distances[i, j], distances[i, j]])
        
        # Convert to PyTorch tensors
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)
        x = torch.tensor(features, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

class NucleusGNN(nn.Module):
    def __init__(self, num_features, hidden_dim=32):
        super(NucleusGNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        
    def forward(self, data):
        """Forward pass through the GNN.
        
        Args:
            data (Data): PyTorch Geometric graph data object
            
        Returns:
            torch.Tensor: Binary classification logits for each nucleus
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Graph convolutions
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        
        # Final classification
        return self.classifier(x)

class NucleusClassifier:
    def __init__(self, feature_extractor=None, graph_builder=None, model=None):
        self.feature_extractor = feature_extractor or NucleusFeatureExtractor()
        self.graph_builder = graph_builder or NucleusGraphBuilder()
        self.model = model or NucleusGNN(len(self.feature_extractor.feature_names))
        
    def process_sample(self, image_sample):
        """Process a single image sample.
        
        Args:
            image_sample (ImageSample): Instance containing images and masks
            
        Returns:
            torch.Tensor: Classification predictions for each nucleus
            np.ndarray: Nucleus centroids
        """
        # Extract features
        features, centroids = self.feature_extractor.extract_features(image_sample)
        
        # Build graph
        graph_data = self.graph_builder.build_graph(features, centroids)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            logits = self.model(graph_data)
            predictions = torch.sigmoid(logits).numpy().flatten()
            
        return predictions, centroids
    
    def generate_cell_bodies_mask(self, image_sample, predictions, centroids):
        """Generate cell bodies mask based on nucleus predictions.
        
        Args:
            image_sample (ImageSample): Input image sample
            predictions (np.ndarray): Binary predictions for each nucleus
            centroids (np.ndarray): Nucleus centroids
            
        Returns:
            np.ndarray: Cell bodies mask with positive cells
        """
        # Use existing cell expansion method
        finder = FindMarker(max_expansion=10)
        marker_channel = image_sample.marker[:, :, 0] if image_sample.marker.ndim == 3 else image_sample.marker
        
        # Get cell bodies mask using cell expansion
        _, _, _, cell_bodies_mask, _ = finder.cell_expansion(
            marker_channel, image_sample.dapi_multi_mask)
        
        # Create positive cell bodies mask based on predictions
        positive_indices = np.where(predictions > 0.5)[0] + 1  # Add 1 since mask labels start at 1
        positive_mask = np.isin(cell_bodies_mask, positive_indices)
        
        return positive_mask.astype(np.uint8) * cell_bodies_mask