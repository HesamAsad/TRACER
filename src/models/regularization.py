import torch
import torch.nn.functional as F
import numpy as np

def geodesic_regularization(ft_image_features, ft_text_features, 
                          pt_image_features=None, pt_text_features=None,
                          use_both_modalities=True):
    """
    Geodesic regularization preserving angular relationships.
    
    Args:
        ft_image_features: Fine-tuned image features (B x D)
        ft_text_features: Fine-tuned text features (B x D)
        pt_image_features: Pre-trained image features (B x D)
        pt_text_features: Pre-trained text features (B x D)
        use_both_modalities: Whether to regularize both image and text
    
    Returns:
        Geodesic loss scalar
    """
    loss = 0.0
    
    if pt_image_features is not None:
        # Compute cosine similarity (dot product for normalized features)
        cos_sim = (ft_image_features * pt_image_features).sum(dim=1)
        # Clamp to avoid numerical issues with arccos
        cos_sim = torch.clamp(cos_sim, -1 + 1e-7, 1 - 1e-7)
        # Geodesic distance squared
        geo_dist_sq = torch.arccos(cos_sim).pow(2)
        loss += geo_dist_sq.mean()
    
    if use_both_modalities and pt_text_features is not None:
        cos_sim = (ft_text_features * pt_text_features).sum(dim=1)
        cos_sim = torch.clamp(cos_sim, -1 + 1e-7, 1 - 1e-7)
        geo_dist_sq = torch.arccos(cos_sim).pow(2)
        loss += geo_dist_sq.mean()
    
    return loss


def spectral_regularization_logdet(ft_image_features, ft_text_features,
                                  epsilon=1e-5, use_both_modalities=True):
    """
    Log-determinant regularization to maximize singular values.
    
    Args:
        ft_image_features: Fine-tuned image features (B x D)
        ft_text_features: Fine-tuned text features (B x D)
        epsilon: Small constant for numerical stability
        use_both_modalities: Whether to regularize both modalities
    
    Returns:
        Log-determinant loss scalar
    """
    def compute_logdet_loss(features):
        # Compute normalized covariance
        features_centered = features - features.mean(dim=0, keepdim=True)
        # For normalized features, we need to re-normalize after centering
        features_centered = F.normalize(features_centered, p=2, dim=1)
        
        # Covariance matrix
        cov = torch.mm(features_centered.t(), features_centered) / features.size(0)
        
        # Add epsilon for stability
        cov_reg = cov + epsilon * torch.eye(cov.size(0), device=cov.device)
        
        # Compute log determinant
        # Using Cholesky for numerical stability
        try:
            L = torch.linalg.cholesky(cov_reg)
            logdet = 2 * torch.sum(torch.log(torch.diagonal(L)))
        except:
            # Fallback to eigendecomposition
            eigenvalues = torch.linalg.eigvalsh(cov_reg)
            logdet = torch.sum(torch.log(eigenvalues + epsilon))
        
        return -logdet  # Negative because we want to maximize
    
    loss = compute_logdet_loss(ft_image_features)
    
    if use_both_modalities:
        loss += compute_logdet_loss(ft_text_features)
        
    return loss


def vmf_entropy_regularization(ft_image_features, ft_text_features,
                             use_both_modalities=True):
    """
    Von Mises-Fisher entropy maximization to prevent concentration.
    
    Args:
        ft_image_features: Fine-tuned image features (B x D)
        ft_text_features: Fine-tuned text features (B x D)
        use_both_modalities: Whether to regularize both modalities
    
    Returns:
        vMF entropy loss scalar
    """
    def compute_vmf_loss(features):
        # Compute mean direction
        mean_direction = features.mean(dim=0)
        # Mean resultant length
        r_bar = torch.norm(mean_direction)
        
        # Estimate concentration parameter kappa
        # Using approximation: kappa ≈ r_bar * (d - r_bar^2) / (1 - r_bar^2)
        d = features.size(1)
        kappa = r_bar * (d - r_bar**2) / (1 - r_bar**2 + 1e-7)
        
        # Entropy loss (negative entropy)
        # Simplified approximation for computational efficiency
        loss = kappa * r_bar - 0.5 * d * torch.log(kappa + 1e-7)
        
        return loss
    
    loss = compute_vmf_loss(ft_image_features)
    
    if use_both_modalities:
        loss += compute_vmf_loss(ft_text_features)
        
    return loss


def grassmannian_subspace_alignment(ft_image_features, ft_text_features,
                                   pt_image_features=None, pt_text_features=None,
                                   subspace_dim=64):
    """
    Grassmannian distance between subspaces spanned by features.
    
    Args:
        ft_image_features: Fine-tuned image features (B x D)
        ft_text_features: Fine-tuned text features (B x D)
        pt_image_features: Pre-trained image features (B x D)
        pt_text_features: Pre-trained text features (B x D)
        subspace_dim: Number of principal components to consider
    
    Returns:
        Grassmannian distance loss
    """
    def compute_principal_angles(features1, features2, k):
        # Compute top-k subspaces via SVD
        U1, _, _ = torch.svd(features1.t())
        U2, _, _ = torch.svd(features2.t())
        
        # Take top-k components
        U1_k = U1[:, :k]
        U2_k = U2[:, :k]
        
        # Compute M = U1_k^T @ U2_k
        M = torch.mm(U1_k.t(), U2_k)
        
        # Singular values of M are cosines of principal angles
        _, S, _ = torch.svd(M)
        
        # Principal angles
        S_clamped = torch.clamp(S, -1 + 1e-7, 1 - 1e-7)
        principal_angles = torch.arccos(S_clamped)
        
        # Grassmannian distance squared
        return torch.sum(principal_angles**2)
    
    loss = 0.0
    k = min(subspace_dim, ft_image_features.size(0), ft_image_features.size(1))
    
    if pt_image_features is not None:
        loss += compute_principal_angles(ft_image_features, pt_image_features, k)
    
    if pt_text_features is not None:
        loss += compute_principal_angles(ft_text_features, pt_text_features, k)
        
    return loss


def sliced_wasserstein_sphere(ft_image_features, ft_text_features,
                            pt_image_features=None, pt_text_features=None,
                            num_projections=50):
    """
    Sliced Wasserstein distance on the sphere.
    
    Args:
        ft_image_features: Fine-tuned image features (B x D)
        ft_text_features: Fine-tuned text features (B x D)
        pt_image_features: Pre-trained image features (B x D)
        pt_text_features: Pre-trained text features (B x D)
        num_projections: Number of random projections
    
    Returns:
        Sliced Wasserstein distance
    """
    def compute_sw_distance(features1, features2, num_proj):
        device = features1.device
        d = features1.size(1)
        
        # Generate random directions on the sphere
        theta = torch.randn(d, num_proj, device=device)
        theta = F.normalize(theta, p=2, dim=0)
        
        # Project features onto random directions
        proj1 = torch.mm(features1, theta)  # B x num_proj
        proj2 = torch.mm(features2, theta)  # B x num_proj
        
        # Sort projections
        proj1_sorted, _ = torch.sort(proj1, dim=0)
        proj2_sorted, _ = torch.sort(proj2, dim=0)
        
        # Compute L2 Wasserstein distance for each projection
        w_distance = torch.mean((proj1_sorted - proj2_sorted)**2, dim=0)
        
        return torch.mean(w_distance)
    
    loss = 0.0
    
    if pt_image_features is not None:
        loss += compute_sw_distance(ft_image_features, pt_image_features, num_projections)
    
    if pt_text_features is not None:
        loss += compute_sw_distance(ft_text_features, pt_text_features, num_projections)
        
    return loss


def stiefel_regularization(ft_image_features, ft_text_features,
                         projection_matrix=None):
    """
    Stiefel manifold constraint for orthogonal projections.
    Note: This is typically applied to the projection matrix W, not features.
    Here we demonstrate feature orthogonality regularization.
    
    Args:
        ft_image_features: Fine-tuned image features (B x D)
        ft_text_features: Fine-tuned text features (B x D)
        projection_matrix: Optional projection matrix to regularize
    
    Returns:
        Orthogonality loss
    """
    if projection_matrix is not None:
        # Standard Stiefel regularization for projection matrix
        gram = torch.mm(projection_matrix.t(), projection_matrix)
        I = torch.eye(gram.size(0), device=gram.device)
        loss = torch.norm(gram - I, p='fro')**2
        return loss
    
    # Alternative: encourage feature diversity through orthogonality
    def feature_orthogonality_loss(features):
        # Compute Gram matrix
        gram = torch.mm(features, features.t())
        # Remove diagonal (self-similarity)
        mask = 1 - torch.eye(gram.size(0), device=gram.device)
        gram_off_diag = gram * mask
        # Penalize high off-diagonal values
        return torch.mean(gram_off_diag**2)
    
    loss = feature_orthogonality_loss(ft_image_features)
    loss += feature_orthogonality_loss(ft_text_features)
    
    return loss


def clip_similarity_preservation(ft_image_features, ft_text_features,
                               pt_image_features=None, pt_text_features=None,
                               temperature=0.07):
    """
    Preserve CLIP's image-text similarity structure.
    
    Args:
        ft_image_features: Fine-tuned image features (B x D)
        ft_text_features: Fine-tuned text features (B x D)
        pt_image_features: Pre-trained image features (B x D)
        pt_text_features: Pre-trained text features (B x D)
        temperature: CLIP temperature parameter
    
    Returns:
        Similarity preservation loss
    """
    if pt_image_features is None or pt_text_features is None:
        # Compute current similarity matrix
        logits = torch.mm(ft_image_features, ft_text_features.t()) / temperature
        # Encourage diagonal (matching pairs)
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels)
        return loss
    
    # Compute similarity matrices
    ft_logits = torch.mm(ft_image_features, ft_text_features.t()) / temperature
    pt_logits = torch.mm(pt_image_features, pt_text_features.t()) / temperature
    
    # KL divergence between similarity distributions
    ft_probs = F.softmax(ft_logits, dim=1)
    pt_probs = F.softmax(pt_logits, dim=1)
    
    loss = F.kl_div(ft_probs.log(), pt_probs, reduction='batchmean')
    
    return loss


def combined_geometric_regularization(ft_image_features, ft_text_features,
                                    pt_image_features=None, pt_text_features=None,
                                    lambda_geo=1.0, lambda_spectral=1.0,
                                    lambda_vmf=1.0, lambda_grass=0.1,
                                    lambda_ot=0.1, lambda_stiefel=0.1,
                                    lambda_clip=1.0):
    """
    Combined geometric continual learning loss.
    
    Args:
        ft_image_features: Fine-tuned image features (B x D)
        ft_text_features: Fine-tuned text features (B x D)
        pt_image_features: Pre-trained image features (B x D)
        pt_text_features: Pre-trained text features (B x D)
        lambda_*: Weight for each regularization term
    
    Returns:
        Total regularization loss
    """
    total_loss = 0.0
    
    # Geodesic regularization
    if lambda_geo > 0 and pt_image_features is not None:
        total_loss += lambda_geo * geodesic_regularization(
            ft_image_features, ft_text_features,
            pt_image_features, pt_text_features
        )
    
    # Spectral regularization
    if lambda_spectral > 0:
        total_loss += lambda_spectral * spectral_regularization_logdet(
            ft_image_features, ft_text_features
        )
    
    # vMF entropy regularization
    if lambda_vmf > 0:
        total_loss += lambda_vmf * vmf_entropy_regularization(
            ft_image_features, ft_text_features
        )
    
    # Grassmannian alignment
    if lambda_grass > 0 and pt_image_features is not None:
        total_loss += lambda_grass * grassmannian_subspace_alignment(
            ft_image_features, ft_text_features,
            pt_image_features, pt_text_features
        )
    
    # Optimal transport
    if lambda_ot > 0 and pt_image_features is not None:
        total_loss += lambda_ot * sliced_wasserstein_sphere(
            ft_image_features, ft_text_features,
            pt_image_features, pt_text_features
        )
    
    # Stiefel regularization
    if lambda_stiefel > 0:
        total_loss += lambda_stiefel * stiefel_regularization(
            ft_image_features, ft_text_features
        )
    
    # CLIP similarity preservation
    if lambda_clip > 0:
        total_loss += lambda_clip * clip_similarity_preservation(
            ft_image_features, ft_text_features,
            pt_image_features, pt_text_features
        )
    
    return total_loss


# Example usage
if __name__ == "__main__":
    # Simulate normalized CLIP features
    batch_size = 128
    feature_dim = 512
    
    # Generate random normalized features
    ft_img = F.normalize(torch.randn(batch_size, feature_dim), p=2, dim=1)
    ft_txt = F.normalize(torch.randn(batch_size, feature_dim), p=2, dim=1)
    pt_img = F.normalize(torch.randn(batch_size, feature_dim), p=2, dim=1)
    pt_txt = F.normalize(torch.randn(batch_size, feature_dim), p=2, dim=1)
    
    # Compute individual regularization terms
    print("Geodesic loss:", geodesic_regularization(ft_img, ft_txt, pt_img, pt_txt).item())
    print("Spectral loss:", spectral_regularization_logdet(ft_img, ft_txt).item())
    print("vMF loss:", vmf_entropy_regularization(ft_img, ft_txt).item())
    print("Grassmann loss:", grassmannian_subspace_alignment(ft_img, ft_txt, pt_img, pt_txt).item())
    print("Wasserstein loss:", sliced_wasserstein_sphere(ft_img, ft_txt, pt_img, pt_txt).item())
    print("Stiefel loss:", stiefel_regularization(ft_img, ft_txt).item())
    print("CLIP similarity loss:", clip_similarity_preservation(ft_img, ft_txt, pt_img, pt_txt).item())
    
    # Combined loss
    total_loss = combined_geometric_regularization(ft_img, ft_txt, pt_img, pt_txt)
    print("\nTotal combined loss:", total_loss.item())