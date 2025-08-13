import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPKnowledgeDistillation(nn.Module):
    """
    Comprehensive CLIP Knowledge Distillation module implementing multiple distillation methods.
    
    Methods implemented:
    1. Contrastive Relational Distillation (CRD)
    2. Feature Distillation (FD) 
    3. Masked Feature Distillation (MFD)
    4. Gradient Distillation (GD)
    5. Interactive Contrastive Learning (ICL)
    6. Augmented Feature Distillation (AFD)
    7. Cross Knowledge Distillation (Cross KD)
    """
    
    def __init__(self, args, embed_dim=512):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        
        # Initialize fusion projections for AFD
        if args.alpha_afd > 0:
            self.visual_fusion_proj = nn.Linear(embed_dim * 2, embed_dim)
            self.text_fusion_proj = nn.Linear(embed_dim * 2, embed_dim)
        
        # Loss functions
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        
    def normalize_features(self, features):
        """Normalize features to unit length."""
        return F.normalize(features, dim=-1)
    
    def get_grad(self, image_features, text_features, logit_scale, labels):
        """
        Compute gradients for Gradient Distillation.
        Returns gradients of CLIP loss w.r.t. image and text features.
        """
        image_features = image_features.clone().detach().requires_grad_(True)
        text_features = text_features.clone().detach().requires_grad_(True)
        
        # Compute logits
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T
        
        # Compute CLIP loss
        clip_loss = (self.cross_entropy(logits_per_image, labels) + 
                    self.cross_entropy(logits_per_text, labels)) / 2
        
        # Compute gradients
        grad_img = torch.autograd.grad(clip_loss, image_features, 
                                     retain_graph=True, create_graph=False)[0]
        grad_txt = torch.autograd.grad(clip_loss, text_features, 
                                     retain_graph=True, create_graph=False)[0]
        
        return grad_img, grad_txt
    
    def contrastive_relational_distillation(self, student_logits_img, student_logits_txt,
                                          teacher_logits_img, teacher_logits_txt):
        """
        Contrastive Relational Distillation (CRD).
        Aligns contrastive distributions between teacher and student using KL divergence.
        """
        # Convert logits to log probabilities for KL divergence
        student_log_probs_img = F.log_softmax(student_logits_img, dim=1)
        student_log_probs_txt = F.log_softmax(student_logits_txt, dim=1)
        
        teacher_probs_img = F.softmax(teacher_logits_img, dim=1)
        teacher_probs_txt = F.softmax(teacher_logits_txt, dim=1)
        
        # KL divergence loss
        crd_loss_img = self.kl_loss(student_log_probs_img, teacher_probs_img)
        crd_loss_txt = self.kl_loss(student_log_probs_txt, teacher_probs_txt)
        
        return (crd_loss_img + crd_loss_txt) / 2
    
    def feature_distillation(self, student_img_features, student_txt_features,
                           teacher_img_features, teacher_txt_features):
        """
        Feature Distillation (FD).
        Direct alignment of visual and text embeddings using MSE loss.
        """
        # Normalize features
        student_img_norm = self.normalize_features(student_img_features)
        student_txt_norm = self.normalize_features(student_txt_features)
        teacher_img_norm = self.normalize_features(teacher_img_features)
        teacher_txt_norm = self.normalize_features(teacher_txt_features)
        
        # Since teacher and student have same dimensions (moving average), no alignment needed
        # MSE loss between normalized features
        fd_loss_img = self.mse_loss(student_img_norm, teacher_img_norm)
        fd_loss_txt = self.mse_loss(student_txt_norm, teacher_txt_norm)
        
        return fd_loss_img + fd_loss_txt
    
    def masked_feature_distillation(self, student_img_features, student_txt_features,
                                  teacher_img_features, teacher_txt_features,
                                  mask_ratio=0.75):
        """
        Masked Feature Distillation (MFD).
        Similar to FD but uses masked images as input to student model.
        Note: Masking should be applied during data preprocessing.
        """
        # For now, this is similar to FD but would use masked inputs
        # The masking logic would be applied in the data preprocessing stage
        return self.feature_distillation(student_img_features, student_txt_features,
                                       teacher_img_features, teacher_txt_features)
    
    def gradient_distillation(self, student_img_features, student_txt_features,
                            teacher_img_features, teacher_txt_features,
                            student_logit_scale, teacher_logit_scale, labels):
        """
        Gradient Distillation (GD).
        Aligns gradient information between teacher and student.
        """
        # Compute teacher gradients (detached)
        with torch.no_grad():
            t_grad_img, t_grad_txt = self.get_grad(teacher_img_features, teacher_txt_features,
                                                 teacher_logit_scale, labels)
        
        # Compute student gradients
        s_grad_img, s_grad_txt = self.get_grad(student_img_features, student_txt_features,
                                             student_logit_scale, labels)
        
        # MSE loss between gradients
        gd_loss_img = self.mse_loss(s_grad_img, t_grad_img.detach())
        gd_loss_txt = self.mse_loss(s_grad_txt, t_grad_txt.detach())
        
        return gd_loss_img + gd_loss_txt
    
    def interactive_contrastive_learning(self, student_img_features, student_txt_features,
                                       teacher_img_features, teacher_txt_features,
                                       logit_scale, labels):
        """
        Interactive Contrastive Learning (ICL).
        Cross-modal contrastive learning where student embeddings contrast with teacher embeddings.
        """
        # Normalize features
        student_img_norm = self.normalize_features(student_img_features)
        student_txt_norm = self.normalize_features(student_txt_features)
        teacher_img_norm = self.normalize_features(teacher_img_features)
        teacher_txt_norm = self.normalize_features(teacher_txt_features)
        
        # Cross-modal logits: student to teacher
        logits_s_img_to_t_txt = logit_scale * student_img_norm @ teacher_txt_norm.T
        logits_s_txt_to_t_img = logit_scale * student_txt_norm @ teacher_img_norm.T
        
        # ICL loss
        icl_loss_img = self.cross_entropy(logits_s_img_to_t_txt, labels)
        icl_loss_txt = self.cross_entropy(logits_s_txt_to_t_img, labels)
        
        return (icl_loss_img + icl_loss_txt) / 2
    
    def augmented_feature_distillation(self, student_img_features, student_txt_features,
                                     teacher_img_features, teacher_txt_features,
                                     logit_scale, labels):
        """
        Augmented Feature Distillation (AFD).
        Concatenates student and teacher embeddings, then applies linear fusion encoders.
        """
        # Normalize features
        student_img_norm = self.normalize_features(student_img_features)
        student_txt_norm = self.normalize_features(student_txt_features)
        teacher_img_norm = self.normalize_features(teacher_img_features)
        teacher_txt_norm = self.normalize_features(teacher_txt_features)
        
        # Concatenate student and teacher features
        img_fusion_feat = torch.cat([student_img_norm, teacher_img_norm], dim=1)
        txt_fusion_feat = torch.cat([student_txt_norm, teacher_txt_norm], dim=1)
        
        # Apply fusion projections
        img_fusion_feat = self.visual_fusion_proj(img_fusion_feat)
        txt_fusion_feat = self.text_fusion_proj(txt_fusion_feat)
        
        # Compute augmented CLIP loss
        logits_per_image = logit_scale * img_fusion_feat @ txt_fusion_feat.T
        logits_per_text = logits_per_image.T
        
        afd_loss_img = self.cross_entropy(logits_per_image, labels)
        afd_loss_txt = self.cross_entropy(logits_per_text, labels)
        
        return (afd_loss_img + afd_loss_txt) / 2
    
    def cross_knowledge_distillation(self, student_img_features, student_txt_features,
                                   teacher_img_features, teacher_txt_features,
                                   teacher_logits_img, teacher_logits_txt,
                                   logit_scale):
        """
        Cross Knowledge Distillation (Cross KD).
        Uses cross-modal teacher-student interactions aligned with teacher's same-modal logits.
        """
        # Normalize features
        student_img_norm = self.normalize_features(student_img_features)
        student_txt_norm = self.normalize_features(student_txt_features)
        teacher_img_norm = self.normalize_features(teacher_img_features)
        teacher_txt_norm = self.normalize_features(teacher_txt_features)
        
        # Cross-modal student-teacher logits
        logits_s_img_to_t_txt = logit_scale * student_img_norm @ teacher_txt_norm.T
        logits_s_txt_to_t_img = logit_scale * student_txt_norm @ teacher_img_norm.T
        
        # Convert to log probabilities
        student_log_probs_img = F.log_softmax(logits_s_img_to_t_txt, dim=1)
        student_log_probs_txt = F.log_softmax(logits_s_txt_to_t_img, dim=1)
        
        teacher_probs_img = F.softmax(teacher_logits_img, dim=1)
        teacher_probs_txt = F.softmax(teacher_logits_txt, dim=1)
        
        # Cross KD loss
        cross_kd_loss_img = self.kl_loss(student_log_probs_img, teacher_probs_img)
        cross_kd_loss_txt = self.kl_loss(student_log_probs_txt, teacher_probs_txt)
        
        return (cross_kd_loss_img + cross_kd_loss_txt) / 2
    
    def compute_temperature_scaled_distillation(self, student_logits_img, student_logits_txt,
                                              teacher_logits_img, teacher_logits_txt,
                                              temperature=1.0):
        """
        Compute temperature-scaled distillation loss (similar to current implementation).
        """
        # Scale teacher logits by temperature
        teacher_logits_img_scaled = teacher_logits_img / temperature
        teacher_logits_txt_scaled = teacher_logits_txt / temperature
        
        # Convert to probabilities
        teacher_probs_img = F.softmax(teacher_logits_img_scaled, dim=1)
        teacher_probs_txt = F.softmax(teacher_logits_txt_scaled, dim=1)
        
        student_log_probs_img = F.log_softmax(student_logits_img, dim=1)
        student_log_probs_txt = F.log_softmax(student_logits_txt, dim=1)
        
        # KL divergence loss
        kl_loss_img = self.kl_loss(student_log_probs_img, teacher_probs_img)
        kl_loss_txt = self.kl_loss(student_log_probs_txt, teacher_probs_txt)
        
        return (kl_loss_img + kl_loss_txt) / 2
    
    def forward(self, student_img_features, student_txt_features,
                teacher_img_features, teacher_txt_features,
                student_logit_scale, teacher_logit_scale,
                labels, temperature=1.0):
        """
        Forward pass computing all enabled distillation losses.
        
        Args:
            student_img_features: Student image features
            student_txt_features: Student text features  
            teacher_img_features: Teacher image features
            teacher_txt_features: Teacher text features
            student_logit_scale: Student logit scale
            teacher_logit_scale: Teacher logit scale
            labels: Ground truth labels
            temperature: Temperature for scaling
            
        Returns:
            Dictionary of individual losses and total distillation loss
        """
        losses = {}
        total_loss = 0.0
        
        # Compute teacher logits
        teacher_logits_img = teacher_logit_scale * teacher_img_features @ teacher_txt_features.T
        teacher_logits_txt = teacher_logits_img.T
        
        # Compute student logits
        student_logits_img = student_logit_scale * student_img_features @ student_txt_features.T
        student_logits_txt = student_logits_img.T
        
        # 1. Contrastive Relational Distillation (CRD)
        if self.args.alpha_crd > 0:
            crd_loss = self.contrastive_relational_distillation(
                student_logits_img, student_logits_txt,
                teacher_logits_img, teacher_logits_txt
            )
            losses['crd_loss'] = crd_loss
            total_loss += self.args.alpha_crd * crd_loss
        
        # 2. Feature Distillation (FD)
        if self.args.alpha_fd > 0:
            fd_loss = self.feature_distillation(
                student_img_features, student_txt_features,
                teacher_img_features, teacher_txt_features
            )
            losses['fd_loss'] = fd_loss
            total_loss += self.args.alpha_fd * fd_loss
        
        # 3. Masked Feature Distillation (MFD)
        if self.args.alpha_mfd > 0:
            mfd_loss = self.masked_feature_distillation(
                student_img_features, student_txt_features,
                teacher_img_features, teacher_txt_features
            )
            losses['mfd_loss'] = mfd_loss
            total_loss += self.args.alpha_mfd * mfd_loss
        
        # 4. Gradient Distillation (GD)
        if self.args.alpha_gd > 0:
            gd_loss = self.gradient_distillation(
                student_img_features, student_txt_features,
                teacher_img_features, teacher_txt_features,
                student_logit_scale, teacher_logit_scale, labels
            )
            losses['gd_loss'] = gd_loss
            total_loss += self.args.alpha_gd * gd_loss
        
        # 5. Interactive Contrastive Learning (ICL)
        if self.args.alpha_icl > 0:
            icl_loss = self.interactive_contrastive_learning(
                student_img_features, student_txt_features,
                teacher_img_features, teacher_txt_features,
                student_logit_scale, labels
            )
            losses['icl_loss'] = icl_loss
            total_loss += self.args.alpha_icl * icl_loss
        
        # 6. Augmented Feature Distillation (AFD)
        if self.args.alpha_afd > 0:
            afd_loss = self.augmented_feature_distillation(
                student_img_features, student_txt_features,
                teacher_img_features, teacher_txt_features,
                student_logit_scale, labels
            )
            losses['afd_loss'] = afd_loss
            total_loss += self.args.alpha_afd * afd_loss
        
        # 7. Cross Knowledge Distillation (Cross KD)
        if self.args.alpha_cross_kd > 0:
            cross_kd_loss = self.cross_knowledge_distillation(
                student_img_features, student_txt_features,
                teacher_img_features, teacher_txt_features,
                teacher_logits_img, teacher_logits_txt,
                student_logit_scale
            )
            losses['cross_kd_loss'] = cross_kd_loss
            total_loss += self.args.alpha_cross_kd * cross_kd_loss
        
        # 8. Temperature-scaled distillation (current implementation)
        if self.args.alpha_temp_distil > 0:
            temp_distil_loss = self.compute_temperature_scaled_distillation(
                student_logits_img, student_logits_txt,
                teacher_logits_img, teacher_logits_txt,
                temperature
            )
            losses['temp_distil_loss'] = temp_distil_loss
            total_loss += self.args.alpha_temp_distil * temp_distil_loss
        
        losses['total_kd_loss'] = total_loss
        return losses


def create_clip_kd_module(args, embed_dim=512):
    """
    Factory function to create CLIP Knowledge Distillation module.
    Since teacher is a moving average of student, they have identical embedding dimensions.
    """
    return CLIPKnowledgeDistillation(args, embed_dim) 


def calculate_teacher_statistics(teacher_logits_img, teacher_logits_text, 
                                student_logits_img, student_logits_text):
    """
    Calculate comprehensive statistics about teacher model predictions for analysis.
    
    Args:
        teacher_logits_img: Teacher logits for image-to-text matching
        teacher_logits_text: Teacher logits for text-to-image matching  
        student_logits_img: Student logits for image-to-text matching
        student_logits_text: Student logits for text-to-image matching
    
    Returns:
        Dictionary containing various teacher statistics
    """
    stats = {}
    
    # Convert logits to probabilities
    teacher_probs_img = F.softmax(teacher_logits_img, dim=1)
    teacher_probs_text = F.softmax(teacher_logits_text, dim=1)
    student_probs_img = F.softmax(student_logits_img, dim=1)
    student_probs_text = F.softmax(student_logits_text, dim=1)
    
    # 1. Entropy calculations (measure of uncertainty)
    def entropy(probs):
        return -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    
    teacher_entropy_img = entropy(teacher_probs_img)
    teacher_entropy_text = entropy(teacher_probs_text)
    
    stats['teacher_avg_entropy_img'] = teacher_entropy_img.mean().item()
    stats['teacher_avg_entropy_text'] = teacher_entropy_text.mean().item()
    stats['teacher_avg_entropy_combined'] = (teacher_entropy_img.mean() + teacher_entropy_text.mean()).item() / 2
    
    # 2. Confidence calculations (max probability)
    teacher_confidence_img = torch.max(teacher_probs_img, dim=1)[0]
    teacher_confidence_text = torch.max(teacher_probs_text, dim=1)[0]
    
    stats['teacher_avg_confidence_img'] = teacher_confidence_img.mean().item()
    stats['teacher_avg_confidence_text'] = teacher_confidence_text.mean().item()
    stats['teacher_avg_confidence_combined'] = (teacher_confidence_img.mean() + teacher_confidence_text.mean()).item() / 2
    
    # 3. Prediction agreement/disagreement with student
    teacher_pred_img = torch.argmax(teacher_probs_img, dim=1)
    teacher_pred_text = torch.argmax(teacher_probs_text, dim=1)
    student_pred_img = torch.argmax(student_probs_img, dim=1)
    student_pred_text = torch.argmax(student_probs_text, dim=1)
    
    agreement_img = (teacher_pred_img == student_pred_img).float().mean().item()
    agreement_text = (teacher_pred_text == student_pred_text).float().mean().item()
    
    stats['teacher_student_agreement_img'] = agreement_img
    stats['teacher_student_agreement_text'] = agreement_text
    stats['teacher_student_agreement_combined'] = (agreement_img + agreement_text) / 2
    
    # 4. KL divergence between teacher and student (asymmetric)
    kl_img = F.kl_div(torch.log(student_probs_img + 1e-8), teacher_probs_img, reduction='batchmean')
    kl_text = F.kl_div(torch.log(student_probs_text + 1e-8), teacher_probs_text, reduction='batchmean')
    
    stats['teacher_student_kl_img'] = kl_img.item()
    stats['teacher_student_kl_text'] = kl_text.item()
    stats['teacher_student_kl_combined'] = (kl_img + kl_text).item() / 2
    
    # 5. Top-k accuracy alignment
    def top_k_accuracy_alignment(teacher_logits, student_logits, k=5):
        teacher_topk = torch.topk(teacher_logits, k, dim=1)[1]
        student_topk = torch.topk(student_logits, k, dim=1)[1]
        
        # Calculate how many of teacher's top-k are in student's top-k
        overlap = 0
        for i in range(teacher_topk.size(0)):
            teacher_set = set(teacher_topk[i].cpu().tolist())
            student_set = set(student_topk[i].cpu().tolist())
            overlap += len(teacher_set.intersection(student_set))
        
        return overlap / (teacher_topk.size(0) * k)
    
    stats['teacher_student_top5_alignment_img'] = top_k_accuracy_alignment(teacher_logits_img, student_logits_img, k=5)
    stats['teacher_student_top5_alignment_text'] = top_k_accuracy_alignment(teacher_logits_text, student_logits_text, k=5)
    
    # 6. Prediction diversity (how spread out are the predictions)
    def prediction_diversity(probs):
        # Calculate the effective number of classes (inverse of Gini coefficient)
        sorted_probs = torch.sort(probs, dim=1, descending=True)[0]
        cumsum = torch.cumsum(sorted_probs, dim=1)
        return (1 - torch.sum(sorted_probs * (2 * cumsum - sorted_probs - 1), dim=1)).mean()
    
    stats['teacher_prediction_diversity_img'] = prediction_diversity(teacher_probs_img).item()
    stats['teacher_prediction_diversity_text'] = prediction_diversity(teacher_probs_text).item()
    
    # 7. Correctness calculation
    # For contrastive learning, correct prediction is when image matches its corresponding text (diagonal)
    batch_size = teacher_logits_img.size(0)
    correct_indices = torch.arange(batch_size).cuda()
    
    teacher_correct_img = (teacher_pred_img == correct_indices).float().mean().item()
    teacher_correct_text = (teacher_pred_text == correct_indices).float().mean().item()
    student_correct_img = (student_pred_img == correct_indices).float().mean().item()
    student_correct_text = (student_pred_text == correct_indices).float().mean().item()
    
    stats['teacher_accuracy_img'] = teacher_correct_img
    stats['teacher_accuracy_text'] = teacher_correct_text
    stats['teacher_accuracy_combined'] = (teacher_correct_img + teacher_correct_text) / 2
    
    stats['student_accuracy_img'] = student_correct_img
    stats['student_accuracy_text'] = student_correct_text
    stats['student_accuracy_combined'] = (student_correct_img + student_correct_text) / 2
    
    # Teacher vs student accuracy difference
    stats['teacher_student_accuracy_diff_img'] = teacher_correct_img - student_correct_img
    stats['teacher_student_accuracy_diff_text'] = teacher_correct_text - student_correct_text
    stats['teacher_student_accuracy_diff_combined'] = stats['teacher_accuracy_combined'] - stats['student_accuracy_combined']
    
    # 8. Ground truth probability analysis (how much probability mass is on the diagonal)
    teacher_gt_probs_img = torch.diag(teacher_probs_img).mean().item()
    teacher_gt_probs_text = torch.diag(teacher_probs_text).mean().item()
    student_gt_probs_img = torch.diag(student_probs_img).mean().item()
    student_gt_probs_text = torch.diag(student_probs_text).mean().item()
    
    stats['teacher_gt_prob_img'] = teacher_gt_probs_img
    stats['teacher_gt_prob_text'] = teacher_gt_probs_text
    stats['teacher_gt_prob_combined'] = (teacher_gt_probs_img + teacher_gt_probs_text) / 2
    
    stats['student_gt_prob_img'] = student_gt_probs_img
    stats['student_gt_prob_text'] = student_gt_probs_text
    stats['student_gt_prob_combined'] = (student_gt_probs_img + student_gt_probs_text) / 2
    
    # 9. Temperature analysis (effective temperature of the teacher)
    def effective_temperature(logits):
        # Estimate temperature by fitting to make entropy match uniform distribution
        target_entropy = torch.log(torch.tensor(logits.size(1), dtype=torch.float32))
        current_entropy = entropy(F.softmax(logits, dim=1)).mean()
        return (target_entropy / current_entropy).item()
    
    stats['teacher_effective_temperature_img'] = effective_temperature(teacher_logits_img)
    stats['teacher_effective_temperature_text'] = effective_temperature(teacher_logits_text)
    
    return stats

