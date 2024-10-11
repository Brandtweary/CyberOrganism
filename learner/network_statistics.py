import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

class NetworkStatistics:
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.stats = defaultdict(list)
        self.last_gradients = None
        self.record_stats = {
            'gradient_stats': {'record': False, 'stat_names': []},
            'weight_update_stats': {'record': False, 'stat_names': []},
            'activation_stats': {'record': False, 'stat_names': []},
            'loss_landscape': {'record': False, 'stat_names': []},
            'gradient_flow': {'record': False, 'stat_names': []},
            'learning_rate_dynamics': {'record': False, 'stat_names': []},
            'batch_statistics': {'record': False, 'stat_names': []},
            'model_capacity_utilization': {'record': False, 'stat_names': []},
            'gradient_predictiveness': {'record': False, 'stat_names': []},
            'layerwise_relevance': {'record': False, 'stat_names': []},
        }
        self.set_stat_names()

    def set_stat_names(self):
        if not self.model:
            return
        for name, param in self.model.named_parameters():
            self.record_stats['gradient_stats']['stat_names'].extend([
                f'{name}_grad_mean',
                f'{name}_grad_max',
                f'{name}_grad_hist'
            ])
            self.record_stats['weight_update_stats']['stat_names'].extend([
                f'{name}_weight_update_mean',
                f'{name}_weight_update_max',
                f'{name}_weight_update_ratio'
            ])
            self.record_stats['activation_stats']['stat_names'].extend([
                f'{name}_act_mean',
                f'{name}_act_var',
                f'{name}_act_hist'
            ])
            self.record_stats['learning_rate_dynamics']['stat_names'].append(
                f'{name}_update_to_grad_ratio'
            )

        self.record_stats['loss_landscape']['stat_names'] = ['loss', 'loss_change']
        self.record_stats['gradient_flow']['stat_names'] = ['gradient_flow']
        self.record_stats['learning_rate_dynamics']['stat_names'].append('learning_rate')
        self.record_stats['batch_statistics']['stat_names'] = ['input_mean', 'input_var']
        self.record_stats['model_capacity_utilization']['stat_names'] = ['capacity_utilization']
        self.record_stats['gradient_predictiveness']['stat_names'] = ['gradient_predictiveness']
        self.record_stats['layerwise_relevance']['stat_names'] = ['layerwise_relevance']

    def update(self, loss, inputs):
        if not self.model:
            return
        if self.record_stats['gradient_stats']['record']:
            self.record_gradient_stats()
        if self.record_stats['weight_update_stats']['record']:
            self.record_weight_update_stats()
        if self.record_stats['activation_stats']['record']:
            self.record_activation_stats(inputs)
        if self.record_stats['loss_landscape']['record']:
            self.record_loss_landscape(loss)
        if self.record_stats['gradient_flow']['record']:
            self.record_gradient_flow()
        if self.record_stats['learning_rate_dynamics']['record']:
            self.record_learning_rate_dynamics()
        if self.record_stats['batch_statistics']['record']:
            self.record_batch_statistics(inputs)
        if self.record_stats['model_capacity_utilization']['record']:
            self.record_model_capacity_utilization()
        if self.record_stats['gradient_predictiveness']['record']:
            self.record_gradient_predictiveness()
        if self.record_stats['layerwise_relevance']['record']:
            self.record_layerwise_relevance()

    def record_gradient_stats(self):
        param_count = 0
        for name, param in self.model.named_parameters():
            param_count += 1
            if param.grad is not None:
                grad = param.grad.data
                self.stats[f'{name}_grad_mean'].append(grad.mean().item())
                self.stats[f'{name}_grad_max'].append(grad.abs().max().item())
                self.stats[f'{name}_grad_hist'].append(grad.histc(bins=10).cpu().numpy())

    def record_weight_update_stats(self):
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                weight = param.data
                update = -self.optimizer.param_groups[0]['lr'] * param.grad.data
                self.stats[f'{name}_weight_update_mean'].append(update.mean().item())
                self.stats[f'{name}_weight_update_max'].append(update.abs().max().item())
                self.stats[f'{name}_weight_update_ratio'].append((update.abs() / (weight.abs() + 1e-8)).mean().item())

    def record_activation_stats(self, inputs):
        activations = {}
        def hook(name):
            def fn(_, __, output):
                activations[name] = output
            return fn

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                module.register_forward_hook(hook(name))

        _ = self.model(inputs)

        for name, act in activations.items():
            self.stats[f'{name}_act_mean'].append(act.mean().item())
            self.stats[f'{name}_act_var'].append(act.var().item())
            self.stats[f'{name}_act_hist'].append(act.histc(bins=10).cpu().numpy())

    def record_loss_landscape(self, loss):
        self.stats['loss'].append(loss.item())
        if len(self.stats['loss']) > 1:
            self.stats['loss_change'].append(self.stats['loss'][-1] - self.stats['loss'][-2])

    def record_gradient_flow(self):
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        self.stats['gradient_flow'].append(total_norm)

    def record_learning_rate_dynamics(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.stats['learning_rate'].append(lr)
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                update_to_grad_ratio = (lr * param.grad.data.norm() / (param.data.norm() + 1e-8)).item()
                self.stats[f'{name}_update_to_grad_ratio'].append(update_to_grad_ratio)

    def record_batch_statistics(self, inputs):
        self.stats['input_mean'].append(inputs.mean().item())
        self.stats['input_var'].append(inputs.var().item())

    def record_model_capacity_utilization(self):
        total_params = 0
        near_zero_params = 0
        for param in self.model.parameters():
            total_params += param.numel()
            near_zero_params += (param.abs() < 1e-3).sum().item()
        self.stats['capacity_utilization'].append(1 - near_zero_params / total_params)

    def record_gradient_predictiveness(self):
        current_gradients = [p.grad.data.view(-1) for p in self.model.parameters() if p.grad is not None]
        if not current_gradients:
            return
        
        current_gradients = torch.cat(current_gradients)
        
        if self.last_gradients is not None:
            cosine_similarity = nn.functional.cosine_similarity(current_gradients, self.last_gradients, dim=0)
            self.stats['gradient_predictiveness'].append(cosine_similarity.item())
        
        self.last_gradients = current_gradients

    def record_layerwise_relevance(self):
        # This is a complex topic and might require a separate implementation
        # For now, we'll just add a placeholder
        self.stats['layerwise_relevance'].append(None)

    def get_stats(self):
        return dict(self.stats)