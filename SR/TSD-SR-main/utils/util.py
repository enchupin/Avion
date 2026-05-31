import torch
from diffusers import SD3Transformer2DModel
from peft import LoraConfig

def show_param(model ,static_dict, print_param=False):
    for key in static_dict:
        frozen_layers = list(filter(lambda param : key in param[0], model.named_parameters()))
        for name, param in frozen_layers:
            print(name,param.requires_grad)
            if print_param:
                print(param, param.dtype)
                print(param.grad)
            else:
                print(param[0])

def load_lora_state_dict(state_dict, model, adapter_name="default", module_prefixes=("transformer", "unet", "vae")):
    state_dict = dict(state_dict)
    for n, p in model.named_parameters():
        if adapter_name not in n:
            continue

        base_name = n.replace(f".{adapter_name}", "")
        candidate_names = [f"{prefix}.{base_name}" for prefix in module_prefixes] + [base_name]
        for candidate_name in candidate_names:
            if candidate_name in state_dict:
                p.data.copy_(state_dict.pop(candidate_name))
                break

    if len(state_dict) > 0:
        print(f"Warning: {len(state_dict)} keys not loaded")
        print(state_dict.keys())
        
def get_trainable_param(model):
    train_param = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            train_param.append(n)
    return train_param
                
