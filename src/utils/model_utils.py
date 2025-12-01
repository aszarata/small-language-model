import torch
from ..model.model_config import LanguageModelConfig
from ..model.transformer_model import LanguageModel
from ..model.classifier import Classifier

def load_model(model_dir, config_file=None, model_file=None):
    if config_file is None:
        config_file = "config.json"
    if model_file is None:
        model_file = 'model.pt'
    
    config_path = model_dir + "/" + config_file
    model_path = model_dir + "/" + model_file

    config = LanguageModelConfig.load(config_path)

    if config.n_classes != None:
        model = Classifier(3, config=config)
    else:
        model = LanguageModel(config=config)
    model.load_state_dict(torch.load(model_path))

    return model