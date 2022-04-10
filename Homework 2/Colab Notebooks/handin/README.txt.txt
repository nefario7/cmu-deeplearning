The final model architecture and configuration below, that gave the best result can be found in model_config.yaml.
config = {
    '': 'ConvNext',
    'batch_size': 256,
    'transforms': True,
    'epochs': 50,
    'backbone': 'convnext',             # convnext, mobilenet, resnet
    'dropout': None,
    'optimizer': 'SGD',                 # SGD, Adam, AdamW
    'loss': 'CrossEL',                  # CrossEL, ArcFace, CosFace, Triplet
    'scheduler': 'ReduceLRonPlateau',   # CosineAnnealingLR, ReduceLRonPlateau
    'optim': {'lr': 0.1, 'momentum':0.9, 'weight_decay':1e-4, 'nesterov':True},
    'subset': False,
    'save': True,
    'log': True,
    'randomize': False,
}

To reach the best model config, I performed various experiments: 
- Backbone: Implemented and tested ResNet, ConvNext, MobileNet
- Optimizers: SGD, Adam, AdamW
- Schedulers: CosineAnnealingLR, ReduceLRonPlateau
- Loss Function: CrossEL
- Different combinations of data augmentation transformations
- Total ablation experiments : 18