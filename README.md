#### Structure of the template

```
autocut
│  .gitignore
│  README.md
│  main.py
│
└─dataset
│
└─output
|
└─config
|  └─ train_config.yaml # parameters
|   
└─trainer # The core code is located in the trainer folder.
   |   ComplexNN # Implementation of complex neural network.
   |     │  CBatchNorm2d.py
   |     │  CConv2d.py
   |     │  CConvTranspose2d.py
   |	 └─ __init__.py
   |
   │  dataLoader.py
   │  dcunet.py	# main models
   │  trainer.py
   │  utils.py # loss function, metrics and so on
   └─ __init__.py
```