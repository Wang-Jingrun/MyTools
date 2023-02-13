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
└─tools # tools for dataset preprocessing
|  │  get_wav.py
|  │  resample.py
|  |  generate_dataset.py
|  │  mat2wav.py
|  |  flac2wav.py
|  └─ sphfile2wav.py
|   
└─trainer # The core code is located in the trainer folder.
   |  ComplexNN # Implementation of complex neural network.
   |    │  CBatchNorm2d.py
   |    │  CConv2d.py
   |    │  CConvTranspose2d.py
   |	└─ __init__.py
   |
   │  dataLoader.py
   │  dcunet.py	# main models
   │  trainer.py
   │  utils.py # loss function, metrics and so on
   └─ __init__.py
```

#### Dataset
```
!wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/2791/clean_trainset_28spk_wav.zip?sequence=2&isAllowed=y
!wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/2791/noisy_trainset_28spk_wav.zip?sequence=6&isAllowed=y

!wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip?sequence=1&isAllowed=y
!wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip?sequence=5&isAllowed=y


!unzip clean_trainset_28spk_wav.zip?sequence=2.3
!unzip noisy_trainset_28spk_wav.zip?sequence=6
!unzip clean_testset_wav.zip?sequence=1
!unzip nnoisy_testset_wav.zip?sequence=5
!ls
```

