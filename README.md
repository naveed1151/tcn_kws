# tcn_kws
mchines can hear us
tcn_kws_scratch/
├── README.md               # Project overview and instructions
├── requirements.txt        # Python dependencies
├── data/
│   └── download_gsc.py     # Script to download Google Speech Commands v2
├── features/
│   └── extract_mfcc.py     # MFCC feature extraction
├── model/
│   ├── tcn.py              # TCN model implementation using NumPy
│   └── layers.py           # Conv1D, ReLU, etc.
├── train/
│   ├── train.py            # Training loop and evaluation
│   └── utils.py            # Utility functions (e.g. accuracy, loss)
├── inference/
│   └── run_inference.py    # Script for running inference on new audio
├── tests/
│   └── test_layers.py      # Unit tests for model components
└── venv/                   # Virtual environment (not pushed to GitHub)

