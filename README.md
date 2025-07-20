# MMD Graph Kernel: Effective Metric Learning for Graphs via Maximum Mean Discrepancy

## 📢 News
- Docker support is now available (see instructions below).
- **Bugs fixed and results are now reproducible.**
- 🎉 Accepted as a *spotlight* paper at **ICLR 2024**.

- We public the code. Here is the structure of this project repo:
```
MMD-Graph-Kernel/         
├── mmdgk/                  # Core module package
│   ├── __init__.py
│   ├── kernels.py          # MMD kernels
│   ├── loss.py             # Loss functions used in training
│   ├── models.py           # GCN Model architecture
│   └── utils/
│       ├── __init__.py
│       ├── arguments.py    # Argument parser
│       ├── evaluation.py   # Evaluation metrics and functions
│       └── get_data.py     # Dataset loading and preprocessing
├── main.py                 # Main training/testing script
├── run_demo.py
```


## TL;DR

1. We introduce a class of maximum mean discrepancy (MMD)-based graph kernels, called **MMD-GK**, which apply MMD to node representations propagated via message passing.
2. Building on this, we propose a class of **deep MMD-GKs** that can adaptively learn implicit graph features in an unsupervised manner.
3. Additionally, we present **supervised deep MMD-GKs**, which incorporate graph labels to learn more discriminative metrics.




## 🧪 Usage

First, install dependencies:

```bash
pip install -r requirements.txt
```

The `data/` folder contains a sample dataset (MUTAG). Configure settings in `utils/arguments.py`.

To run the vanilla version (MMDGK):
```bash
python main.py --model 'vanilla' --dis_gamma 1e0 --bandwidth "[1e0, 1e1]"
```

To run the deep version (Deep MMDGK):
```bash
python main.py --model 'deep' --dataname 'MUTAG' --epochs 10 
```

## 🐳 Docker Support

### Build the image
  From the project root:

  ```bash
  docker build -t mmdgk-image .
  ```
### Run the container
  To run the project in the container. This executes `bash run_demo.sh` by default.
  ```bash
  docker run --rm mmdgk-image
  ```

### Optional: Interactive mode
   To open an interactive shell in the container. From there, you can run bash `run_demo.sh` or directly execute `main.py`.
  ```bash
  docker run -it --rm mmdgk-image /bin/bash
  ```
  

### Optional: Enable GPU support
   If you have an NVIDIA GPU and the NVIDIA Container Toolkit installed:
```bash
docker run --rm --gpus all mmdgk-image
```
### Optional: Save logs or outputs
   To save logs or outputs to your local machine. This maps the container's `/workspace/runs` directory to your local `./logs` folder.
  ```bash
  docker run --rm -v $(pwd)/logs:/workspace/runs mmdgk-image
  ```
  


## 📖 Citation
If you use this code, please cite:
```
@inproceedings{sun2023mmd,
  title={MMD Graph Kernel: Effective Metric Learning for Graphs via Maximum Mean Discrepancy},
  author={Sun, Yan and Fan, Jicong},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
```