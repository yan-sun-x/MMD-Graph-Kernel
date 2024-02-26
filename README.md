# MMD Graph Kernel: Effective Metric Learning for Graphs via Maximum Mean Discrepancy

> Published as a conference paper at *ICLR 2024* as a spotplight paper.

1. We present a class of maximum mean discrepancy (MMD) based graph kernels, called **MMD-GK**. These kernels are computed by applying MMD to the node representations of two graphs with message-passing propagation. 
2. Based on this vanilla version, we provide a class of deep MMD-GKs that are able to learn graph kernels and implicit graph features adaptively in an unsupervised manner. 
3. Apart from that, we propose a class of supervised deep MMD-GKs that are able to utilize label information of graphs and hence yield more discriminative metrics.

## How to Use

Remember to install all the dependencies as below.

```bash
pip install -r requirements.txt
```

We provide a sample dataset (MUTAG) in the `data` folder. Please configure your settings in `utils/arguments.py`

Run the vanilla version (MMDGK) with a command:
```bash
python main.py --model 'vanilla'
```

Run the deep version (Deep MMDGK) with a command:
```bash
python main.py --model 'deep'
```