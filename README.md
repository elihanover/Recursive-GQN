# Generative Query Network

This is a PyTorch implementation of the Generative Query Network (GQN)
described in the DeepMind paper "Neural scene representation and
rendering" by Eslami et al. For an introduction to the model and problem
described in the paper look at the article by [DeepMind](https://deepmind.com/blog/neural-scene-representation-and-rendering/).

![](https://storage.googleapis.com/deepmind-live-cms/documents/gif_2.gif)

# Recursive GQN
This repository is part of an ongoing research into GQN's potential for transfer learning and implements a "recursive" variant of the model which enables higher level, specialized, scene representations that can be shared across tasks.

The authors demonstrate that the scene representation can be used to reduce the number of training steps for a robotic arm grasping task by 75%.  Using the scene representation rather than input directly makes sense given it is lower dimensional, and this lower dimensional space "specializes" in storing the most important features of the scene for it to be rerendered.

In the context of this Recursive GQN, this would be an example of "layer 1" transfer learning.

#### Downloading and Converting Data
The current implementation generalises to any of the datasets described
in the paper. However, currently, *only the Shepard-Metzler dataset* has
been implemented. To use this dataset you can use the provided script in
``` bash
sh data.sh data-dir batch-size
# for example
sh data.sh .. 20
```
##### As of now, the above script has problems converting some but not all of the dataset.

#### Train Model
The model can be trained in full by in accordance to the paper by running the
file `run-gqn.py` or by using the provided training script
``` bash
sh gpu.sh data-dir
# for example
sh gpu.sh ../shepard_metzler_5_parts
```

## Roadmap
