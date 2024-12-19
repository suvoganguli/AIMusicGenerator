# Chapter 7 - Training Magenta models

[![Magenta Version 1.1.7](../docs/magenta-v1.1.7-badge.svg)](https://github.com/magenta/magenta/releases/tag/1.1.7)

This chapter will show how to tune hyperparameters, like batch size, learning rate, and network size, to optimize network performance and training time. We’ll also show common training problems such as overfitting and models not converging. Once a model's training is complete, we'll show how to use the trained model to generate new sequences. Finally, we'll show how to use the Google Cloud Platform to train models faster on the cloud.

## Magenta Versioning

```diff
- A newer version of this code is available.
```

This branch shows the code for Magenta v1.1.7, which corresponds to the code in the book. For a more recent version, use the updated [Magenta v2.0.1 branch](https://github.com/PacktPublishing/hands-on-music-generation-with-magenta/tree/magenta-v2.0.1/Chapter07).

## Code

This chapter doesn't contain a lot of code, refer to the book for the content. Before you start, follow the [installation instructions for Magenta 1.1.7](https://github.com/PacktPublishing/hands-on-music-generation-with-magenta/tree/master/Chapter01#installing-magenta).

### [Example 1](chapter_07_example_01.py)

Configuration for the MusicVAE model, using the MIDI bass programs. To launch the training:

```bash
python chapter_07_example_01.py --config="cat-bass_2bar_small" --run_dir="..."
```

### [Example 3](chapter_07_example_02.py)

Tensor validator and note sequence splitter (training and evaluation datasets) for the MusicVAE model.

```bash
python chapter_07_example_02.py --config="cat-drums_2bar_small" --input="notesequences.tfrecord" --output_dir="sequence_examples"
```

### [Example 3](chapter_07_example_03.py)

Configuration for the Drums RNN model that inverts the snares and bass drums.
