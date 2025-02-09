# DDTasnet Model Architecture

This document outlines the main architectural components and the design rationale behind DDTasnet, a model for real‐time, low‐latency music source separation. DDTasnet is an evolution of the original HStasNet and is designed to fuse information from both time–domain and spectral–domain representations.

## Overview

DDTasnet processes multi–channel audio by simultaneously extracting time–domain features and spectrogram features from the input waveform. Each representation is handled by its own encoder, memory modules, and attention mechanisms before being fused via a hybrid RNN. The fused representation is then split into domain–specific streams to generate masks that are applied to the encoded representations. Two decoders finally reconstruct the separated source signals, which are merged to produce the final output.

## Architectural Components

### Time–Domain Branch

The time–domain branch begins with a learned convolutional encoder that transforms the input waveform into a latent representation and produces a normalisation factor. The resulting features are reshaped so that each time frame contains a concatenated representation across channels. These features pass through an RNN module augmented with a skip attention mechanism and a channel–spatial attention block (CBAM). A secondary memory module with a linear skip connection (projecting the original encoded input to the same dimension) further refines the representation. This branch ultimately generates a mask via a fully connected layer that modulates the original time–domain features.

### Spectral–Domain Branch

In parallel, the spectral–domain branch uses a spectrogram encoder to compute magnitude and phase representations from the input waveform. Each spectral component is reshaped into a tensor that groups frequency bins across channels. Both the magnitude and phase tensors are independently processed through RNN-based memory modules, skip attention, and CBAM blocks. A linear projection (skip connection) is applied to preserve key spectral details. Fully connected layers generate masks for the magnitude and phase components, which are applied to the encoded spectrogram features.

### Hybrid Fusion

A key innovation in DDTasnet is the introduction of a hybrid RNN fusion module. The outputs from the time–domain branch and the two spectral branches are concatenated along the feature dimension to form a joint representation that encapsulates both temporal and frequency–based information. This hybrid RNN learns to integrate the complementary cues from both domains. After fusion, the combined features are split back into streams corresponding to the original branches. Additional memory modules and linear skip connections help preserve low-level details and ensure smooth gradient flow.

### Decoding and Reconstruction

Once domain–specific masks are applied to the encoded representations, two separate decoders reconstruct the separated source signals: one for time–domain features and one for spectrogram features. The decoder outputs are summed to form the final output, which is reshaped to match the original waveform dimensions and adjusted (via padding or trimming) as necessary.

## Rationale Behind the Design

The hybrid architecture is motivated by the need to capture complementary information in audio signals. The time–domain branch preserves fine temporal structure critical for percussive elements and transients, whereas the spectral–domain branch provides the frequency resolution required for harmonic and tonal content. By processing both representations in parallel and fusing them with a hybrid RNN, DDTasnet leverages the strengths of each domain. Attention modules (skip attention and CBAM) further refine the representations, emphasising important features while suppressing redundancy.

Enhanced skip connections using linear projections ensure robust information flow and training stability in a causal (real–time) setup. Carefully chosen hyperparameters—such as a window size of 1024 samples and a hop size of 512—balance spectral resolution with the low latency necessary for real–time applications.

## Improvements Over the Original HStasNet

DDTasnet builds upon the original HStasNet by introducing several key refinements:
- **Modular Dual–Branch Structure:** Distinct time–domain and spectral–domain branches provide dedicated processing streams.
- **Attention Mechanisms:** The incorporation of skip attention and CBAM in both branches improves feature discrimination.
- **Hybrid Fusion RNN:** A novel fusion module integrates complementary cues from both domains more effectively.
- **Enhanced Skip Connections:** Linear projections are used to match dimensions and preserve crucial low-level details.
- **Optimised Hyperparameters:** Carefully chosen window and hop sizes ensure a balance between performance and low latency.

These updates yield improved separation quality and robustness while maintaining the low latency required for real-time music applications.