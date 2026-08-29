# Methodology: Sentinel-2 Super-Resolution Mapping

This document describes the technical methodology behind the Sentinel-2 Super-Resolution Mapping (SRM) pipeline.

---

## 1. Data Normalization & Preprocessing

The input Sentinel-2 Level-2A surface reflectance bands (B04-Red, B03-Green, B02-Blue, and B08-NIR) are natively stored in 16-bit unsigned integers representing reflectance values (typically in the range `[0, 10000]`). 

To prepare them for the network, they are divided by a scaling factor of $10000.0$:

$$\mathbf{I}_{\text{norm}} = \frac{\mathbf{I}_{\text{raw}}}{10000.0}$$

This maps the surface reflectance values strictly to the range `[0.0, 1.0]`. Normalization is identical for both the low-resolution input and high-resolution target, preserving spectral shape.

---

## 2. Model Architecture: Residual SwinIR

The core architecture is a modified **SwinIR** (Swin Transformer for Image Restoration) tailored for satellite data:
- **Four Input Channels**: Red, Green, Blue, NIR (RGBN) instead of standard 3-channel RGB.
- **Four Output Channels**: High-resolution RGBN.
- **Scale Factor**: $\times 4$ spatial upsampling.
- **Lightweight Design**: Approximately $912,244$ parameters to allow fast inference on edge GPUs (such as the NVIDIA RTX 2050 4GB).

---

## 3. Seamless Tiled Inference & 2D Hann Blending

When running inference on large, arbitrary-sized real Sentinel-2 scenes, processing the entire image at once can exceed GPU VRAM constraints. The pipeline resolves this using a **seamless tiled sliding-window inference** technique:

1. **Overlapping Patch Extraction**: The input image is split into overlapping patches of size $64 \times 64$ pixels (in LR) with a default stride of $48$ (resulting in a $25\%$ overlap).
2. **Padding**: Reflection padding is applied on the edges if the scene dimensions are not perfectly divisible by the tile size.
3. **2D Hann Window Weighting**: For each upscaled HR tile ($256 \times 256$), a 2D Hann window is applied to downweight the edges:
   
   $$w_{2D}(y, x) = w_{1D}(y) \times w_{1D}(x)$$
   
   Where $w_{1D}$ is a standard 1D Hanning window.
4. **Weighted Accumulation**: Upscaled patches are multiplied by the 2D window and accumulated into a global buffer, while a parallel weight buffer accumulates the window values.
5. **Normalization**: The final upscaled image is divided by the weight buffer. This completely eliminates edge seams and blocking artifacts.
