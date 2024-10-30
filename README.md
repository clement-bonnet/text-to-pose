# From Text to Pose to Image: Improving Diffusion Model Control and Quality

- [Link to the paper](paper.pdf) 📝
- [Link to Text-To-Pose model](https://huggingface.co/clement-bonnet/t2p-transformer-v0) 🤗
- [Link to CLaPP (Contrastive Language-Pose Pretraining) model](https://huggingface.co/clement-bonnet/clapp-v0) 🤗
- [Link to Pose Adapter model](https://huggingface.co/clement-bonnet/t2i-adapter-sdxl-dwpose) 🤗
- [Link to created COCO-2017 annotated dataset](https://huggingface.co/datasets/clement-bonnet/coco_val2017_100_text_image_pose) 🤗


This repository contains the code for the paper _From Text to Pose to Image: Improving Diffusion Model Control and Quality_, published at the NeurIPS 2024 Workshop on Compositional Learning: Perspectives, Methods, and Paths Forward ([link to workshop](https://compositional-learning.github.io/)).

<div align="center" style="display: flex; justify-content: center; flex-wrap: nowrap;">
  <figure style="margin: 10px; width: 45%;">
    <img src="images/fig_1_a.png" style="width: 100%;" />
    <figcaption>Standard <em>text-to-image</em> generation</figcaption>
  </figure>
  <figure style="margin: 10px; width: 45%;">
    <img src="images/fig_1_b.png" style="width: 100%;" />
    <figcaption>Ours: <em>text-to-pose-to-image</em> generation</figcaption>
  </figure>
</div>

## Text To Pose

<div align="center">
    <img src="images/t2p_architecture.png" width="100%" style="max-width: 600px" />
    <figcaption>Text-to-pose transformer architecture</figcaption>
</div>


## Pose Adapter

<div align="center">
    <img src="images/adapter_generations.png" width="100%" style="max-width: 600px" />
    <figcaption>Generated poses using the Tencent pose adapter and ours</figcaption>
</div>


# Citation

If you use this paper in your work, please cite the paper using (citation to come):

```
@misc{...}
```
