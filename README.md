<div align="center">

# PartSAM: A Scalable Promptable Part Segmentation Model Trained on Native 3D Data

</div>

<div align="center">

[Zhe Zhu](https://scholar.google.com/citations?user=pM4ebg0AAAAJ&hl=zh-CN)<sup>1</sup>, [Le Wan](https://scholar.google.com/citations?user=pM4ebg0AAAAJ&hl=zh-CN)<sup>2</sup>, [Rui Xu](https://ruixu.me/)<sup>3</sup>, [Yiheng Zhang](https://openreview.net/profile?id=~Yiheng_Zhang4)<sup>4</sup>, [Honghua Chen](https://chenhonghua.github.io/clay.github.io/)<sup>5</sup>, [Zhiyang Dou](https://frank-zy-dou.github.io/)<sup>3</sup>, [Cheng Lin](https://clinplayer.github.io/)<sup>6</sup>, [Yuan Liu](https://liuyuan-pal.github.io/)<sup>2&dagger;</sup>, [Mingqiang Wei](https://scholar.google.com/citations?user=TdrJj8MAAAAJ&hl=en)<sup>1&dagger;</sup>
<br>
&dagger; Corresponding authors

**Affiliations:**
<sup>1</sup> Nanjing University of Aeronautics and Astronautics
<sup>2</sup> Hong Kong University of Science and Technology 
<sup>3</sup> The University of Hong Kong
<sup>4</sup> National University of Singapore
<sup>5</sup> Lingnan University
<sup>6</sup> Macau University of Science and Technology

</div>

<p align="center">
  <img src="assets/teaser.png" alt="teaser">
</p>

<div align="center">

Code is coming soon


## Abstract

Segmenting 3D objects into parts is a long-standing challenge in computer vision. To overcome taxonomy constraints and generalize to unseen 3D objects, recent works turn to open-world part segmentation. These approaches typically transfer supervision from 2D foundation models, such as SAM, by lifting multi-view masks into 3D. However, this indirect paradigm fails to capture intrinsic geometry, leading to surface-only understanding, uncontrolled decomposition, and limited generalization. We present PartSAM, the first promptable part segmentation model trained natively on large-scale 3D data. Following the design philosophy of SAM, PartSAM employs an encoder-decoder architecture in which a triplane-based dual-branch encoder produces spatially structured tokens for scalable part-aware representation learning. To enable large-scale supervision, we further introduce a model-in-the-loop annotation pipeline that curates over five million 3D shape-part pairs from online assets, providing diverse and fine-grained labels. This combination of scalable architecture and diverse 3D data yields emergent open-world capabilities: with a single prompt, PartSAM achieves highly accurate part identification, and in a Segment-Every-Part mode, it automatically decomposes shapes into both surface and internal structures. Extensive experiments show that PartSAM outperforms state-of-the-art methods by large margins across multiple benchmarks, marking a decisive step toward foundation models for 3D part understanding. Our code and model will be released soon.


## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{zhu2025partsam,
      title={PartSAM: A Scalable Promptable Part Segmentation Model Trained on Native 3D Data}, 
      author={Zhe Zhu and Le Wan and Rui Xu and Yiheng Zhang and Honghua Chen and Zhiyang Dou and Cheng Lin and Yuan Liu and Mingqiang Wei},
      journal={arXiv preprint arXiv:2509.21965},
      year={2025}
}
```
