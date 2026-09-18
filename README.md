My fork of https://github.com/tue-mps/eomt supporting **1-channel inputs and better SAR processing**. WIP.



EOMT Credits:
# Your ViT is Secretly an Image Segmentation Model  
[![Papers with Code: SOTA on BRAVO (OOD)](https://paperswithcode.co/api/v1/papers/2503.19108/leaderboard-badge.svg?eval=640&live=1)](https://paperswithcode.co/benchmark/bravo-ood?task=image-segmentation&eval=640)
[![Papers with Code: SOTA on COCO 2017 Panoptic Segmentation](https://paperswithcode.co/api/v1/papers/2503.19108/leaderboard-badge.svg?eval=6260&live=1)](https://paperswithcode.co/benchmark/coco-2017-panoptic-segmentation?task=image-segmentation&eval=6260)


**CVPR 2025 ✨ Highlight** · [📄 Paper](https://arxiv.org/abs/2503.19108)

**[Tommie Kerssies](https://tommiekerssies.com)<sup>1</sup>, [Niccolò Cavagnero](https://scholar.google.com/citations?user=Pr4XHRAAAAAJ)<sup>2,*</sup>, [Alexander Hermans](https://scholar.google.de/citations?user=V0iMeYsAAAAJ)<sup>3</sup>, [Narges Norouzi](https://scholar.google.com/citations?user=q7sm490AAAAJ)<sup>1</sup>, [Giuseppe Averta](https://www.giuseppeaverta.me/)<sup>2</sup>, [Bastian Leibe](https://scholar.google.com/citations?user=ZcULDB0AAAAJ)<sup>3</sup>, [Gijs Dubbelman](https://scholar.google.nl/citations?user=wy57br8AAAAJ)<sup>1</sup>, [Daan de Geus](https://ddegeus.github.io)<sup>1,3</sup>**

¹ Eindhoven University of Technology  
² Polytechnic of Turin  
³ RWTH Aachen University  
\* Work done while visiting RWTH Aachen University

## Citation
If you find this work useful in your research, please cite it using the BibTeX entry below:

```BibTeX
@inproceedings{kerssies2025eomt,
  author    = {Kerssies, Tommie and Cavagnero, Niccol\`{o} and Hermans, Alexander and Norouzi, Narges and Averta, Giuseppe and Leibe, Bastian and Dubbelman, Gijs and {de Geus}, Daan},
  title     = {{Your ViT is Secretly an Image Segmentation Model}},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025},
}
```

## Acknowledgements

This project builds upon code from the following libraries and repositories:

- [Hugging Face Transformers](https://github.com/huggingface/transformers) (Apache-2.0 License)  
- [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models) (Apache-2.0 License)  
- [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) (Apache-2.0 License)  
- [TorchMetrics](https://github.com/Lightning-AI/torchmetrics) (Apache-2.0 License)  
- [Mask2Former](https://github.com/facebookresearch/Mask2Former) (Apache-2.0 License)
- [Detectron2](https://github.com/facebookresearch/detectron2) (Apache-2.0 License)
