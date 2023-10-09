
<p align="center">
  
  <h3 align="center"><strong>STARNet: Sensor Trustworthiness and Anomaly Recognition via Approximated Likelihood Regret for Robust Edge Autonomy</strong></h3>

  <p align="center">
    <a href="https://scholar.google.com/citations?user=ZKdsKvQAAAAJ&hl=en&oi=ao">Nastaran Darabi</a>1*<sup></sup>&nbsp;&nbsp;
    <a href="https://scholar.google.com/citations?user=GjfKPkUAAAAJ&hl=en&oi=ao">Sina Tayebati</a>1*<sup></sup>&nbsp;&nbsp;
    <a href="#">Sureshkumar S.</a><sup>1</sup>&nbsp;&nbsp;
    <a href="https://scholar.google.com/citations?user=FW-0thoAAAAJ&hl=en">Sathya Ravi</a><sup>1</sup>&nbsp;&nbsp;
    <a href="https://scholar.google.com/citations?user=K6FIDzYAAAAJ&hl=en">Theja Tulabandhula</a><sup>1</sup>&nbsp;&nbsp;
    <a href="https://scholar.google.com/citations?user=Thpd0HkAAAAJ&hl=en">Amit R. Trivedi</a><sup>1</sup>
    <br>
    <sup>1</sup>University of Illinois Chicago&nbsp;&nbsp;&nbsp;
    <sup>*</sup>Both authors contributed equally to this work&nbsp;&nbsp;&nbsp;
  </p>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2309.11006" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-%F0%9F%93%83-blue">
  </a>
  
  <a href="#" target='_blank'>
    <img src="https://img.shields.io/badge/Project-%F0%9F%94%97-lightblue">
  </a>
  
  <a href="#" target='_blank'>
    <img src="https://img.shields.io/badge/Demo-%F0%9F%8E%AC-blue">
  </a>
</p>

## About
**STARNet** introduces a deep network designed for anomaly/corruption detection in LiDAR-Camera data pipelines, with focus on implementation on edge AI devices such as micro drones. STARNet employs a gradient-free likelihood regret concept integrated with a variational autoencoder, making implementation on low-complexity edge hardware possible.

<p align="center">
  <img src="docs/figs/Picture1.png" align="center" width="80%">
</p>

## Demo: STARNet in action
Bellow we demonstrate a demo of STARNet integrated with <a href="https://airsim-fork.readthedocs.io/en/docs/#">AIRSim</a>, detecting OOD and IN-D data in online streaming.

<p align="center">
    <img src="https://github.com/sinatayebati/STARNet/blob/main/docs/gif/demo.GIF" width="400" />
</p>
