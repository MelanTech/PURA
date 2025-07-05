# PURA: Parameter Update-Recovery Test-Time Adaption for RGB-T Tracking

<p align="center">
  📖<a href="https://melantech.github.io/PURA/" target="_blank">[Project Page]</a> 
</p>

This is the official repository for **PURA: Parameter Update-Recovery Test-Time Adaption for RGB-T Tracking** (CVPR 2025).

<div style="text-align: center;">
  <img src="imgs/framework.png" style="width: 90%;" />
</div>

## Introduction

✨ PURA is a test-time adaptation framework for RGB-T tracking, designed to robustly adapt to domain transition scenarios during the testing process.

✨ PURA adapts online through parameter update and recovery mechanisms, avoiding drastic parameter changes and heavy computational burden.

## Results

We have released the results of our method on the RGBT234, RGBT210 and GTOT datasets:

<table style="text-align: center;">
  <tr>
    <th rowspan="2">Method</th>
    <th colspan="2">RGBT234</th>
    <th colspan="2">RGBT210</th>
    <th colspan="2">GTOT</th>
    <th rowspan="2">FPS</th>
    <th rowspan="2">Raw Result</th>
  </tr>
  <tr>
    <th>MPR(%)</th>
    <th>MSR(%)</th>
    <th>PR(%)</th>
    <th>SR(%)</th>
    <th>MPR(%)</th>
    <th>MSR(%)</th>
  </tr>
  <tr>
    <td>No Adapt.</td>
    <td>90.8</td>
    <td>67.6</td>
    <td>88.6</td>
    <td>65.1</td>
    <td>95.1</td>
    <td>78.2</td>
    <td>59.5</td>
    <td><a href="https://drive.google.com/drive/folders/19ojw2JUciB-R9WdNFxwtrnVR0FPQGhFC?usp=sharing" target="_blank">Google Drive</a></td>
  </tr>
  <tr>
    <td>PURA</td>
    <td>93.3</td>
    <td>70.3</td>
    <td>90.3</td>
    <td>66.8</td>
    <td>95.7</td>
    <td>78.6</td>
    <td>42.0</td>
    <td><a href="https://drive.google.com/drive/folders/19ojw2JUciB-R9WdNFxwtrnVR0FPQGhFC?usp=sharing" target="_blank">Google Drive</a></td>
  </tr>
</table>

* Note: The full code and weights of our method will be released soon.

## Usage

For applying PURA to your own RGB-T tracker based on [pytracking](https://github.com/visionml/pytracking), you can follow the steps below:

1. Copy `pura.py` to `lib\test\tracker` folder.
2. Modify `lib\test\tracker\xxx_track.py` to include the following code:

```python
from lib.test.tracker import pura  # import PURA

...


class XXXTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(XXXTrack, self).__init__(params)
        network = build_xxx_track(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()

        pura.replace_batchnorm(self.network.box_head)  # replace batchnorm in box_head with PURA
        self.network = pura.configure_model(self.network)  # configure model

        self.preprocessor = Preprocessor()
        self.state = None
        ...
```

## Acknowledgments

We use the implementation of the SVD decomposition from the [PGrad](https://github.com/QData/PGrad) repo.

## Citation

If our method is helpful for your research, please consider citing our paper:

```bibtex
 @inproceedings{shao2025pura,
    title={PURA: Parameter Update-Recovery Test-Time Adaption for RGB-T Tracking},
    author={Zekai, Shao and Yufan, Hu and Bin, Fan and Hongmin, Liu},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2025}
    }
```