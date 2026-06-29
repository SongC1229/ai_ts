---
tags:
- model_hub_mixin
- pytorch_model_hub_mixin
license: openrail
language:
- en
metrics:
- accuracy
base_model:
- microsoft/wavlm-large
pipeline_tag: audio-classification
---
# WavLM-Large for Age and Sex Classification

# Model Description
This model includes the implementation of age and sex classification described in Vox-Profile: A Speech Foundation Model Benchmark for Characterizing Diverse Speaker and Speech Traits (https://arxiv.org/pdf/2505.14648)

The sex labels are: ["Female", "Male"].
The age output is from 0-1, and times 100 is the actual age.

- Library: https://github.com/tiantiaf0627/vox-profile-release
- Docs: [More Information Needed]

## Kindly cite our paper if you are using our model or find it useful in your work
```
@article{feng2025vox,
  title={Vox-Profile: A Speech Foundation Model Benchmark for Characterizing Diverse Speaker and Speech Traits},
  author={Feng, Tiantian and Lee, Jihwan and Xu, Anfeng and Lee, Yoonjeong and Lertpetchpun, Thanathai and Shi, Xuan and Wang, Helin and Thebaud, Thomas and Moro-Velazquez, Laureano and Byrd, Dani and others},
  journal={arXiv preprint arXiv:2505.14648},
  year={2025}
}
```
Responsible use of the Model: the Model is released under Open RAIL license, and users should respect the privacy and consent of the data subjects, and adhere to the relevant laws and regulations in their jurisdictions in using our model.

❌ **Out-of-Scope Use**
- Clinical or diagnostic applications
- Surveillance
- Privacy-invasive applications
- No commercial use