# Awesome coding programs
Read & Learn.

[TOC]

## Deep Learning
### segment-anything
A nice blog post reviewed SAM in Chinese: 
https://blog.csdn.net/qq_43426908/article/details/136284259?spm=1001.2014.3001.5502

A nice github repo collects existing works on SAM in Medical Images field: 
https://github.com/YichiZhang98/SAM4MIS

<img src='./figs/sam_overview.jpg'>

Project framework: 
```
segment_anything
    |-modeling
        |-__init__.py
        |-common.py
        |-image_encoder.py
        |-mask_decoder.py
        |-prompt_encoder.py
        |-sam.py
        |-transformer.py
    |-utils
        |-__init__.py
        |-amg.py
        |-onnx.py
        |-transforms.py
    |-__init__.py
    |-automatic_mask_generator.py
    |-build_sam.py
    |-predictor.py 

```