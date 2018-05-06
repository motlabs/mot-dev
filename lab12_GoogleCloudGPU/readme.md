# Comparison: Colab GPU use vs. gcloud GPU use


### Conclusion
- We have 10 people in MoT team
- We spend $30 per person for 24 hour use in gcloud
- Assuming 5 people use gcloud once per week
- A person spends 24 hour per gcloud-use
- Then, we monthly spend 
```bash
5 people * 4 weeks * $30 = $600 per month
```
- A four-1080ti-GPU machine requires around $10 hundred to buy
- For 16.66 month less use, gcloud is cheaper.


### Information

| Items                     |   Colab GPU                   |   gcloud GPU  per hour                      |
| :------------------------:| :------------------------ --: | :-----------------------------------------: |
| Time based price          |   Free (limited to 12 hours)  |   $1.3578 / hour (BASIC_GPU in Asia-pacific) |
| Training unit based price |   Free (Limited to 12 hours)  |   $2.5144 / hour (BASIC_GPU in Asia-pacific) |


**BASIC_GPU scale tier**
A single worker instance with a single NVIDIA Tesla K80 GPU.

#### gcloud GPU pricing 

**Time based Price**
```bash
(Price per hour / 60 ) * job duration in minites
```

- A case study
    - use 12 hours
    - BASIC_GPU scale tier in Asia-pacific
```bash 
Price for use = (1.3578 / 60) * 12 * 60 = $16.2936
```
 
 
**Training unit based price**
```bash
(training units * base price per hour / 60) * job duration in minutes
```

- A case study
    - use 12 hours
    - BASIC_GPU scale tier  in Asia-pacific
    - base price per hour = $0.49   
```bash
Price for use = (2.5144 * $0.49 / 60) * 12 * 60 = $14.7846
```


- Reference: 
    - About Pricing: https://cloud.google.com/ml-engine/docs/pricing  
    - About scale tier: https://cloud.google.com/ml-engine/docs/tensorflow/training-overview#scale_tier