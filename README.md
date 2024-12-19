# MambaLCT https://arxiv.org/abs/2412.13615
# MambaLCT 
The official implementation for the **AAAI'2025** paper [_MambaLCT: Boosting Tracking via Long-term Context State Space Model_](https://arxiv.org/abs/2412.13615) 

Models:[[Models]](https://drive.google.com/drive/folders/1PtpomZNItT6B7gdf4hnH3nGdnRJPVVT0?usp=drive_link)
Raw Results:[[Raw Results]](https://drive.google.com/drive/folders/1nuU4LyH1NLPs1U9mrxNCdkhptv6Vc2Nr?usp=drive_link)


## :sunny: Structure of AQATrack 
![structure](https://github.com/GXNU-ZhongLab/MambaLCT/blob/main/assets/arch.png)


## :sunny: Highlights

### :star2: New Autoregressive Query-based Tracking Framework
AQATrack is a simple, high-performance **autoregressive query-based spatio-temporal tracker** for adaptive learning the instantaneous target appearance changes in a sliding window
fashion. Without any additional upadate strategy, AQATrack achieves SOTA performance on multiple benchmarks.

| Tracker     | LaSOT (AUC)|LaSOT<sub>ext (AUC)|UAV123 (AUC)|TrackingNet (AUC)|TNL2K(AUC)|GOT-10K (AO)
|:-----------:|:----------:|:-----------------:|:----------:|:---------------:|:--------:|:----------:
| AQATrack-256| 71.4       | 51.2              | 70.7       | 83.8            | 57.8     | 73.8         
| AQATrack-384| 72.7       | 52.7              | 71.2       | 84.8            | 59.3     | 76.0         
