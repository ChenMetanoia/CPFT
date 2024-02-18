# CPFT
This repo is for anonymous paper: Confidence-aware Fine-tuning of Sequential Recommendation
Systems through Conformal Prediction 

## Requirements

```
recbole==1.1.1
python==3.8.5
cudatoolkit==11.3.1
pytorch==1.12.1
pandas==1.3.0
transformers==4.18.0
```

### 1. Our Code was built upon [UnisRec](https://github.com/RUCAIBox/UniSRec)

### 2. Download [Amazon meta dataset](https://nijianmo.github.io/amazon/index.html)
Category: Scientific

Data: metadata

Put dataset into ```dataset/raw/Scientific/Metadata``` directory. For example ```dataset/raw/Metadata/meta_Electronics.json.gz```

Data: ratings

Put dataset into ```dataset/raw/Scientific/Ratings``` directory. For example ```dataset/raw/Ratings/meta_Scientific.csv```

### 3. Process data
```
cd dataset/preprocessing
python process_amazon_CPFT.py --dataset Scientific --input_path dataset/raw/
```

### 4. Run regular train
```
python run_baseline.py -m SASRec -d Scientific
```

### 5. Run CPFT
```
python run_cpft.py -m SASRec -d Scientific -r "trained_model.pth" --alpha 0.3 --beta 10 --gamma 1 --dis_k 10
```