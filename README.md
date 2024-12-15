# Predicting London Emissions Using Geographic Data (csck503_ema_group_c)

This library contains the supporting code for the Group C's EMA project.
The goal is to test three different ML models to support the prediction of emissions:
* Gradient Boosting Machine (GBM)
* Random Forest (RF)
* Linear Regression

The following four emissions are supported:
* CO2
* NOX
* PM10
* PM2.5

## Pre-Requisites

Python 3.13

Install requirements.txt

`pip install -r /path/to/requirements.txt`

Additionally, the [LAEI2019 Data Files](https://data.london.gov.uk/dataset/london-atmospheric-emissions-inventory--laei--2019) are expected to be unpacked and populated as per the following directory structure. **This is already done by default.**

```
raw_data/
├─ emissions_summary/
│  ├─ Shapefile SHP/
│  │  ├─ LAEI2019-*.shp
supporting_data/
├─ grid/
├─ rail/
├─ road/
│  ├─ excel/
│  ├─ shape/
├─ shipping/
```

## Installation

No installation necessary, use the scripts directly as per the usage section

## Usage

The data_parser.py module parses the geometric and excel data into a final data frame which is stashed in `parsed_data/final_df.pkl`

```bash
python data_parser.py
```

Each model type can then be tested directly.
```bash
python gradient_boosting_machine.py
python linear_regression.py
python random_forest.py
```

Alternatively the models can be indivually experimented with to tweak parameters.
All models support the following key word arguments:
    * include_outliers: Whether or not outliers should be included in the model (default: True)
    * include_other_pollutants: Whether or not the other three pollutants should be included as training features (default: False)

```python
import gradient_boosting_machine as gbm
gbm.gbm_pollution("co2", include_outliers=True, include_other_pollutants=False)
```

## License
[MIT](https://choosealicense.com/licenses/mit/)