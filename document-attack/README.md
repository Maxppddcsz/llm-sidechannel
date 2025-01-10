# prepare documents

1. download data and generate PDF documents
```shell
python data/pre.py
python data/prepare_pdf.py
```

# run victim and attacker apps
```python
streamlit run attacker_pdf.py
streamlit run victim_pdf.py
```

# analyze the result
```python
python plot.py
```
