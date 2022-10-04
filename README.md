# Semi-supervise
# Amazon Group2
## 1. Semi supervise 
### train_baseline.py+model.py+dataloader.py: use RobertaForSequenceClassification to classification. But the classification effect is not ideal.  

`0.6: best f1 0.535089 and acc 0.635226` 

`0.5：best f1 0.704604 and acc 0.544012 `

`0.4: best f1 0.812973 and acc 0.684882`

`0.3: best f1 0.884315 and acc 0.792621`  

But we want to set threshold 0.7(greater than 0.7 is 1, less than 0.7 is 0), the acc is poor. 

## 2. Contrasting learning 
### train_triple.py + model_triple.py + dataloader_tuple.py + preprocess_data_triple.py.  

(1) Reassemble the original data into new Tuples and save into csv. three tuples: anchor,positive,negative. 

(2)  Train Roberta using contrast learning and improve embedding's abilities with TripletMarginWithDistanceLoss. The loss is about 0.99~1


