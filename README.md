# M4 (MLP-Mixer based Multi-modal image-text retrieval)

![image](https://user-images.githubusercontent.com/47732974/146780955-da28d6e1-4192-4c0e-a2fe-f81edbc43ec2.png)

### Image:
Original image is cropped with 16 x 16 patch size without overlap. Then, it is reshaped to (batch, (hxw), (patch x patch x channel)). 

### Text:
Also, original text is tokenized and embedded with BERT-based approach (BERT-base-uncased).

### Data processing:
When we train our model, we randomly samples(50 %) reports to make the matched- and un-matched image-text set.
Basically, matched and un-matched set is classified with label information using chexpert labeler, we consider unmatched set when randomly sampled report is not exactly same with original one.


Mixer based approach is trained efficiently with xxxx throuput with xxx accuracy.



### Exp settings.

batch: 256 batch 
epoch: 50 epoch




### Chest X-ray Image-reports retrieval
Model spec:
patch size:16, embedding dim: 768

Input spec:
img size: 224x224x3 -> pathch size: (224/16) x (224/16)
text max len: 128 legth

input embedding: cls, txt, sep, img 
output: matched or unmatched



### Results

![image](https://user-images.githubusercontent.com/47732974/146780704-d159b33f-720e-41fa-8ce8-df13680f1c0a.png)

