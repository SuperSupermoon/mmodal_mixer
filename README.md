# mmodal_mixer


## MLP-Mixer based Multi-modal image-text retrieval

### Image:
Original image is cropped with 16 x 16 patch size without overlap. Then, it is reshaped to (batch, (hxw), (patch x patch x channel)). 

### Text:
Also, original text is tokenized and embedded with BERT-based approach (BERT-base-uncased).

### Data processing:
When we train our model, we randomly samples(50 %) reports to make the matched- and un-matched image-text set.
Basically, matched and un-matched set is classified with label information using chexpert labeler, we consider unmatched set when randomly sampled report is not exactly same with original one.


Mixer based approach is trained efficiently with xxxx throuput with xxx accuracy.


### Chest X-ray Image-reports retrieval
Model spec:
patch size:16, embedding dim: 768

Input spec:
img size: 224x224x3 -> pathch size: (224/16) x (224/16)
text max len: 128 legth

input embedding: cls, txt, sep, img 
output: matched or unmatched



### Results

Max memory: 15.903G

Model forward time:
Model backward time:
