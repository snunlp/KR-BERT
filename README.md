## Korean based Bert pre-trained (K-BERT)

Korean-specific, small-scale, with comparable or better performances


### Vocab, Parameters and Data

|                |                              Mulitlingual BERT<br>(Google) |                KorBERT<br>(ETRI) |                              KoBERT<br>(SKT) |                       K-BERT character |                   K-BERT sub-character |
| -------------: | ---------------------------------------------: | ---------------------: | ----------------------------------: | -------------------------------------: | -------------------------------------: |
|     vocab size |                                        119,547 |                 30,797 |                               8,002 |                                 16,424 |                                 12,367 |
| parameter size |                                    167,356,416 |            109,973,391 |                          92,186,880 |                             99,265,066 |                             96,145,233 |
|      data size | -<br>(The Wikipedia data<br>for 104 languages) | 23GB<br>4.7B morphemes | -<br>(25M sentences,<br>233M words) | 2.47GB<br>20M sentences,<br>233M words | 2.47GB<br>20M sentences,<br>233M words |

| Model                                       | Masked LM Accuracy |
| ------------------------------------------- | ------------------ |
| KoBERT                                      | 0.750              |
| K-BERT character BidirectionalWordPiece     | **0.779**              |
| K-BERT sub-character BidirectionalWordPiece | 0.769              |

### Sub-character

Korean text is basically represented with Hangul syllable characters, which can be decomposed into sub-characters, or graphemes. To accommodate such characteristics, we trained a new vocabulary and BERT model on two different representations of a corpus: syllable characters and sub-characters.

In case of using our sub-character model, you should preprocess your data with the code below.

```
import torch
from transformers import BertConfig, BertModel, BertForPreTraining, BertTokenizer
from unicodedata import normalize

tokenizer_kbert = BertTokenizer.from_pretrained('/path/to/vocab_file.txt', do_lower_case=False)

# convert a string into sub-char
def to_subchar(string):
    return normalize('NFKD', string)

sentence = '토크나이저 예시입니다.'
print(tokenizer_kbert.tokenize(to_subchar(sentence)))

```

### Tokenization

#### BidirectionalWordPiece Tokenizer

We use the BidirectionalWordPiece model to reduce search costs while maintaining the possibility of choice. This model applies BPE in both forward and backward directions to obtain two candidates and chooses the one that has a higher frequency.


|                                         |     Mulitlingual BERT     |   KorBERT<br>character   |          KoBERT           | K-BERT<br>character<br>WordPiece | K-BERT<br>character<br>BidirectionalWordPiece | K-BERT<br>sub-character<br>WordPiece | K-BERT<br>sub-character<br>BidirectionalWordPiece |
| :-------------------------------------: | :-----------------------: | :-----------------------: | :-----------------------: | :------------------------------: | :-------------------------------------------: | :----------------------------------: | :-----------------------------------------------: |
| 냉장고<br>nayngcangko<br>"refrigerator" | 냉#장#고<br>nayng#cang#ko | 냉#장#고<br>nayng#cang#ko | 냉#장#고<br>nayng#cang#ko |      냉장고<br>nayngcangko       |             냉장고<br>nayngcangko             |        냉장고<br>nayngcangko         |               냉장고<br>nayngcangko               |
|        춥다<br>chwupta<br>"cold"        |           [UNK]           |     춥#다<br>chwup#ta     |     춥#다<br>chwup#ta     |        춥#다<br>chwup#ta         |               춥#다<br>chwup#ta               |         추#ㅂ다<br>chwu#pta          |                추#ㅂ다<br>chwu#pta                |
|     뱃사람<br>paytsalam<br>"seaman"     |           [UNK]           |   뱃#사람<br>payt#salam   |   뱃#사람<br>payt#salam   |      뱃#사람<br>payt#salam       |             뱃#사람<br>payt#salam             |      배#ㅅ#사람<br>pay#t#salam       |             배#ㅅ#사람<br>pay#t#salam             |
|    마이크<br>maikhu<br>"microphone"     |   마#이#크<br>ma#i#khu    |    마이#크<br>mai#khu     |   마#이#크<br>ma#i#khu    |         마이크<br>maikhu         |               마이크<br>maikhu                |           마이크<br>maikhu           |                 마이크<br>maikhu                  |

<br>

### models

##### tensorflow

* BERT tokenizer, character model [download](https://drive.google.com/open?id=1SG5m-3R395VjEEnt0wxWM7SE1j6ndVsX)
* BidirectionalWordPiece tokenizer, character model [download](https://drive.google.com/open?id=1YhFobehwzdbIxsHHvyFU5okp-HRowRKS)
* BERT tokenizer, sub-character model [download](https://drive.google.com/open?id=13oguhQvYD9wsyLwKgU-uLCacQVWA4oHg)
* BidirectionalWordPiece tokenizer, sub-character model [download](https://drive.google.com/open?id=12izU0NZXNz9I6IsnknUbencgr7gWHDeM)


#### Requirements

- transformers == 2.1.1
- tensorflow < 2.0

<br>

## Downstream tasks

### Naver Sentiment Movie Corpus (NSMC)

If you want to use the sub-character version of our models, let the *subchar* argument be True.
And you can use the original BERT WordPiece tokenizer by entering 'bert' for the *tokenizer* argument, and if you use 'rank(ed)' you can use our BidirectionalWordPiece tokenizer.

```
# pytorch
python3 train.py --subchar {True, False} --tokenizer {bert, rank}

# tensorflow
python3 run_classifier.py \
  --task_name=NSMC \
  --subchar={True, False} \
  --tokenizer={bert, ranked} \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --do_lower_case=False\
  --max_seq_length=128 \
  --train_batch_size=128 \
  --learning_rate=5e-05 \
  --num_train_epochs=5.0 \
  --output_dir={output_dir}
```
<br>

### NSMC Acc.

|       | multilingual BERT | KorBERT | KoBERT | K-BERT character WordPiece | K-BERT<br>character Bidirectional WordPiece | K-BERT sub-character WordPiece | K-BERT<br>sub-character Bidirectional WordPiece |
|:-----:|-------------------:|----------------:|--------:|----------------------------:|-----------------------------------------:|--------------------------------:|---------------------------------------------:|
| pytorch |              | **89.84**   | 89.01  | 89.34                      | **89.38**                                   | 89.20                          | 89.34                                       |
| tensorflow | 	87.08		|	85.94  |   n/a |  89.86 | **90.10** | 89.76 | 89.86 |


<br>

## Contacts

nlp.snu@gmail.com




