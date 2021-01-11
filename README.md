## KoRean based Bert pre-trained (KR-BERT)

This is a release of Korean-specific, small-scale BERT models with comparable or better performances developed by Computational Linguistics Lab at Seoul National University, referenced in [KR-BERT: A Small-Scale Korean-Specific Language Model](https://arxiv.org/abs/2008.03979).

<br>

### Vocab, Parameters and Data

|                |                              Mulitlingual BERT<br>(Google) |                KorBERT<br>(ETRI) |                              KoBERT<br>(SKT) |                       KR-BERT character |                   KR-BERT sub-character |
| -------------: | ---------------------------------------------: | ---------------------: | ----------------------------------: | -------------------------------------: | -------------------------------------: |
|     vocab size |                                        119,547 |                 30,797 |                               8,002 |                                 16,424 |                                 12,367 |
| parameter size |                                    167,356,416 |            109,973,391 |                          92,186,880 |                             99,265,066 |                             96,145,233 |
|      data size | -<br>(The Wikipedia data<br>for 104 languages) | 23GB<br>4.7B morphemes | -<br>(25M sentences,<br>233M words) | 2.47GB<br>20M sentences,<br>233M words | 2.47GB<br>20M sentences,<br>233M words |

| Model                                       | Masked LM Accuracy |
| ------------------------------------------- | ------------------ |
| KoBERT                                      | 0.750              |
| KR-BERT character BidirectionalWordPiece     | **0.779**              |
| KR-BERT sub-character BidirectionalWordPiece | 0.769              |

<br>

### Sub-character

Korean text is basically represented with Hangul syllable characters, which can be decomposed into sub-characters, or graphemes. To accommodate such characteristics, we trained a new vocabulary and BERT model on two different representations of a corpus: syllable characters and sub-characters.

In case of using our sub-character model, you should preprocess your data with the code below.

```python
import torch
from transformers import BertConfig, BertModel, BertForPreTraining, BertTokenizer
from unicodedata import normalize

tokenizer_krbert = BertTokenizer.from_pretrained('/path/to/vocab_file.txt', do_lower_case=False)

# convert a string into sub-char
def to_subchar(string):
    return normalize('NFKD', string)

sentence = '토크나이저 예시입니다.'
print(tokenizer_krbert.tokenize(to_subchar(sentence)))

```

### Tokenization

#### BidirectionalWordPiece Tokenizer

We use the BidirectionalWordPiece model to reduce search costs while maintaining the possibility of choice. This model applies BPE in both forward and backward directions to obtain two candidates and chooses the one that has a higher frequency.


|                                         |     Mulitlingual BERT     |   KorBERT<br>character   |          KoBERT           | KR-BERT<br>character<br>WordPiece | KR-BERT<br>character<br>BidirectionalWordPiece | KR-BERT<br>sub-character<br>WordPiece | KR-BERT<br>sub-character<br>BidirectionalWordPiece |
| :-------------------------------------: | :-----------------------: | :-----------------------: | :-----------------------: | :------------------------------: | :-------------------------------------------: | :----------------------------------: | :-----------------------------------------------: |
| 냉장고<br>nayngcangko<br>"refrigerator" | 냉#장#고<br>nayng#cang#ko | 냉#장#고<br>nayng#cang#ko | 냉#장#고<br>nayng#cang#ko |      냉장고<br>nayngcangko       |             냉장고<br>nayngcangko             |        냉장고<br>nayngcangko         |               냉장고<br>nayngcangko               |
|        춥다<br>chwupta<br>"cold"        |           [UNK]           |     춥#다<br>chwup#ta     |     춥#다<br>chwup#ta     |        춥#다<br>chwup#ta         |               춥#다<br>chwup#ta               |         추#ㅂ다<br>chwu#pta          |                추#ㅂ다<br>chwu#pta                |
|     뱃사람<br>paytsalam<br>"seaman"     |           [UNK]           |   뱃#사람<br>payt#salam   |   뱃#사람<br>payt#salam   |      뱃#사람<br>payt#salam       |             뱃#사람<br>payt#salam             |      배#ㅅ#사람<br>pay#t#salam       |             배#ㅅ#사람<br>pay#t#salam             |
|    마이크<br>maikhu<br>"microphone"     |   마#이#크<br>ma#i#khu    |    마이#크<br>mai#khu     |   마#이#크<br>ma#i#khu    |         마이크<br>maikhu         |               마이크<br>maikhu                |           마이크<br>maikhu           |                 마이크<br>maikhu                  |

<br>


### Models

|     | TensorFlow | | PyTorch | |
|:---:|:-------------------------------:|:----------------------------:|:----------------------------:|:----------------------------:|
|     | character | sub-character | character | sub-character |
| WordPiece <br> tokenizer | [WP char](https://drive.google.com/open?id=1SG5m-3R395VjEEnt0wxWM7SE1j6ndVsX) | [WP subchar](https://drive.google.com/open?id=13oguhQvYD9wsyLwKgU-uLCacQVWA4oHg) | [WP char](https://drive.google.com/file/d/18lsZzx_wonnOezzB5QxqSliA2KL5BF0x/view?usp=sharing) | [WP subchar](https://drive.google.com/open?id=1c1en4AMlCv2k7QapIzqjefnYzNOoh5KZ)
| Bidirectional <br> WordPiece <br> tokenizer | [BiWP char](https://drive.google.com/open?id=1YhFobehwzdbIxsHHvyFU5okp-HRowRKS) | [BiWP subchar](https://drive.google.com/open?id=12izU0NZXNz9I6IsnknUbencgr7gWHDeM) | [BiWP char](https://drive.google.com/open?id=1C87CCHD9lOQhdgWPkMw_6ZD5M2km7f1p) | [BiWP subchar](https://drive.google.com/file/d/1JvNYFQyb20SWgOiDxZn6h1-n_fjTU25S/view?usp=sharing)

<!-- 
#### tensorflow

* BERT tokenizer, character model ([download](https://drive.google.com/open?id=1SG5m-3R395VjEEnt0wxWM7SE1j6ndVsX))
* BidirectionalWordPiece tokenizer, character model ([download](https://drive.google.com/open?id=1YhFobehwzdbIxsHHvyFU5okp-HRowRKS))
* BERT tokenizer, sub-character model ([download](https://drive.google.com/open?id=13oguhQvYD9wsyLwKgU-uLCacQVWA4oHg))
* BidirectionalWordPiece tokenizer, sub-character model ([download](https://drive.google.com/open?id=12izU0NZXNz9I6IsnknUbencgr7gWHDeM))


#### pytorch

* BERT tokenizer, character model ([download](https://drive.google.com/file/d/18lsZzx_wonnOezzB5QxqSliA2KL5BF0x/view?usp=sharing))
* BidirectionalWordPiece tokenizer, character model ([download](https://drive.google.com/open?id=1C87CCHD9lOQhdgWPkMw_6ZD5M2km7f1p))
* BERT tokenizer, sub-character model ([download](https://drive.google.com/open?id=1c1en4AMlCv2k7QapIzqjefnYzNOoh5KZ))
* BidirectionalWordPiece tokenizer, sub-character model ([download](https://drive.google.com/file/d/1JvNYFQyb20SWgOiDxZn6h1-n_fjTU25S/view?usp=sharing))
 -->

<br>

### Requirements

- transformers == 2.1.1
- tensorflow < 2.0

<br>

## Downstream tasks

### Naver Sentiment Movie Corpus (NSMC)

* If you want to use the sub-character version of our models, let the `subchar` argument be `True`.
* And you can use the original BERT WordPiece tokenizer by entering `bert` for the `tokenizer` argument, and if you use `ranked` you can use our BidirectionalWordPiece tokenizer.

* tensorflow: After downloading our pretrained models, put them in a `models` directory in the `krbert_tensorflow` directory.

* pytorch: After downloading our pretrained models, put them in a `pretrained` directory in the `krbert_pytorch` directory.


```sh
# pytorch
python3 train.py --subchar {True, False} --tokenizer {bert, ranked}

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

The pytorch code structure refers to that of https://github.com/aisolab/nlp_implementation .

<br>

### NSMC Acc.

|       | multilingual BERT | KorBERT | KoBERT | KR-BERT character WordPiece | KR-BERT<br>character Bidirectional WordPiece | KR-BERT sub-character WordPiece | KR-BERT<br>sub-character Bidirectional WordPiece |
|:-----:|-------------------:|----------------:|--------:|----------------------------:|-----------------------------------------:|--------------------------------:|---------------------------------------------:|
| pytorch |        -      | **89.84**   | 89.01  | 89.34                      | **89.38**                                   | 89.20                          | 89.34                                       |
| tensorflow | 	87.08		|	85.94  |   n/a |  89.86 | **90.10** | 89.76 | 89.86 |


<br>


## Citation

If you use these models, please cite the following paper:
```
@article{lee2020krbert,
    title={KR-BERT: A Small-Scale Korean-Specific Language Model},
    author={Sangah Lee and Hansol Jang and Yunmee Baik and Suzi Park and Hyopil Shin},
    year={2020},
    journal={ArXiv},
    volume={abs/2008.03979}
  }
```

<br>

## Contacts

nlp.snu@gmail.com
